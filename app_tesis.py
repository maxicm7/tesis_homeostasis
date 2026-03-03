# ============================================================================
# 🎓 TESIS DOCTORAL: Modelo DCC-GARCH Homeostático con EVT (Gumbel)
# ============================================================================
# Archivo: app.py
# Ejecutar: streamlit run app.py
# ============================================================================

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import gumbel_r, norm
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="DCC-GARCH Homeostático - Tesis Doctoral",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# 🎨 ESTILOS CSS PERSONALIZADOS
# ============================================================================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #2c3e50;}
    .metric-card {background-color: #f8f9fa; padding: 20px; border-radius: 10px; 
                  border-left: 5px solid #1f77b4;}
    .warning-box {background-color: #fff3cd; padding: 15px; border-radius: 5px; 
                  border-left: 5px solid #ffc107;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 5px; 
                  border-left: 5px solid #28a745;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 📦 FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

@st.cache_data
def download_data(tickers, start_date, end_date):
    """Descarga datos desde Yahoo Finance"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, 
                          progress=False, auto_adjust=True)
        
        # Manejar estructura de columnas de yfinance
        if len(tickers) == 1:
            data = pd.DataFrame(data)
        
        # Extraer precios de cierre ajustados
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close']
        else:
            prices = data
        
        # Eliminar filas con muchos NaN
        prices = prices.dropna(how='all')
        
        # Forward fill para días festivos diferentes entre mercados
        prices = prices.ffill()
        
        return prices
    except Exception as e:
        st.error(f"Error descargando datos: {str(e)}")
        return None

def calculate_returns(prices):
    """Calcula retornos logarítmicos"""
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

# ============================================================================
# 📈 MODELO GARCH UNIVARIADO (SIMPLIFICADO)
# ============================================================================

def garch_filter(returns, p=1, q=1):
    """
    Filtrado GARCH(1,1) simplificado para residuos estandarizados
    Nota: Para producción usar librería 'arch'
    """
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = returns.var()
    
    # Parámetros GARCH típicos (estimados)
    omega = 0.00001
    alpha = 0.1
    beta = 0.85
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
    
    sigma = np.sqrt(sigma2)
    z_std = returns / sigma
    
    return z_std, sigma

# ============================================================================
# 🎯 DISTRIBUCIÓN DE GUMBEL Y UMBRALES
# ============================================================================

def fit_gumbel_threshold(residuals, confidence=0.95, window=252):
    """
    Ajusta distribución de Gumbel y calcula umbrales de tensión homeostática
    """
    thresholds = {}
    indicators = pd.DataFrame(index=residuals.index)
    
    for col in residuals.columns:
        # Usar ventana móvil para parámetros de Gumbel
        locs = []
        scales = []
        
        for t in range(window, len(residuals)):
            window_data = np.abs(residuals[col].iloc[t-window:t])
            loc, scale = gumbel_r.fit(window_data)
            locs.append(loc)
            scales.append(scale)
        
        # Promedio de parámetros
        avg_loc = np.mean(locs) if locs else gumbel_r.fit(np.abs(residuals[col]))[0]
        avg_scale = np.mean(scales) if scales else gumbel_r.fit(np.abs(residuals[col]))[1]
        
        # Umbral crítico (valor extremo)
        threshold = gumbel_r.ppf(confidence, loc=avg_loc, scale=avg_scale)
        thresholds[col] = threshold
        
        # Indicador binario
        indicators[col] = (np.abs(residuals[col]) > threshold).astype(int)
    
    return thresholds, indicators

def calculate_systemic_indicator(indicators, kappa=0.3):
    """
    Calcula indicador sistémico H_t
    H_t = 1 si >= kappa proporción de activos en tensión
    """
    prop_stressed = indicators.mean(axis=1)
    H_t = (prop_stressed >= kappa).astype(int)
    return H_t, prop_stressed

# ============================================================================
# 🔗 MODELO DCC-GARCH HOMEOSTÁTICO
# ============================================================================

def dcc_homeostatic(z_std, H_indicator, Q_bar=None):
    """
    Implementación simplificada del DCC-GARCH Homeostático
    z_std: residuos estandarizados
    H_indicator: indicador de tensión homeostática
    """
    T = len(z_std)
    N = z_std.shape[1]
    
    if Q_bar is None:
        Q_bar = np.corrcoef(z_std.T)
    
    # Parámetros DCC (estimados o fijados)
    a = 0.05
    b = 0.90
    gamma = 0.15  # Parámetro homeostático
    
    # Matriz de correlación de estrés (estimada de periodos H_t=1)
    stress_periods = z_std[H_indicator == 1]
    if len(stress_periods) > 10:
        Q_stress = np.corrcoef(stress_periods.T)
    else:
        Q_stress = Q_bar
    
    # Evolución de Q_t
    Q_t = np.zeros((T, N, N))
    R_t = np.zeros((T, N, N))
    Q_t[0] = Q_bar
    
    for t in range(1, T):
        if H_indicator.iloc[t-1] == 1:
            # Régimen homeostático activo
            Q_t[t] = (1 - a - b - gamma) * Q_bar + \
                     a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                     b * Q_t[t-1] + \
                     gamma * Q_stress
        else:
            # Régimen normal
            Q_t[t] = (1 - a - b) * Q_bar + \
                     a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                     b * Q_t[t-1]
        
        # Normalizar a matriz de correlación
        D_inv = np.diag(1 / np.sqrt(np.diag(Q_t[t])))
        R_t[t] = D_inv @ Q_t[t] @ D_inv
    
    return R_t, Q_t, {'a': a, 'b': b, 'gamma': gamma}

# ============================================================================
# ⚠️ CÁLCULO DE VaR Y BACKTESTING
# ============================================================================

def calculate_var(returns, R_t, weights=None, confidence=0.95):
    """Calcula Value-at-Risk condicional"""
    T = returns.shape[0]
    N = returns.shape[1]
    
    if weights is None:
        weights = np.ones(N) / N
    
    var_series = np.zeros(T)
    
    for t in range(T):
        # Varianza del portafolio
        sigma2_p = weights @ R_t[t] @ weights * returns.iloc[t].var()
        sigma_p = np.sqrt(sigma2_p)
        
        # VaR (asumiendo distribución t-Student o normal)
        z_score = norm.ppf(1 - confidence)
        var_series[t] = -sigma_p * z_score
    
    return var_series

def backtest_var(returns, var_series, confidence=0.95):
    """Backtesting de VaR (Kupiec Test)"""
    portfolio_return = returns.mean(axis=1)  # Retorno de portafolio equally weighted
    violations = (portfolio_return < -var_series).astype(int)
    
    n_violations = violations.sum()
    n_observations = len(violations)
    expected_violations = n_observations * (1 - confidence)
    
    # Kupiec POF Test
    p_hat = n_violations / n_observations
    p = 1 - confidence
    
    if p_hat > 0 and p_hat < 1:
        lr_stat = -2 * (n_observations * np.log(1-p) + n_violations * np.log(p/(1-p)) -
                       n_observations * np.log(1-p_hat) - n_violations * np.log(p_hat/(1-p_hat)))
    else:
        lr_stat = np.inf
    
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr_stat, 1)
    
    return {
        'violations': n_violations,
        'expected': expected_violations,
        'violation_rate': n_violations / n_observations,
        'expected_rate': 1 - confidence,
        'kupiec_lr': lr_stat,
        'kupiec_pvalue': p_value,
        'passed': p_value > 0.05
    }

# ============================================================================
# 📊 VISUALIZACIONES
# ============================================================================

def plot_correlation_heatmap(R_t, dates, tickers, title="Matriz de Correlación"):
    """Heatmap de correlaciones"""
    avg_corr = np.mean(R_t[-60:], axis=0)  # Últimos 60 días
    
    fig = go.Figure(data=go.Heatmap(
        z=avg_corr,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(avg_corr, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Activos",
        yaxis_title="Activos",
        height=500
    )
    
    return fig

def plot_correlation_timeseries(R_t, dates, tickers, pair=(0, 1)):
    """Serie temporal de correlación entre dos activos"""
    corr_series = [R_t[t][pair[0], pair[1]] for t in range(len(R_t))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=corr_series,
        mode='lines',
        name=f"{tickers[pair[0]]} - {tickers[pair[1]]}",
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"Correlación Dinámica: {tickers[pair[0]]} vs {tickers[pair[1]]}",
        xaxis_title="Fecha",
        yaxis_title="Correlación",
        yaxis=dict(range=[-1, 1]),
        height=400
    )
    
    return fig

def plot_homeostatic_indicator(H_t, prop_stressed, dates):
    """Gráfico del indicador homeostático"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, row_heights=[0.3, 0.7])
    
    # Proporción de activos en tensión
    fig.add_trace(go.Scatter(
        x=dates, y=prop_stressed,
        mode='lines', name='Proporción en Tensión',
        line=dict(color='#ff7f0e', width=2)
    ), row=1, col=1)
    
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                  annotation_text="Umbral κ=0.3", row=1, col=1)
    
    # Indicador H_t
    fig.add_trace(go.Scatter(
        x=dates, y=H_t,
        mode='lines', name='H_t (Homeostasis Activa)',
        line=dict(color='#2ca02c', width=3),
        fill='tozeroy'
    ), row=2, col=1)
    
    fig.update_layout(
        title="🏠 Indicador de Tensión Homeostática del Sistema",
        height=500,
        showlegend=True
    )
    
    return fig

def plot_var_backtesting(returns, var_series, dates):
    """Gráfico de VaR vs Retornos reales"""
    portfolio_return = returns.mean(axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_return,
        mode='lines', name='Retorno Portafolio',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=-var_series,
        mode='lines', name='VaR (95%)',
        line=dict(color='#d62728', width=2, dash='dash')
    ))
    
    # Marcar violaciones
    violations = portfolio_return < -var_series
    violation_dates = dates[violations]
    violation_values = portfolio_return[violations]
    
    fig.add_trace(go.Scatter(
        x=violation_dates, y=violation_values,
        mode='markers', name='Violaciones VaR',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title="⚠️ Backtesting de Value-at-Risk",
        xaxis_title="Fecha",
        yaxis_title="Retorno",
        height=400
    )
    
    return fig

# ============================================================================
# 🖥️ INTERFAZ STREAMLIT
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🎓 Modelo DCC-GARCH Homeostático con EVT</p>', 
                unsafe_allow_html=True)
    st.markdown("**Tesis Doctoral en Economía Financiera** | Detección de Regímenes de Corrección Homeostática")
    st.markdown("---")
    
    # Sidebar - Configuración
    st.sidebar.header("⚙️ Configuración del Modelo")
    
    # Selección de tickers
    st.sidebar.subheader("1. Selección de Activos")
    
    preset portfolios
    portfolio_choice = st.sidebar.selectbox(
        "Portafolio Predefinido",
        ["Mínimo (6 activos)", "Completo (12 activos)", "Personalizado"]
    )
    
    TICKERS_MINIMUM = ['^GSPC', '^STOXX50E', 'TLT', 'GLD', 'UUP', 'EEM']
    TICKERS_COMPLETE = ['^GSPC', '^STOXX50E', '^N225', '^VIX', 'TLT', 'HYG', 
                        'GLD', 'USO', 'FXE', 'UUP', 'EEM', 'BTC-USD']
    
    if portfolio_choice == "Mínimo (6 activos)":
        default_tickers = TICKERS_MINIMUM
    elif portfolio_choice == "Completo (12 activos)":
        default_tickers = TICKERS_COMPLETE
    else:
        default_tickers = TICKERS_MINIMUM
    
    tickers_input = st.sidebar.text_area(
        "Tickers (separados por coma)",
        value=", ".join(default_tickers),
        help="Ejemplo: ^GSPC, TLT, GLD, UUP"
    )
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    
    # Rango de fechas
    st.sidebar.subheader("2. Período de Análisis")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Inicio", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("Fin", value=datetime.now())
    
    # Parámetros del modelo
    st.sidebar.subheader("3. Parámetros del Modelo")
    confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.90, 0.99, 0.95, 0.01)
    kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.1, 0.5, 0.3, 0.05)
    var_confidence = st.sidebar.slider("Confianza VaR", 0.90, 0.99, 0.95, 0.01)
    garch_window = st.sidebar.slider("Ventana GARCH (días)", 60, 500, 252)
    
    # Botón de ejecución
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        with st.spinner("Descargando datos y ejecutando modelo..."):
            
            # 1. Descarga de datos
            st.markdown('<p class="sub-header">📥 1. Descarga de Datos</p>', 
                       unsafe_allow_html=True)
            
            prices = download_data(tickers, start_date, end_date)
            
            if prices is None or prices.empty:
                st.error("No se pudieron descargar los datos. Verifica los tickers.")
                st.stop()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Activos", len(tickers))
            with col2:
                st.metric("Período", f"{prices.index[0].strftime('%Y-%m-%d')} a {prices.index[-1].strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Observaciones", len(prices))
            
            # Mostrar datos
            with st.expander("📋 Ver Datos de Precios"):
                st.dataframe(prices.tail(10))
            
            # 2. Cálculo de retornos
            st.markdown("---")
            st.markdown('<p class="sub-header">📈 2. Cálculo de Retornos y Filtrado GARCH</p>', 
                       unsafe_allow_html=True)
            
            returns = calculate_returns(prices)
            z_std, sigma = garch_filter(returns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Retorno Medio Anual", f"{returns.mean().mean()*252:.2%}")
            with col2:
                st.metric("Volatilidad Anual", f"{returns.std().std()*np.sqrt(252):.2%}")
            
            # 3. Modelo de Valores Extremos (Gumbel)
            st.markdown("---")
            st.markdown('<p class="sub-header">🎯 3. Distribución de Gumbel y Umbrales</p>', 
                       unsafe_allow_html=True)
            
            thresholds, indicators = fit_gumbel_threshold(z_std, confidence_gumbel, garch_window)
            H_t, prop_stressed = calculate_systemic_indicator(indicators, kappa_threshold)
            
            # Mostrar umbrales
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 Umbrales por Activo (Gumbel)**")
                threshold_df = pd.DataFrame({
                    'Ticker': list(thresholds.keys()),
                    'Umbral': list(thresholds.values())
                })
                st.dataframe(threshold_df.style.format({'Umbral': '{:.4f}'}))
            
            with col2:
                st.markdown("**📊 Estadísticas de H_t**")
                st.metric("Días en Homeostasis", int(H_t.sum()))
                st.metric("Porcentaje del Tiempo", f"{H_t.mean()*100:.1f}%")
            
            # Gráfico de indicador homeostático
            st.plotly_chart(
                plot_homeostatic_indicator(H_t.values, prop_stressed.values, 
                                          prop_stressed.index),
                use_container_width=True
            )
            
            # 4. Modelo DCC-GARCH Homeostático
            st.markdown("---")
            st.markdown('<p class="sub-header">🔗 4. Correlación Dinámica (DCC-H)</p>', 
                       unsafe_allow_html=True)
            
            R_t, Q_t, params = dcc_homeostatic(z_std, H_t)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parámetro α (shock)", f"{params['a']:.3f}")
            with col2:
                st.metric("Parámetro β (persistencia)", f"{params['b']:.3f}")
            with col3:
                st.metric("Parámetro γ (homeostasis)", f"{params['gamma']:.3f}")
            
            # Heatmap de correlaciones
            st.plotly_chart(
                plot_correlation_heatmap(R_t, returns.index, tickers, 
                                        "Matriz de Correlación Promedio (Últimos 60 días)"),
                use_container_width=True
            )
            
            # Selector de par para serie temporal de correlación
            st.markdown("**Seleccionar par de activos para ver evolución de correlación:**")
            col1, col2 = st.columns(2)
            with col1:
                asset1 = st.selectbox("Activo 1", tickers, index=0)
            with col2:
                asset2 = st.selectbox("Activo 2", tickers, index=1 if len(tickers) > 1 else 0)
            
            idx1 = tickers.index(asset1)
            idx2 = tickers.index(asset2)
            
            st.plotly_chart(
                plot_correlation_timeseries(R_t, returns.index, tickers, (idx1, idx2)),
                use_container_width=True
            )
            
            # 5. Value-at-Risk y Backtesting
            st.markdown("---")
            st.markdown('<p class="sub-header">⚠️ 5. Value-at-Risk y Backtesting</p>', 
                       unsafe_allow_html=True)
            
            var_series = calculate_var(returns, R_t, confidence=var_confidence)
            backtest_results = backtest_var(returns, var_series, var_confidence)
            
            # Métricas de backtesting
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Violaciones Observadas", backtest_results['violations'])
            with col2:
                st.metric("Violaciones Esperadas", f"{backtest_results['expected']:.1f}")
            with col3:
                st.metric("Tasa Observada", f"{backtest_results['violation_rate']*100:.2f}%")
            with col4:
                st.metric("Tasa Esperada", f"{backtest_results['expected_rate']*100:.2f}%")
            
            # Resultado del test Kupiec
            if backtest_results['passed']:
                st.success(f"✅ Test de Kupiec APROBADO (p-value: {backtest_results['kupiec_pvalue']:.4f})")
            else:
                st.error(f"❌ Test de Kupiec RECHAZADO (p-value: {backtest_results['kupiec_pvalue']:.4f})")
            
            # Gráfico de VaR
            st.plotly_chart(
                plot_var_backtesting(returns, var_series, returns.index),
                use_container_width=True
            )
            
            # 6. Exportar resultados
            st.markdown("---")
            st.markdown('<p class="sub-header">💾 6. Exportar Resultados</p>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Exportar series temporales
                results_df = pd.DataFrame({
                    'Date': returns.index,
                    'H_Indicator': H_t.values,
                    'Prop_Stressed': prop_stressed.values,
                    'VaR': var_series
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Series Temporales (CSV)",
                    data=csv,
                    file_name=f"dcc_homeostatic_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Exportar resumen
                summary = {
                    'Modelo': 'DCC-GARCH Homeostático',
                    'Activos': len(tickers),
                    'Período': f"{start_date} a {end_date}",
                    'Confianza Gumbel': confidence_gumbel,
                    'Umbral κ': kappa_threshold,
                    'Días Homeostasis': int(H_t.sum()),
                    'Violaciones VaR': backtest_results['violations'],
                    'Kupiec p-value': backtest_results['kupiec_pvalue']
                }
                
                summary_df = pd.DataFrame(summary, index=['Valor'])
                csv_summary = summary_df.to_csv()
                st.download_button(
                    label="📥 Descargar Resumen del Modelo (CSV)",
                    data=csv_summary,
                    file_name=f"dcc_homeostatic_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    else:
        # Pantalla de bienvenida
        st.markdown("""
        ### 🎯 Bienvenido a la Aplicación del Modelo DCC-GARCH Homeostático
        
        Esta herramienta implementa el modelo desarrollado para la tesis doctoral:
        **"Dinámica de Corrección Homeostática en Mercados Financieros"**
        
        #### ¿Qué hace este modelo?
        
        1. **📊 Filtrado GARCH**: Extrae residuos estandarizados de los retornos
        2. **🎯 Teoría de Valores Extremos**: Ajusta distribución de Gumbel para detectar eventos extremos
        3. **🏠 Indicador Homeostático**: Identifica cuando el sistema está en "tensión" (H_t = 1)
        4. **🔗 DCC Modificado**: La correlación dinámica cambia según el régimen homeostático
        5. **⚠️ VaR Condicional**: Calcula riesgo ajustado al estado del sistema
        
        #### Hipótesis que se pueden testear:
        
        - **H1**: Los umbrales de Gumbel predicen mejor los eventos extremos que la distribución normal
        - **H2**: Las correlaciones cambian significativamente cuando H_t = 1
        - **H3**: El VaR condicional tiene menos violaciones que modelos estándar
        
        ---
        
        <div class="warning-box">
        <strong>⚠️ Nota Académica:</strong> Esta aplicación es para fines de investigación. 
        Los resultados no constituyen recomendación de inversión. Para una tesis doctoral, 
        se recomienda validar con múltiples períodos y realizar pruebas de robustez.
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar tickers recomendados
        st.markdown('<p class="sub-header">📋 Tickers Recomendados para Investigación</p>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏛️ Índices y Equities**")
            st.code("^GSPC  - S&P 500 (US)")
            st.code("^STOXX50E - EURO STOXX 50")
            st.code("^N225 - Nikkei 225 (Japón)")
            st.code("EEM - Emerging Markets")
        
        with col2:
            st.markdown("**🛡️ Safe Havens**")
            st.code("TLT - Treasury Bonds 20+")
            st.code("GLD - Gold ETF")
            st.code("UUP - Dollar Index")
            st.code("VIX - Volatility Index")
        
        st.markdown("**📦 Commodities & Otros**")
        st.code("USO - Oil | HYG - High Yield Bonds | FXE - Euro | BTC-USD - Bitcoin")

# ============================================================================
# 🚀 EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
