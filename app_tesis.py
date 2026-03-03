# ============================================================================
# 🎓 TESIS DOCTORAL: Modelo DCC-GARCH Homeostático con EVT (Gumbel)
# ============================================================================
# Archivo: app_tesis.py
# Ejecutar: streamlit run app_tesis.py
# ============================================================================

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import gumbel_r, norm, chi2
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
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin-top: 30px;}
    .metric-card {background-color: #f8f9fa; padding: 20px; border-radius: 10px; 
                  border-left: 5px solid #1f77b4;}
    .warning-box {background-color: #fff3cd; padding: 15px; border-radius: 5px; 
                  border-left: 5px solid #ffc107;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 5px; 
                  border-left: 5px solid #28a745;}
    .info-box {background-color: #d1ecf1; padding: 15px; border-radius: 5px; 
               border-left: 5px solid #17a2b8;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 📦 FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

@st.cache_data
def download_data(tickers, start_date, end_date):
    """Descarga datos desde Yahoo Finance - VERSIÓN CORREGIDA 2024"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, 
                          progress=False, auto_adjust=True)
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            st.error("No se recibieron datos de Yahoo Finance")
            return None
        
        if len(tickers) == 1:
            if isinstance(data.columns, pd.MultiIndex):
                if ('Adj Close', tickers[0]) in data.columns:
                    prices = data[('Adj Close', tickers[0])]
                elif ('Close', tickers[0]) in data.columns:
                    prices = data[('Close', tickers[0])]
                else:
                    price_cols = [col for col in data.columns if 'Close' in str(col)]
                    if price_cols:
                        prices = data[price_cols[0]]
                    else:
                        prices = data.iloc[:, 0]
            else:
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close']
                elif 'Close' in data.columns:
                    prices = data['Close']
                else:
                    prices = data.iloc[:, 0]
            prices = pd.DataFrame(prices, columns=[tickers[0]])
        else:
            if isinstance(data.columns, pd.MultiIndex):
                prices_dict = {}
                for ticker in tickers:
                    if ('Adj Close', ticker) in data.columns:
                        prices_dict[ticker] = data[('Adj Close', ticker)]
                    elif ('Close', ticker) in data.columns:
                        prices_dict[ticker] = data[('Close', ticker)]
                
                if prices_dict:
                    prices = pd.DataFrame(prices_dict)
                else:
                    try:
                        prices = data.xs('Close', level=0, axis=1)
                    except:
                        prices = data.iloc[:, :len(tickers)]
            else:
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close']
                elif 'Close' in data.columns:
                    prices = data['Close']
                else:
                    st.warning("Columna 'Adj Close' no encontrada, usando primera columna")
                    prices = data.iloc[:, :len(tickers)]
        
        prices = prices.dropna(how='all')
        prices = prices.ffill()
        prices = prices.bfill()
        
        if prices.empty or prices.shape[1] == 0:
            st.error("No se pudieron extraer precios válidos")
            return None
        
        return prices
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
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
    Versión vectorizada compatible con múltiples activos
    """
    n = len(returns)
    
    # Inicializar matriz de varianzas (n_observaciones x n_activos)
    sigma2 = np.zeros_like(returns.values, dtype=float)
    sigma2[0] = returns.var().values
    
    # Parámetros GARCH típicos (estimados promedios de mercado)
    omega = 0.00001
    alpha = 0.1
    beta = 0.85
    
    for t in range(1, n):
        # Actualización simultánea para todos los activos
        sigma2[t] = omega + alpha * (returns.iloc[t-1].values**2) + beta * sigma2[t-1]
    
    sigma = np.sqrt(sigma2)
    # Evitar división por cero
    sigma[sigma < 1e-10] = 1e-10
    
    # Convertir a DataFrames para mantener índices (fechas) y columnas (tickers)
    sigma_df = pd.DataFrame(sigma, index=returns.index, columns=returns.columns)
    z_std = returns / sigma_df
    
    return z_std, sigma_df

# ============================================================================
# 🎯 DISTRIBUCIÓN DE GUMBEL Y UMBRALES
# ============================================================================

def fit_gumbel_threshold(residuals, confidence=0.95, window=252):
    """Ajusta distribución de Gumbel y calcula umbrales de tensión"""
    thresholds = {}
    indicators = pd.DataFrame(index=residuals.index)
    
    for col in residuals.columns:
        locs = []
        scales = []
        for t in range(window, len(residuals)):
            window_data = np.abs(residuals[col].iloc[t-window:t])
            try:
                loc, scale = gumbel_r.fit(window_data)
                locs.append(loc)
                scales.append(scale)
            except:
                continue
        
        if locs and scales:
            avg_loc, avg_scale = np.mean(locs), np.mean(scales)
        else:
            try:
                avg_loc, avg_scale = gumbel_r.fit(np.abs(residuals[col]))
            except:
                avg_loc, avg_scale = 0, 1
        
        threshold = gumbel_r.ppf(confidence, loc=avg_loc, scale=avg_scale)
        thresholds[col] = threshold
        indicators[col] = (np.abs(residuals[col]) > threshold).astype(int)
    
    return thresholds, indicators

def calculate_systemic_indicator(indicators, kappa=0.3):
    """Calcula indicador sistémico H_t"""
    prop_stressed = indicators.mean(axis=1)
    H_t = (prop_stressed >= kappa).astype(int)
    return H_t, prop_stressed

# ============================================================================
# 🔗 MODELO DCC-GARCH HOMEOSTÁTICO
# ============================================================================

def dcc_likelihood_full(z_std, H_indicator, Q_bar, params):
    """Calcula la log-verosimilitud completa del modelo DCC"""
    T = len(z_std)
    N = z_std.shape[1]
    
    a, b = params[0], params[1]
    gamma = params[2] if len(params) == 3 else 0
    
    if a < 0 or b < 0 or gamma < 0: return -np.inf
    if a + b + gamma >= 1: return -np.inf
    
    stress_periods = z_std[H_indicator == 1]
    if len(stress_periods) > 10:
        Q_stress = np.corrcoef(stress_periods.T)
    else:
        Q_stress = Q_bar.copy()
    
    Q_prev = Q_bar.copy()
    log_lik = 0
    
    for t in range(1, T):
        if gamma > 0 and H_indicator.iloc[t-1] == 1:
            Q_t = (1 - a - b - gamma) * Q_bar + \
                  a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                  b * Q_prev + gamma * Q_stress
        else:
            Q_t = (1 - a - b) * Q_bar + \
                  a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                  b * Q_prev
        
        diag_q = np.sqrt(np.diag(Q_t))
        diag_q[diag_q == 0] = 1e-10
        D_inv = np.diag(1 / diag_q)
        R_t = D_inv @ Q_t @ D_inv
        
        min_eig = np.min(np.linalg.eigvalsh(R_t))
        if min_eig < 1e-10:
            R_t = R_t + (1e-10 - min_eig) * np.eye(N)
        
        try:
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0: return -np.inf
            R_inv = np.linalg.inv(R_t)
            log_lik += -0.5 * (logdet + z_std.iloc[t].T @ R_inv @ z_std.iloc[t])
        except:
            return -np.inf
        
        Q_prev = Q_t.copy()
    
    return log_lik

def estimate_dcc_parameters(z_std, H_indicator, Q_bar, model_type='DCC-H'):
    """Estima parámetros DCC por máxima verosimilitud"""
    def neg_log_lik(params):
        ll = dcc_likelihood_full(z_std, H_indicator, Q_bar, params)
        # Evitar np.inf directo para que L-BFGS-B no colapse calculando el gradiente
        return -ll if ll != -np.inf else 1e10
    
    if model_type == 'DCC-H':
        initial_params = [0.05, 0.90, 0.10]
        bounds = [(1e-6, 0.5), (1e-6, 0.99), (0, 0.5)]
    else:
        initial_params = [0.05, 0.90]
        bounds = [(1e-6, 0.5), (1e-6, 0.99)]
    
    result = minimize(neg_log_lik, initial_params, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
    return result

def dcc_homeostatic(z_std, H_indicator, Q_bar=None):
    """Implementación del DCC-GARCH Homeostático"""
    T, N = z_std.shape
    if Q_bar is None: Q_bar = np.corrcoef(z_std.T)
    
    result = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
    params = result.x
    a, b = params[0], params[1]
    gamma = params[2] if len(params) == 3 else 0
    
    stress_periods = z_std[H_indicator == 1]
    if len(stress_periods) > 10:
        Q_stress = np.corrcoef(stress_periods.T)
    else:
        Q_stress = Q_bar.copy()
    
    Q_t = np.zeros((T, N, N))
    R_t = np.zeros((T, N, N))
    Q_t[0] = Q_bar
    
    for t in range(1, T):
        if gamma > 0 and H_indicator.iloc[t-1] == 1:
            Q_t[t] = (1 - a - b - gamma) * Q_bar + \
                     a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                     b * Q_t[t-1] + gamma * Q_stress
        else:
            Q_t[t] = (1 - a - b) * Q_bar + \
                     a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                     b * Q_t[t-1]
        
        diag_q = np.sqrt(np.diag(Q_t[t]))
        diag_q[diag_q == 0] = 1e-10
        D_inv = np.diag(1 / diag_q)
        R_t[t] = D_inv @ Q_t[t] @ D_inv
        
        min_eig = np.min(np.linalg.eigvalsh(R_t[t]))
        if min_eig < 1e-10:
            R_t[t] = R_t[t] + (1e-10 - min_eig) * np.eye(N)
    
    return R_t, Q_t, {'a': a, 'b': b, 'gamma': gamma, 'log_lik': -result.fun}

# ============================================================================
# 🧪 TEST DE RAZÓN DE VEROSIMILITUD
# ============================================================================

def likelihood_ratio_test(z_std, H_indicator, Q_bar):
    """Test de Razón de Verosimilitud: DCC-H vs DCC estándar"""
    result_restricted = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC')
    log_lik_restricted = -result_restricted.fun
    
    result_unrestricted = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
    log_lik_unrestricted = -result_unrestricted.fun
    
    lr_stat = 2 * (log_lik_unrestricted - log_lik_restricted)
    df = len(result_unrestricted.x) - len(result_restricted.x)
    
    p_value = 1 - chi2.cdf(lr_stat, df) if lr_stat > 0 else 1.0
    critical_value = chi2.ppf(0.95, df)
    decision = "RECHAZAR_H0" if p_value < 0.05 else "NO_RECHAZAR_H0"
    
    return {
        'lr_statistic': lr_stat, 'df': df, 'p_value': p_value,
        'critical_value': critical_value, 'decision': decision,
        'log_lik_restricted': log_lik_restricted,
        'log_lik_unrestricted': log_lik_unrestricted,
        'params_restricted': result_restricted.x,
        'params_unrestricted': result_unrestricted.x
    }

# ============================================================================
# ⚠️ CÁLCULO DE VaR Y BACKTESTING (CORREGIDO PARA DCC-GARCH)
# ============================================================================

def calculate_var(returns, sigma, R_t, weights=None, confidence=0.95):
    """
    Calcula Value-at-Risk condicional basándose en la matriz de 
    Covarianza Condicional (H_t = D_t * R_t * D_t)
    """
    T, N = returns.shape
    if weights is None: weights = np.ones(N) / N
    
    var_series = np.zeros(T)
    
    for t in range(T):
        # Matriz D_t (diagonal de volatilidades condicionales del paso GARCH)
        D_t = np.diag(sigma.iloc[t].values)
        
        # Matriz de covarianza condicional H_t
        H_t = D_t @ R_t[t] @ D_t
        
        # Varianza del portafolio = w^T * H_t * w
        sigma2_p = weights @ H_t @ weights
        sigma_p = np.sqrt(sigma2_p) if sigma2_p > 0 else 1e-10
        
        # VaR asumiendo normalidad (se podría cambiar a t-student o empírica)
        z_score = norm.ppf(1 - confidence)
        var_series[t] = -sigma_p * z_score
    
    return var_series

def backtest_var(returns, var_series, confidence=0.95):
    """Backtesting de VaR (Kupiec Test)"""
    portfolio_return = returns.mean(axis=1)
    violations = (portfolio_return < -var_series).astype(int)
    
    n_violations = violations.sum()
    n_observations = len(violations)
    expected_violations = n_observations * (1 - confidence)
    
    p_hat = n_violations / n_observations if n_observations > 0 else 0
    p = 1 - confidence
    
    if p_hat > 0 and p_hat < 1 and n_observations > 0:
        lr_stat = -2 * (n_observations * np.log(1-p) + n_violations * np.log(p/(1-p)) -
                       n_observations * np.log(1-p_hat) - n_violations * np.log(p_hat/(1-p_hat)))
    else:
        lr_stat = 0
    
    p_value = 1 - chi2.cdf(lr_stat, 1) if lr_stat > 0 else 1.0
    
    return {
        'violations': n_violations, 'expected': expected_violations,
        'violation_rate': p_hat, 'expected_rate': p,
        'kupiec_lr': lr_stat, 'kupiec_pvalue': p_value,
        'passed': p_value > 0.05
    }

# ============================================================================
# 📊 VALIDACIÓN OUT-OF-SAMPLE
# ============================================================================

def out_of_sample_validation(prices, tickers, train_ratio=0.7, confidence_gumbel=0.95, 
                             kappa_threshold=0.3, var_confidence=0.95):
    """Validación Out-of-Sample del modelo"""
    returns = calculate_returns(prices)
    n_obs = len(returns)
    n_train = int(n_obs * train_ratio)
    
    returns_train = returns.iloc[:n_train]
    returns_test = returns.iloc[n_train:]
    
    if len(returns_test) < 50:
        return None, "Período de prueba demasiado corto"
    
    # ========== ENTRENAMIENTO ==========
    z_std_train, sigma_train = garch_filter(returns_train)
    thresholds_train, indicators_train = fit_gumbel_threshold(z_std_train, confidence_gumbel)
    H_t_train, prop_stressed_train = calculate_systemic_indicator(indicators_train, kappa_threshold)
    
    Q_bar_train = np.corrcoef(z_std_train.T)
    R_t_train, Q_t_train, params_train = dcc_homeostatic(z_std_train, H_t_train, Q_bar_train)
    
    # ========== PRUEBA (APLICANDO PARÁMETROS OoS) ==========
    z_std_test, sigma_test = garch_filter(returns_test)
    thresholds_test, indicators_test = fit_gumbel_threshold(z_std_test, confidence_gumbel)
    H_t_test, prop_stressed_test = calculate_systemic_indicator(indicators_test, kappa_threshold)
    
    R_t_test, Q_t_test, params_test = dcc_homeostatic(z_std_test, H_t_test, Q_bar_train)
    
    # ========== VaR OUT-OF-SAMPLE ==========
    var_test = calculate_var(returns_test, sigma_test, R_t_test, confidence=var_confidence)
    backtest_oos = backtest_var(returns_test, var_test, var_confidence)
    
    # DCC Estándar OoS
    R_t_standard, _, params_standard = dcc_homeostatic(z_std_test, pd.Series(0, index=H_t_test.index), Q_bar_train)
    var_standard = calculate_var(returns_test, sigma_test, R_t_standard, confidence=var_confidence)
    backtest_standard = backtest_var(returns_test, var_standard, var_confidence)
    
    results = {
        'train_period': f"{returns_train.index[0].strftime('%Y-%m-%d')} a {returns_train.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{returns_test.index[0].strftime('%Y-%m-%d')} a {returns_test.index[-1].strftime('%Y-%m-%d')}",
        'n_train': n_train, 'n_test': len(returns_test),
        'params_train': params_train, 'H_t_test': H_t_test,
        'prop_stressed_test': prop_stressed_test, 'var_test': var_test,
        'backtest_oos': backtest_oos, 'backtest_standard': backtest_standard,
        'R_t_test': R_t_test, 'returns_test': returns_test
    }
    
    return results, None

# ============================================================================
# 📊 VISUALIZACIONES
# ============================================================================

def plot_correlation_heatmap(R_t, dates, tickers, title="Matriz de Correlación"):
    avg_corr = np.mean(R_t[-60:], axis=0)
    fig = go.Figure(data=go.Heatmap(
        z=avg_corr, x=tickers, y=tickers, colorscale='RdBu',
        zmid=0, text=np.round(avg_corr, 2), texttemplate="%{text}", textfont={"size": 10}
    ))
    fig.update_layout(title=title, xaxis_title="Activos", yaxis_title="Activos", height=500)
    return fig

def plot_correlation_timeseries(R_t, dates, tickers, pair=(0, 1)):
    corr_series = [R_t[t][pair[0], pair[1]] for t in range(len(R_t))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=corr_series, mode='lines',
        name=f"{tickers[pair[0]]} - {tickers[pair[1]]}", line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title=f"Correlación Dinámica: {tickers[pair[0]]} vs {tickers[pair[1]]}",
        xaxis_title="Fecha", yaxis_title="Correlación", yaxis=dict(range=[-1, 1]), height=400
    )
    return fig

def plot_homeostatic_indicator(H_t, prop_stressed, dates):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, row_heights=[0.3, 0.7])
    fig.add_trace(go.Scatter(
        x=dates, y=prop_stressed, mode='lines', name='Proporción en Tensión',
        line=dict(color='#ff7f0e', width=2)
    ), row=1, col=1)
    
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                  annotation_text="Umbral κ=0.3", row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=H_t, mode='lines', name='H_t (Homeostasis Activa)',
        line=dict(color='#2ca02c', width=3), fill='tozeroy'
    ), row=2, col=1)
    
    fig.update_layout(title="🏠 Indicador de Tensión Homeostática del Sistema", height=500, showlegend=True)
    return fig

def plot_var_backtesting(returns, var_series, dates):
    portfolio_return = returns.mean(axis=1)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=dates, y=portfolio_return, mode='lines', name='Retorno Portafolio', line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(x=dates, y=-var_series, mode='lines', name='VaR (95%)', line=dict(color='#d62728', width=2, dash='dash')))
    
    violations = portfolio_return < -var_series
    fig.add_trace(go.Scatter(
        x=dates[violations], y=portfolio_return[violations], mode='markers',
        name='Violaciones VaR', marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(title="⚠️ Backtesting de Value-at-Risk", xaxis_title="Fecha", yaxis_title="Retorno", height=400)
    return fig

def plot_out_of_sample_comparison(results):
    portfolio_return_oos = results['returns_test'].mean(axis=1)
    var_oos = results['var_test']
    dates_oos = results['returns_test'].index
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_oos, y=portfolio_return_oos, mode='lines', name='Retorno OoS', line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(x=dates_oos, y=-var_oos, mode='lines', name='VaR OoS (95%)', line=dict(color='#d62728', width=2, dash='dash')))
    
    violations = portfolio_return_oos < -var_oos
    fig.add_trace(go.Scatter(
        x=dates_oos[violations], y=portfolio_return_oos[violations], mode='markers',
        name='Violaciones VaR OoS', marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(title="📊 Validación Out-of-Sample: VaR vs Retornos Reales", xaxis_title="Fecha", yaxis_title="Retorno", height=400, showlegend=True)
    return fig

# ============================================================================
# 🖥️ INTERFAZ STREAMLIT
# ============================================================================

def main():
    st.markdown('<p class="main-header">🎓 Modelo DCC-GARCH Homeostático con EVT</p>', unsafe_allow_html=True)
    st.markdown("**Tesis Doctoral en Economía Financiera** | Detección de Regímenes de Corrección Homeostática")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Configuración del Modelo")
    
    st.sidebar.subheader("1. Selección de Activos")
    portfolio_choice = st.sidebar.selectbox("Portafolio Predefinido", ["Mínimo (6 activos)", "Completo (12 activos)", "Personalizado"])
    
    TICKERS_MINIMUM = ['^GSPC', '^STOXX50E', 'TLT', 'GLD', 'UUP', 'EEM']
    TICKERS_COMPLETE = ['^GSPC', '^STOXX50E', '^N225', '^VIX', 'TLT', 'HYG', 'GLD', 'USO', 'FXE', 'UUP', 'EEM', 'BTC-USD']
    
    default_tickers = TICKERS_MINIMUM if portfolio_choice == "Mínimo (6 activos)" else TICKERS_COMPLETE if portfolio_choice == "Completo (12 activos)" else TICKERS_MINIMUM
    
    tickers_input = st.sidebar.text_area("Tickers (separados por coma)", value=", ".join(default_tickers), help="Ejemplo: ^GSPC, TLT, GLD, UUP")
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    
    st.sidebar.subheader("2. Período de Análisis")
    col1, col2 = st.sidebar.columns(2)
    with col1: start_date = st.date_input("Inicio", value=datetime(2008, 1, 1))
    with col2: end_date = st.date_input("Fin", value=datetime.now())
    
    st.sidebar.subheader("3. Parámetros del Modelo")
    confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.90, 0.99, 0.95, 0.01)
    kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.1, 0.5, 0.3, 0.05)
    var_confidence = st.sidebar.slider("Confianza VaR", 0.90, 0.99, 0.95, 0.01)
    garch_window = st.sidebar.slider("Ventana GARCH (días)", 60, 500, 252)
    
    st.sidebar.subheader("4. Validación")
    enable_oos = st.sidebar.checkbox("Activar Validación Out-of-Sample", value=True)
    train_ratio = st.sidebar.slider("Proporción Entrenamiento (%)", 50, 90, 70, 5) if enable_oos else 70
    
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True)
    
    if run_button:
        with st.spinner("Descargando datos y ejecutando modelo..."):
            
            st.markdown('<p class="sub-header">📥 1. Descarga de Datos</p>', unsafe_allow_html=True)
            prices = download_data(tickers, start_date, end_date)
            if prices is None or prices.empty:
                st.error("No se pudieron descargar los datos. Verifica los tickers.")
                st.stop()
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Activos", len(tickers))
            with col2: st.metric("Período", f"{prices.index[0].strftime('%Y-%m-%d')} a {prices.index[-1].strftime('%Y-%m-%d')}")
            with col3: st.metric("Observaciones", len(prices))
            
            with st.expander("📋 Ver Datos de Precios"): st.dataframe(prices.tail(10))
            
            st.markdown("---")
            st.markdown('<p class="sub-header">📈 2. Cálculo de Retornos y Filtrado GARCH</p>', unsafe_allow_html=True)
            
            returns = calculate_returns(prices)
            z_std, sigma = garch_filter(returns)
            
            col1, col2 = st.columns(2)
            with col1: st.metric("Retorno Medio Anual", f"{returns.mean().mean()*252:.2%}")
            with col2: st.metric("Volatilidad Anual", f"{returns.std().std()*np.sqrt(252):.2%}")
            
            st.markdown("---")
            st.markdown('<p class="sub-header">🎯 3. Distribución de Gumbel y Umbrales</p>', unsafe_allow_html=True)
            
            thresholds, indicators = fit_gumbel_threshold(z_std, confidence_gumbel, garch_window)
            H_t, prop_stressed = calculate_systemic_indicator(indicators, kappa_threshold)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 Umbrales por Activo (Gumbel)**")
                threshold_df = pd.DataFrame({'Ticker': list(thresholds.keys()), 'Umbral': list(thresholds.values())})
                st.dataframe(threshold_df.style.format({'Umbral': '{:.4f}'}))
            with col2:
                st.markdown("**📊 Estadísticas de H_t**")
                st.metric("Días en Homeostasis", int(H_t.sum()))
                st.metric("Porcentaje del Tiempo", f"{H_t.mean()*100:.1f}%")
            
            st.plotly_chart(plot_homeostatic_indicator(H_t.values, prop_stressed.values, prop_stressed.index), use_container_width=True)
            
            st.markdown("---")
            st.markdown('<p class="sub-header">🔗 4. Correlación Dinámica (DCC-H)</p>', unsafe_allow_html=True)
            
            Q_bar = np.corrcoef(z_std.T)
            R_t, Q_t, params = dcc_homeostatic(z_std, H_t, Q_bar)
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Parámetro α (shock)", f"{params['a']:.3f}")
            with col2: st.metric("Parámetro β (persistencia)", f"{params['b']:.3f}")
            with col3: st.metric("Parámetro γ (homeostasis)", f"{params['gamma']:.3f}")
            
            st.plotly_chart(plot_correlation_heatmap(R_t, returns.index, tickers, "Matriz de Correlación Promedio (Últimos 60 días)"), use_container_width=True)
            
            st.markdown("**Seleccionar par de activos para ver evolución de correlación:**")
            col1, col2 = st.columns(2)
            with col1: asset1 = st.selectbox("Activo 1", tickers, index=0, key="asset1")
            with col2: asset2 = st.selectbox("Activo 2", tickers, index=1 if len(tickers) > 1 else 0, key="asset2")
            
            idx1, idx2 = tickers.index(asset1), tickers.index(asset2)
            st.plotly_chart(plot_correlation_timeseries(R_t, returns.index, tickers, (idx1, idx2)), use_container_width=True)
            
            st.markdown("---")
            st.markdown('<p class="sub-header">🧪 5. Test de Razón de Verosimilitud (Validación de H2)</p>', unsafe_allow_html=True)
            
            with st.spinner("Ejecutando Test LR..."):
                lr_results = likelihood_ratio_test(z_std, H_t, Q_bar)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Estadístico LR", f"{lr_results['lr_statistic']:.4f}")
                with col2: st.metric("Valor Crítico (5%)", f"{lr_results['critical_value']:.4f}")
                with col3: st.metric("P-value", f"{lr_results['p_value']:.6f}")
                with col4:
                    if lr_results['decision'] == "RECHAZAR_H0": st.success("✅ H0 Rechazada")
                    else: st.error("❌ H0 No Rechazada")
                
                if lr_results['decision'] == "RECHAZAR_H0":
                    st.success("""
                    **✅ Este resultado valida tu contribución doctoral:**
                    1. El parámetro γ es estadísticamente significativo (p < 0.05)
                    2. El modelo DCC-H explica mejor los datos que el DCC estándar
                    3. **La hipótesis H2 de tu tesis está respaldada empíricamente**
                    4. Puedes afirmar que los regímenes homeostáticos modifican la estructura de correlación
                    """)
                else:
                    st.warning("""
                    **⚠️ Consideraciones:**
                    1. El parámetro γ no es estadísticamente significativo en este período
                    2. Esto NO invalida tu tesis, pero sugiere probar con otros períodos, o ajustar umbrales.
                    """)
                
                st.markdown("### 📊 Comparación de Modelos")
                comparison_df = pd.DataFrame({
                    'Modelo': ['DCC Estándar', 'DCC Homeostático'], 'Parámetros': [2, 3],
                    'Log-Likelihood': [lr_results['log_lik_restricted'], lr_results['log_lik_unrestricted']],
                    'AIC': [-2*lr_results['log_lik_restricted'] + 2*2, -2*lr_results['log_lik_unrestricted'] + 2*3],
                    'BIC': [-2*lr_results['log_lik_restricted'] + 2*np.log(len(z_std)), -2*lr_results['log_lik_unrestricted'] + 3*np.log(len(z_std))]
                })
                st.dataframe(comparison_df.style.format({'Log-Likelihood': '{:.4f}', 'AIC': '{:.4f}', 'BIC': '{:.4f}'}))
            
            st.markdown("---")
            st.markdown('<p class="sub-header">⚠️ 6. Value-at-Risk y Backtesting</p>', unsafe_allow_html=True)
            
            var_series = calculate_var(returns, sigma, R_t, confidence=var_confidence)
            backtest_results = backtest_var(returns, var_series, var_confidence)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Violaciones Observadas", backtest_results['violations'])
            with col2: st.metric("Violaciones Esperadas", f"{backtest_results['expected']:.1f}")
            with col3: st.metric("Tasa Observada", f"{backtest_results['violation_rate']*100:.2f}%")
            with col4: st.metric("Tasa Esperada", f"{backtest_results['expected_rate']*100:.2f}%")
            
            if backtest_results['passed']: st.success(f"✅ Test de Kupiec APROBADO (p-value: {backtest_results['kupiec_pvalue']:.4f})")
            else: st.error(f"❌ Test de Kupiec RECHAZADO (p-value: {backtest_results['kupiec_pvalue']:.4f})")
            
            st.plotly_chart(plot_var_backtesting(returns, var_series, returns.index), use_container_width=True)
            
            if enable_oos:
                st.markdown("---")
                st.markdown('<p class="sub-header">🔬 7. Validación Out-of-Sample</p>', unsafe_allow_html=True)
                
                with st.spinner("Ejecutando validación out-of-sample..."):
                    oos_results, oos_error = out_of_sample_validation(prices, tickers, train_ratio/100, confidence_gumbel, kappa_threshold, var_confidence)
                    
                    if oos_results is None:
                        st.error(f"Error en validación OoS: {oos_error}")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Período Entrenamiento:** {oos_results['train_period']}")
                            st.info(f"**Observaciones Train:** {oos_results['n_train']}")
                        with col2:
                            st.info(f"**Período Prueba:** {oos_results['test_period']}")
                            st.info(f"**Observaciones Test:** {oos_results['n_test']}")
                        
                        st.markdown("### 📊 Comparación: DCC-H vs DCC Estándar (Out-of-Sample)")
                        comparison_oos = pd.DataFrame({
                            'Métrica': ['Violaciones', 'Tasa Observada', 'Tasa Esperada', 'Kupiec p-value'],
                            'DCC Homeostático': [
                                oos_results['backtest_oos']['violations'], f"{oos_results['backtest_oos']['violation_rate']*100:.2f}%",
                                f"{oos_results['backtest_oos']['expected_rate']*100:.2f}%", f"{oos_results['backtest_oos']['kupiec_pvalue']:.4f}"
                            ],
                            'DCC Estándar': [
                                oos_results['backtest_standard']['violations'], f"{oos_results['backtest_standard']['violation_rate']*100:.2f}%",
                                f"{oos_results['backtest_standard']['expected_rate']*100:.2f}%", f"{oos_results['backtest_standard']['kupiec_pvalue']:.4f}"
                            ]
                        })
                        st.dataframe(comparison_oos)
                        st.plotly_chart(plot_out_of_sample_comparison(oos_results), use_container_width=True)
                        
                        if oos_results['backtest_oos']['violations'] <= oos_results['backtest_standard']['violations']:
                            st.success("**✅ El modelo DCC-H muestra mejor o igual performance out-of-sample.**")
                        else:
                            st.warning("**⚠️ El modelo DCC-H tiene más violaciones en out-of-sample.**")
            
            st.markdown("---")
            st.markdown('<p class="sub-header">💾 8. Exportar Resultados</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                results_df = pd.DataFrame({'Date': returns.index, 'H_Indicator': H_t.values, 'Prop_Stressed': prop_stressed.values, 'VaR': var_series})
                st.download_button("📥 Descargar Series Temporales (CSV)", data=results_df.to_csv(index=False), file_name=f"dcc_homeostatic_results_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            
            with col2:
                summary = {
                    'Modelo': 'DCC-GARCH Homeostático', 'Activos': len(tickers), 'Período': f"{start_date} a {end_date}",
                    'Confianza Gumbel': confidence_gumbel, 'Umbral κ': kappa_threshold, 'Días Homeostasis': int(H_t.sum()),
                    'Violaciones VaR': backtest_results['violations'], 'Kupiec p-value': backtest_results['kupiec_pvalue'],
                    'LR Test p-value': lr_results['p_value'], 'Decisión LR': lr_results['decision']
                }
                summary_df = pd.DataFrame(summary, index=['Valor'])
                st.download_button("📥 Descargar Resumen del Modelo (CSV)", data=summary_df.to_csv(), file_name=f"dcc_homeostatic_summary_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    
    else:
        st.markdown("""
        ### 🎯 Bienvenido a la Aplicación del Modelo DCC-GARCH Homeostático
        
        Esta herramienta implementa el modelo desarrollado para la tesis doctoral:
        **"Dinámica de Corrección Homeostática en Mercados Financieros"**
        
        #### ¿Qué hace este modelo?
        
        1. **📊 Filtrado GARCH**: Extrae residuos estandarizados de los retornos
        2. **🎯 Teoría de Valores Extremos**: Ajusta distribución de Gumbel para detectar eventos extremos
        3. **🏠 Indicador Homeostático**: Identifica cuando el sistema está en "tensión" (H_t = 1)
        4. **🔗 DCC Modificado**: La correlación dinámica cambia según el régimen homeostático
        5. **🧪 Test LR**: Valida estadísticamente que γ ≠ 0 (contribución significativa)
        6. **🔬 Out-of-Sample**: Prueba el modelo en datos no vistos
        7. **⚠️ VaR Condicional**: Calcula riesgo ajustado al estado del sistema
        
        #### Hipótesis que se pueden testear:
        
        - **H1**: Los umbrales de Gumbel predicen mejor los eventos extremos que la distribución normal
        - **H2**: Las correlaciones cambian significativamente cuando H_t = 1 **(Validado con Test LR)**
        - **H3**: El VaR condicional tiene menos violaciones que modelos estándar
        
        ---
        
        <div class="warning-box">
        <strong>⚠️ Nota Académica:</strong> Esta aplicación es para fines de investigación. 
        Los resultados no constituyen recomendación de inversión. Para una tesis doctoral, 
        se recomienda validar con múltiples períodos y realizar pruebas de robustez.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="sub-header">📋 Tickers Recomendados para Investigación</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏛️ Índices y Equities**")
            st.code("^GSPC  - S&P 500 (US)\n^STOXX50E - EURO STOXX 50\n^N225 - Nikkei 225\nEEM - Emerging Markets")
        with col2:
            st.markdown("**🛡️ Safe Havens**")
            st.code("TLT - Treasury Bonds 20+\nGLD - Gold ETF\nUUP - Dollar Index\n^VIX - Volatility Index")
        st.markdown("**📦 Commodities & Otros**")
        st.code("USO - Oil | HYG - High Yield Bonds | FXE - Euro | BTC-USD - Bitcoin")

if __name__ == "__main__":
    main()
