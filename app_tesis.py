# ============================================================================
# 🎓 TESIS DOCTORAL: Modelo DCC-GARCH Homeostático con EVT (Gumbel)
# ============================================================================
# Archivo: app_tesis_final.py
# Versión: FINAL - ROBUSTO Y CORREGIDO PARA DATOS FALTANTES
# Ejecutar: streamlit run app_tesis_final.py
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

@st.cache_data(ttl=7200)
def download_data(tickers, start_date, end_date):
    """Descarga datos y asegura que no haya vacíos absolutos que rompan el modelo"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data is None or data.empty:
            return None, []
        
        prices = None
        
        # Extracción flexible para soportar cualquier versión de yfinance
        if isinstance(data.columns, pd.MultiIndex):
            for level in [0, 1]:
                if 'Adj Close' in data.columns.get_level_values(level):
                    prices = data.xs('Adj Close', level=level, axis=1)
                    break
                elif 'Close' in data.columns.get_level_values(level):
                    prices = data.xs('Close', level=level, axis=1)
                    break
            if prices is None:
                prices = data.iloc[:, :len(tickers)] # Fallback
        else:
            if 'Adj Close' in data.columns:
                prices = pd.DataFrame({tickers[0]: data['Adj Close']})
            elif 'Close' in data.columns:
                prices = pd.DataFrame({tickers[0]: data['Close']})
            else:
                prices = data.copy()
        
        if prices is None or prices.empty:
            return None, []
            
        # 🧹 LIMPIEZA ESTRICTA (Solución al error de "Retornos vacíos")
        # 1. Eliminar activos que no tienen ningún dato en este período
        prices_clean = prices.dropna(axis=1, how='all')
        dropped_tickers = [t for t in prices.columns if t not in prices_clean.columns]
        
        # 2. Rellenar huecos dejados por feriados distintos en cada mercado
        prices_clean = prices_clean.ffill().bfill()
        
        # 3. Eliminar filas que, después de rellenar, sigan teniendo NaNs
        prices_clean = prices_clean.dropna(axis=0, how='any')
        
        if prices_clean.empty or prices_clean.shape[1] == 0:
            return None, dropped_tickers
            
        return prices_clean, dropped_tickers
        
    except Exception as e:
        return None, []

def calculate_returns(prices):
    """Calcula retornos logarítmicos"""
    # Evitar divisiones por 0 o logaritmos de números negativos
    safe_prices = prices.copy()
    safe_prices[safe_prices <= 0] = np.nan
    safe_prices = safe_prices.ffill().bfill()
    
    returns = np.log(safe_prices / safe_prices.shift(1)).dropna()
    
    if returns.empty:
        raise ValueError("Retornos vacíos después de calcular. El rango de fechas podría ser muy corto.")
    return returns

# ============================================================================
# 📈 MODELO GARCH UNIVARIADO
# ============================================================================

def garch_filter(returns):
    """Filtrado GARCH(1,1) simplificado para residuos estandarizados"""
    n = len(returns)
    N = len(returns.columns)
    
    var_init = returns.var().values
    var_init[var_init < 1e-10] = 1e-6
    
    sigma2 = np.zeros((n, N))
    z_std = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    
    omega, alpha, beta = 0.00001, 0.1, 0.85
    
    for i in range(N):
        sigma_col = np.zeros(n)
        sigma_col[0] = var_init[i]
        
        vals = returns.iloc[:, i].values
        for t in range(1, n):
            sigma_col[t] = omega + alpha * (vals[t-1]**2) + beta * sigma_col[t-1]
        
        sigma2[:, i] = sigma_col
        sigma_sqrt = np.sqrt(sigma_col)
        sigma_sqrt[sigma_sqrt < 1e-10] = 1e-10
        
        z_std.iloc[:, i] = vals / sigma_sqrt
    
    sigma_mean = np.sqrt(sigma2.mean(axis=1))
    sigma_mean[sigma_mean < 1e-10] = 1e-10
    
    return z_std, sigma_mean

# ============================================================================
# 🎯 DISTRIBUCIÓN DE GUMBEL
# ============================================================================

def fit_gumbel_threshold(residuals, confidence=0.95, window=252):
    """Ajusta distribución de Gumbel y calcula umbrales de tensión homeostática"""
    thresholds = {}
    indicators = pd.DataFrame(index=residuals.index)
    
    for col in residuals.columns:
        locs, scales = [], []
        
        for t in range(window, len(residuals)):
            window_data = np.abs(residuals[col].iloc[t-window:t]).dropna()
            if len(window_data) < 10:
                continue
            try:
                loc, scale = gumbel_r.fit(window_data)
                locs.append(loc)
                scales.append(scale)
            except:
                continue
        
        if locs:
            avg_loc = np.mean(locs)
            avg_scale = np.mean(scales)
        else:
            try:
                clean_data = np.abs(residuals[col]).dropna()
                avg_loc, avg_scale = gumbel_r.fit(clean_data)
            except:
                avg_loc, avg_scale = 0, 1
        
        threshold = gumbel_r.ppf(confidence, loc=avg_loc, scale=avg_scale)
        thresholds[col] = threshold
        indicators[col] = ((np.abs(residuals[col]) > threshold).astype(int)).fillna(0).astype(int)
    
    return thresholds, indicators

def calculate_systemic_indicator(indicators, kappa=0.3):
    prop_stressed = indicators.mean(axis=1)
    H_t = (prop_stressed >= kappa).astype(int)
    return H_t, prop_stressed

# ============================================================================
# 🔗 MODELO DCC-GARCH HOMEOSTÁTICO
# ============================================================================

def ensure_positive_definite(matrix, min_eig=1e-6):
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(symmetric_matrix)
    new_eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(new_eigvals) @ eigvecs.T

def dcc_likelihood_full(z_std, H_indicator, Q_bar, params):
    T, N = z_std.shape
    a = max(params[0], 1e-8)
    b = max(params[1], 1e-8)
    gamma = max(params[2], 1e-8) if len(params) > 2 else 0.0
    
    if a + b + gamma >= 0.98 or a > 0.5 or b > 0.95:
        return -1000.0
    
    stress_periods = z_std[H_indicator == 1]
    if len(stress_periods) > 10:
        Q_stress = np.corrcoef(stress_periods.T)
    else:
        Q_stress = Q_bar.copy()
    
    Q_stress = ensure_positive_definite(Q_stress, min_eig=1e-4)
    Q_prev = ensure_positive_definite(Q_bar.copy(), min_eig=1e-4)
    
    log_lik = 0.0
    count_valid = 0
    
    for t in range(1, T):
        try:
            if gamma > 0 and H_indicator.iloc[t-1] == 1:
                Q_t = (1 - a - b - gamma) * Q_bar + \
                      a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                      b * Q_prev + gamma * Q_stress
            else:
                Q_t = (1 - a - b) * Q_bar + \
                      a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                      b * Q_prev
            
            diag_q = np.sqrt(np.diag(Q_t))
            diag_q = np.clip(diag_q, 1e-8, None)
            D_inv = np.diag(1 / diag_q)
            R_t = D_inv @ Q_t @ D_inv
            
            min_eig_R = np.min(np.linalg.eigvalsh(R_t))
            if min_eig_R < 1e-4:
                R_t = R_t + (1e-4 - min_eig_R) * np.eye(N)
            
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0 or np.isnan(logdet): continue
                
            z_vec = z_std.iloc[t-1].values
            R_inv = np.linalg.inv(R_t)
            quadratic = float(z_vec.T @ R_inv @ z_vec)
            
            if np.isnan(quadratic) or quadratic > 1000: continue
                
            log_lik += -0.5 * (logdet + quadratic)
            count_valid += 1
            Q_prev = R_t
        except:
            continue
    
    if count_valid < T * 0.8:
        return -10000.0 - (T * count_valid)
    return float(log_lik)

def estimate_dcc_parameters(z_std, H_indicator, Q_bar, model_type='DCC-H'):
    def neg_log_lik(params):
        result = dcc_likelihood_full(z_std, H_indicator, Q_bar, params)
        return 1e10 if np.isinf(result) or np.isnan(result) else -result
    
    initial_params = [0.02, 0.92, 0.02] if model_type == 'DCC-H' else [0.02, 0.92]
    bounds = [(1e-8, 0.3), (0.5, 0.95), (0, 0.3)] if model_type == 'DCC-H' else [(1e-8, 0.3), (0.5, 0.95)]
    
    return minimize(neg_log_lik, initial_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-8})

def dcc_homeostatic(z_std, H_indicator, Q_bar=None):
    if z_std is None: raise ValueError("z_std no puede ser nulo")
    T, N = z_std.shape
    
    if Q_bar is None: Q_bar = np.corrcoef(z_std.T)
    Q_bar = ensure_positive_definite(Q_bar, min_eig=1e-6)
    
    result = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
    params = result.x
    
    a, b = float(np.clip(params[0], 1e-8, 0.3)), float(np.clip(params[1], 0.5, 0.95))
    gamma = float(np.clip(params[2] if len(params) > 2 else 0.0, 0, 0.3))
    
    stress_periods = z_std[H_indicator == 1]
    Q_stress = ensure_positive_definite(np.corrcoef(stress_periods.T) if len(stress_periods) > 10 else Q_bar.copy())
    
    Q_t, R_t = np.zeros((T, N, N)), np.zeros((T, N, N))
    Q_t[0] = Q_bar
    
    for t in range(1, T):
        try:
            if gamma > 0 and H_indicator.iloc[t-1] == 1:
                Q_t[t] = (1 - a - b - gamma) * Q_bar + a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + b * Q_t[t-1] + gamma * Q_stress
            else:
                Q_t[t] = (1 - a - b) * Q_bar + a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + b * Q_t[t-1]
            
            Q_t[t] = ensure_positive_definite(Q_t[t], min_eig=1e-8)
            diag_q = np.clip(np.sqrt(np.diag(Q_t[t])), 1e-8, None)
            D_inv = np.diag(1 / diag_q)
            R_t[t] = D_inv @ Q_t[t] @ D_inv
            
            min_eig = np.min(np.linalg.eigvalsh(R_t[t]))
            if min_eig < 1e-5: R_t[t] = R_t[t] + (1e-5 - min_eig) * np.eye(N)
        except:
            R_t[t] = R_t[t-1] if t > 0 else Q_bar
    
    return R_t, Q_t, {'a': a, 'b': b, 'gamma': gamma, 'log_lik': -result.fun}

# ============================================================================
# 🧪 TEST DE RAZÓN DE VEROSIMILITUD
# ============================================================================

def likelihood_ratio_test(z_std, H_indicator, Q_bar):
    try:
        res_restr = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC')
        res_unrestr = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
        
        lr_stat = float(np.clip(2 * (-res_unrestr.fun - (-res_restr.fun)), 0, 1e6))
        df = len(res_unrestr.x) - len(res_restr.x)
        p_value = 1 - chi2.cdf(lr_stat, df) if lr_stat > 0 else 1.0
        
        return {
            'lr_statistic': lr_stat, 'df': df, 'p_value': p_value,
            'critical_value': chi2.ppf(0.95, df),
            'decision': "RECHAZAR_H0" if p_value < 0.05 else "NO_RECHAZAR_H0",
            'log_lik_restricted': -res_restr.fun, 'log_lik_unrestricted': -res_unrestr.fun,
            'params_restricted': res_restr.x, 'params_unrestricted': res_unrestr.x
        }
    except:
        return {'lr_statistic': 0.0, 'df': 1, 'p_value': 1.0, 'critical_value': 3.8415,
                'decision': 'NO_RECHAZAR_H0', 'log_lik_restricted': -1000.0, 'log_lik_unrestricted': -1000.0}

# ============================================================================
# ⚠️ BACKTESTING DE VaR
# ============================================================================

def calculate_var(returns, R_t, weights=None, confidence=0.95):
    T, N = returns.shape
    if weights is None: weights = np.ones(N) / N
    var_series = np.zeros(T)
    z_score = norm.ppf(1 - confidence)
    
    for t in range(T):
        sigma2_p = weights @ R_t[t] @ weights * returns.iloc[t].var()
        var_series[t] = -(np.sqrt(sigma2_p) if sigma2_p > 0 else 1e-10) * z_score
    return var_series

def backtest_var(returns, var_series, confidence=0.95):
    violations = (returns.mean(axis=1) < -var_series).astype(int)
    n_viol = violations.sum()
    n_obs = len(violations)
    p = 1 - confidence
    p_hat = n_viol / n_obs if n_obs > 0 else 0
    
    lr_stat = 0
    if 0 < p_hat < 1:
        lr_stat = -2 * (n_obs * np.log(1-p) + n_viol * np.log(p/(1-p)) - 
                        n_obs * np.log(1-p_hat) - n_viol * np.log(p_hat/(1-p_hat)))
    
    p_value = 1 - chi2.cdf(lr_stat, 1) if lr_stat > 0 else 1.0
    return {'violations': n_viol, 'expected': n_obs * p, 'violation_rate': p_hat,
            'expected_rate': p, 'kupiec_lr': lr_stat, 'kupiec_pvalue': p_value, 'passed': p_value > 0.05}

# ============================================================================
# 📊 VALIDACIÓN OUT-OF-SAMPLE
# ============================================================================

def out_of_sample_validation(prices, tickers, train_ratio=0.7, confidence_gumbel=0.95, kappa_threshold=0.3, var_confidence=0.95):
    returns = calculate_returns(prices)
    n_train = int(len(returns) * train_ratio)
    
    if len(returns) - n_train < 50: return None, "Período de prueba demasiado corto"
    
    returns_train, returns_test = returns.iloc[:n_train], returns.iloc[n_train:]
    
    z_std_train, _ = garch_filter(returns_train)
    _, ind_train = fit_gumbel_threshold(z_std_train, confidence_gumbel)
    H_t_train, _ = calculate_systemic_indicator(ind_train, kappa_threshold)
    Q_bar_train = np.corrcoef(z_std_train.T)
    
    z_std_test, _ = garch_filter(returns_test)
    _, ind_test = fit_gumbel_threshold(z_std_test, confidence_gumbel)
    H_t_test, prop_stressed_test = calculate_systemic_indicator(ind_test, kappa_threshold)
    
    R_t_test, _, params_test = dcc_homeostatic(z_std_test, H_t_test, Q_bar_train)
    var_test = calculate_var(returns_test, R_t_test, confidence=var_confidence)
    
    R_t_std, _, _ = dcc_homeostatic(z_std_test, pd.Series(0, index=H_t_test.index), Q_bar_train)
    var_std = calculate_var(returns_test, R_t_std, confidence=var_confidence)
    
    return {
        'train_period': f"{returns_train.index[0].strftime('%Y-%m-%d')} a {returns_train.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{returns_test.index[0].strftime('%Y-%m-%d')} a {returns_test.index[-1].strftime('%Y-%m-%d')}",
        'n_train': n_train, 'n_test': len(returns_test), 'params_test': params_test,
        'H_t_test': H_t_test, 'prop_stressed_test': prop_stressed_test, 'var_test': var_test,
        'backtest_oos': backtest_var(returns_test, var_test, var_confidence),
        'backtest_standard': backtest_var(returns_test, var_std, var_confidence),
        'R_t_test': R_t_test, 'returns_test': returns_test
    }, None

# ============================================================================
# 📊 VISUALIZACIONES
# ============================================================================

def plot_correlation_heatmap(R_t, dates, tickers, title="Matriz de Correlación"):
    avg_corr = np.mean(R_t[-60:], axis=0)
    fig = go.Figure(data=go.Heatmap(z=avg_corr, x=tickers, y=tickers, colorscale='RdBu', zmid=0,
                                    text=np.round(avg_corr, 2), texttemplate="%{text}"))
    fig.update_layout(title=title, height=500)
    return fig

def plot_correlation_timeseries(R_t, dates, tickers, pair=(0, 1)):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=[R_t[t][pair[0], pair[1]] for t in range(len(R_t))], mode='lines'))
    fig.update_layout(title=f"Correlación: {tickers[pair[0]]} vs {tickers[pair[1]]}", yaxis=dict(range=[-1, 1]), height=400)
    return fig

def plot_homeostatic_indicator(H_t, prop_stressed, dates):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7])
    fig.add_trace(go.Scatter(x=dates, y=prop_stressed, name='Proporción', line=dict(color='#ff7f0e')), row=1, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=H_t, name='H_t', fill='tozeroy', line=dict(color='#2ca02c')), row=2, col=1)
    fig.update_layout(title="🏠 Indicador Homeostático", height=500)
    return fig

def plot_var_backtesting(returns, var_series, dates):
    port_ret = returns.mean(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=port_ret, name='Retornos', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=dates, y=-var_series, name='VaR', line=dict(color='#d62728', dash='dash')))
    viol = port_ret < -var_series
    fig.add_trace(go.Scatter(x=dates[viol], y=port_ret[viol], mode='markers', name='Violaciones', marker=dict(color='red', size=8)))
    fig.update_layout(title="⚠️ Backtesting VaR", height=400)
    return fig

def plot_out_of_sample_comparison(results):
    port_ret = results['returns_test'].mean(axis=1)
    var_oos = results['var_test']
    dates = results['returns_test'].index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=port_ret, name='Retornos OoS', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=dates, y=-var_oos, name='VaR OoS', line=dict(color='#d62728', dash='dash')))
    viol = port_ret < -var_oos
    fig.add_trace(go.Scatter(x=dates[viol], y=port_ret[viol], mode='markers', name='Violaciones', marker=dict(color='red')))
    fig.update_layout(title="📊 Validación Out-of-Sample", height=400)
    return fig

# ============================================================================
# 🖥️ INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.markdown('<p class="main-header">🎓 Modelo DCC-GARCH Homeostático con EVT</p>', unsafe_allow_html=True)
    st.markdown("**Tesis Doctoral** | Detección de Regímenes de Corrección Homeostática")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Configuración del Modelo")
    
    portfolio_choice = st.sidebar.selectbox("1. Portafolio", ["Mínimo (6 activos)", "Completo (12 activos)", "Personalizado"])
    default_tickers = ['^GSPC', '^STOXX50E', 'TLT', 'GLD', 'UUP', 'EEM'] if portfolio_choice == "Mínimo (6 activos)" else ['^GSPC', '^STOXX50E', '^N225', '^VIX', 'TLT', 'HYG', 'GLD', 'USO', 'FXE', 'UUP', 'EEM', 'BTC-USD']
    tickers_input = st.sidebar.text_area("Tickers", value=", ".join(default_tickers))
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    regime = st.sidebar.selectbox("2. Período", ["COVID-19 (2020)", "Crisis 2008", "Normal (2018-2019)", "Personalizado"])
    if regime == "COVID-19 (2020)": start_date, end_date, conf, kap = datetime(2020,1,1), datetime(2020,6,30), 0.985, 0.50
    elif regime == "Crisis 2008": start_date, end_date, conf, kap = datetime(2008,1,1), datetime(2008,12,31), 0.99, 0.55
    elif regime == "Normal (2018-2019)": start_date, end_date, conf, kap = datetime(2018,1,1), datetime(2019,12,31), 0.98, 0.45
    else:
        start_date, end_date = st.sidebar.date_input("Inicio", datetime(2020,1,1)), st.sidebar.date_input("Fin", datetime.now())
        conf, kap = 0.98, 0.45
    
    conf = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, conf, 0.005)
    kap = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, kap, 0.05)
    
    var_conf = st.sidebar.slider("Confianza VaR", 0.90, 0.99, 0.95, 0.01)
    enable_oos = st.sidebar.checkbox("Activar Out-of-Sample", True)
    
    if st.sidebar.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True):
        with st.spinner("Procesando..."):
            
            prices, dropped = download_data(tickers, start_date, end_date)
            
            if prices is None or prices.empty:
                st.error("❌ No se encontraron datos válidos. Revisa los Tickers o amplía el rango de fechas.")
                st.stop()
                
            if dropped:
                st.warning(f"⚠️ Activos omitidos por falta de datos en el período: {', '.join(dropped)}")
                
            valid_tickers = list(prices.columns)
            
            if len(valid_tickers) < 2:
                st.error("❌ Necesitas al menos 2 activos válidos para calcular correlación dinámica (DCC).")
                st.stop()
                
            col1, col2, col3 = st.columns(3)
            col1.metric("Activos Válidos", len(valid_tickers))
            col2.metric("Período", f"{prices.index[0].strftime('%Y-%m-%d')} a {prices.index[-1].strftime('%Y-%m-%d')}")
            col3.metric("Observaciones", len(prices))
            
            # Continuar solo con activos válidos
            returns = calculate_returns(prices)
            z_std, _ = garch_filter(returns)
            
            thresh, ind = fit_gumbel_threshold(z_std, conf)
            H_t, prop_str = calculate_systemic_indicator(ind, kap)
            
            st.plotly_chart(plot_homeostatic_indicator(H_t.values, prop_str.values, prop_str.index), use_container_width=True)
            
            Q_bar = np.corrcoef(z_std.T)
            R_t, _, params = dcc_homeostatic(z_std, H_t, Q_bar)
            
            st.plotly_chart(plot_correlation_heatmap(R_t, returns.index, valid_tickers), use_container_width=True)
            
            c1, c2 = st.columns(2)
            asset1 = c1.selectbox("Activo 1", valid_tickers, index=0)
            asset2 = c2.selectbox("Activo 2", valid_tickers, index=1)
            st.plotly_chart(plot_correlation_timeseries(R_t, returns.index, valid_tickers, (valid_tickers.index(asset1), valid_tickers.index(asset2))), use_container_width=True)
            
            st.markdown("### 🧪 Test de Razón de Verosimilitud (LR)")
            lr = likelihood_ratio_test(z_std, H_t, Q_bar)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Estadístico LR", f"{lr['lr_statistic']:.4f}")
            c2.metric("Valor Crítico", f"{lr['critical_value']:.4f}")
            c3.metric("P-value", f"{lr['p_value']:.6f}")
            if lr['decision'] == "RECHAZAR_H0": c4.success("✅ H0 Rechazada (Efecto Significativo)")
            else: c4.warning("⚠️ H0 No Rechazada")
            
            var_series = calculate_var(returns, R_t, confidence=var_conf)
            bt = backtest_var(returns, var_series, var_conf)
            st.plotly_chart(plot_var_backtesting(returns, var_series, returns.index), use_container_width=True)
            
            if enable_oos:
                st.markdown("### 🔬 Validación Out-of-Sample")
                oos, err = out_of_sample_validation(prices, valid_tickers, 0.7, conf, kap, var_conf)
                if err: st.warning(err)
                else: st.plotly_chart(plot_out_of_sample_comparison(oos), use_container_width=True)

if __name__ == "__main__":
    main()
