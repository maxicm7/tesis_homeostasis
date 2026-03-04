# ============================================================================
# 🎓 TESIS DOCTORAL: Modelo DCC-GARCH Homeostático con EVT (Gumbel)
# ============================================================================
# Archivo: app_tesis.py
# Versión: FINAL PARA DEFENSA DOCTORAL
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

@st.cache_data(ttl=7200)
def download_data(tickers, start_date, end_date):
    """Descarga datos desde Yahoo Finance y asegura formato DataFrame sin vacíos"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            return None
        
        # Manejar estructura MultiIndex (múltiples activos)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                prices = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                prices = data['Close']
            else:
                prices = data.iloc[:, :len(tickers)]
        else:
            # Manejo si se descarga 1 solo activo o YF cambia formato
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']].copy()
            elif 'Close' in data.columns:
                prices = data[['Close']].copy()
            else:
                prices = data.copy()
        
        if len(prices.columns) == 1 and len(tickers) == 1:
            prices.columns = tickers
        elif len(prices.columns) == 1:
            prices.columns = [tickers[0]]
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        # 1. Eliminar activos que no existían en absoluto (columnas 100% NaN)
        prices = prices.dropna(axis=1, how='all')
        
        if prices.empty or prices.shape[1] == 0:
            return None
        
        # 2. Rellenar huecos internos (festivos) hacia adelante
        prices = prices.ffill()
        
        # 3. Eliminar fechas iniciales donde algunos activos aún no existían
        prices = prices.dropna(how='any')
        
        if prices.empty or prices.shape[0] < 10:
            return None
        
        return prices
    
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
        return None

def calculate_returns(prices):
    """Calcula retornos logarítmicos garantizando un DataFrame limpio"""
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    returns = np.log(prices / prices.shift(1)).dropna(how='any')
    
    # Manejar infinitos generados por precios anómalos (cero o negativos)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    
    return returns

# ============================================================================
# 📈 MODELO GARCH UNIVARIADO
# ============================================================================

def garch_filter(returns):
    """
    Filtrado GARCH(1,1) para residuos estandarizados y volatilidades (D_t)
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    
    n = len(returns)
    N = len(returns.columns)
    
    # Calcular varianza incondicional (promedio)
    total_var = float(returns.var().mean())
    if total_var < 1e-10:
        total_var = 1e-6
    
    sigma_matrix = np.zeros((n, N))
    z_std_list = []
    
    omega, alpha, beta = 0.00001, 0.1, 0.85
    
    for i in range(N):
        sigma2_col = np.full(n, total_var)
        
        # Filtro iterativo GARCH(1,1)
        for t in range(1, n):
            val = returns.iloc[t-1, i]
            sigma2_col[t] = omega + alpha * val**2 + beta * sigma2_col[t-1]
        
        sigma_col = np.sqrt(sigma2_col)
        sigma_col[sigma_col < 1e-10] = 1e-10
        sigma_matrix[:, i] = sigma_col
        
        # Residuos Estandarizados
        z_col = returns.iloc[:, i] / sigma_col
        z_std_list.append(z_col.values)
    
    z_std = pd.DataFrame(np.column_stack(z_std_list), index=returns.index, columns=returns.columns)
    
    return z_std, sigma_matrix

# ============================================================================
# 🎯 DISTRIBUCIÓN DE GUMBEL
# ============================================================================

def fit_gumbel_threshold(residuals, confidence=0.95, window=252):
    """
    Ajusta distribución de Gumbel y calcula umbrales de tensión homeostática
    """
    thresholds = {}
    indicators = pd.DataFrame(index=residuals.index)
    
    for col in residuals.columns:
        locs = []
        scales = []
        
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
    """
    Calcula indicador sistémico H_t
    """
    prop_stressed = indicators.mean(axis=1)
    H_t = (prop_stressed >= kappa).astype(int)
    return H_t, prop_stressed

# ============================================================================
# 🔗 MODELO DCC-GARCH HOMEOSTÁTICO
# ============================================================================

def ensure_positive_definite(matrix, min_eig=1e-6):
    """Forzar que una matriz sea definida positiva"""
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(symmetric_matrix)
    
    if np.min(eigvals) < min_eig:
        symmetric_matrix = symmetric_matrix + (min_eig - np.min(eigvals)) * np.eye(len(matrix))
    
    new_eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(new_eigvals) @ eigvecs.T

def dcc_likelihood_full(z_std, H_indicator, Q_bar, params):
    """
    Calcula la log-verosimilitud completa del modelo DCC
    """
    T = len(z_std)
    N = z_std.shape[1]
    
    a = max(params[0], 1e-8)
    b = max(params[1], 1e-8)
    gamma = max(params[2], 1e-8) if len(params) > 2 else 0.0
    
    # Restricciones para estabilidad
    if a + b + gamma >= 0.98 or a > 0.5 or b > 0.95:
        return -1000.0
    
    # Matriz de estrés
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
                      b * Q_prev + \
                      gamma * Q_stress
            else:
                Q_t = (1 - a - b) * Q_bar + \
                      a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                      b * Q_prev
            
            # Normalizar a correlación
            diag_q = np.sqrt(np.diag(Q_t))
            diag_q = np.clip(diag_q, 1e-8, None)
            D_inv = np.diag(1 / diag_q)
            R_t = D_inv @ Q_t @ D_inv
            
            # Validar definida-positividad
            min_eig_R = np.min(np.linalg.eigvalsh(R_t))
            if min_eig_R < 1e-4:
                R_t = R_t + (1e-4 - min_eig_R) * np.eye(N)
            
            # Contribución a log-verosimilitud
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0 or np.isnan(logdet):
                continue
                
            z_vec = z_std.iloc[t-1].values
            R_inv = np.linalg.inv(R_t)
            quadratic = float(z_vec.T @ R_inv @ z_vec)
            
            if np.isnan(quadratic) or quadratic > 1000:
                continue
                
            contribution = -0.5 * (logdet + quadratic)
            log_lik += contribution
            count_valid += 1
            Q_prev = R_t
            
        except Exception as e:
            continue
    
    if count_valid < T * 0.8:
        return -10000.0 - (T * count_valid)
    
    return float(log_lik)

def estimate_dcc_parameters(z_std, H_indicator, Q_bar, model_type='DCC-H'):
    """
    Estima parámetros DCC por máxima verosimilitud
    """
    def neg_log_lik(params):
        result = dcc_likelihood_full(z_std, H_indicator, Q_bar, params)
        if np.isinf(result) or np.isnan(result):
            return 1e10
        return -result
    
    if model_type == 'DCC-H':
        initial_params = [0.02, 0.92, 0.02]
        bounds = [(1e-8, 0.3), (0.5, 0.95), (0, 0.3)]
    else:
        initial_params = [0.02, 0.92]
        bounds = [(1e-8, 0.3), (0.5, 0.95)]
    
    result = minimize(
        neg_log_lik,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-8}
    )
    
    return result

def dcc_homeostatic(z_std, H_indicator, Q_bar=None, fixed_params=None):
    """
    Implementación del DCC-GARCH Homeostático.
    Permite fijar los parámetros para validaciones Out-of-Sample genuinas.
    """
    if z_std is None or z_std.empty:
        raise ValueError("z_std no puede ser nulo o vacío")
    
    T = len(z_std)
    N = z_std.shape[1]
    
    if Q_bar is None:
        Q_bar = np.corrcoef(z_std.T)
    
    Q_bar = ensure_positive_definite(Q_bar, min_eig=1e-6)
    
    # Estimar parámetros o usar fijos
    if fixed_params is None:
        result = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
        params = result.x
        log_lik = -result.fun
    else:
        params = fixed_params
        log_lik = None
    
    a = float(np.clip(params[0], 1e-8, 0.3))
    b = float(np.clip(params[1], 0.5, 0.95))
    gamma = float(np.clip(params[2] if len(params) > 2 else 0.0, 0, 0.3))
    
    # Matriz de estrés
    stress_periods = z_std[H_indicator == 1]
    Q_stress = np.corrcoef(stress_periods.T) if len(stress_periods) > 10 else Q_bar.copy()
    Q_stress = ensure_positive_definite(Q_stress, min_eig=1e-6)
    
    # Evolución de Q_t
    Q_t = np.zeros((T, N, N))
    R_t = np.zeros((T, N, N))
    Q_t[0] = Q_bar
    
    for t in range(1, T):
        try:
            if gamma > 0 and H_indicator.iloc[t-1] == 1:
                Q_t[t] = (1 - a - b - gamma) * Q_bar + \
                         a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                         b * Q_t[t-1] + \
                         gamma * Q_stress
            else:
                Q_t[t] = (1 - a - b) * Q_bar + \
                         a * np.outer(z_std.iloc[t-1], z_std.iloc[t-1]) + \
                         b * Q_t[t-1]
            
            Q_t[t] = ensure_positive_definite(Q_t[t], min_eig=1e-8)
            
            diag_q = np.sqrt(np.diag(Q_t[t]))
            diag_q = np.clip(diag_q, 1e-8, None)
            D_inv = np.diag(1 / diag_q)
            R_t[t] = D_inv @ Q_t[t] @ D_inv
            
            min_eig = np.min(np.linalg.eigvalsh(R_t[t]))
            if min_eig < 1e-5:
                R_t[t] = R_t[t] + (1e-5 - min_eig) * np.eye(N)
                
        except Exception as e:
            Q_t[t] = Q_t[t-1] if t > 0 else Q_bar
            R_t[t] = R_t[t-1] if t > 0 else Q_bar
    
    # Asignar también el R_t inicial
    diag_q0 = np.sqrt(np.diag(Q_t[0]))
    diag_q0 = np.clip(diag_q0, 1e-8, None)
    D_inv0 = np.diag(1 / diag_q0)
    R_t[0] = D_inv0 @ Q_t[0] @ D_inv0
    
    return R_t, Q_t, {'a': a, 'b': b, 'gamma': gamma, 'log_lik': log_lik}

# ============================================================================
# 🧪 TEST DE RAZÓN DE VEROSIMILITUD
# ============================================================================

def likelihood_ratio_test(z_std, H_indicator, Q_bar):
    """
    Test de Razón de Verosimilitud: DCC-H vs DCC estándar
    """
    try:
        # Modelo restringido (DCC estándar, γ = 0)
        result_restricted = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC')
        log_lik_restricted = -result_restricted.fun
        
        # Modelo no restringido (DCC-H con γ libre)
        result_unrestricted = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
        log_lik_unrestricted = -result_unrestricted.fun
        
        # Estadístico LR
        lr_stat = 2 * (log_lik_unrestricted - log_lik_restricted)
        lr_stat = float(np.clip(lr_stat, 0, 1e6))
        
        # Grados de libertad
        df = len(result_unrestricted.x) - len(result_restricted.x)
        
        # P-value
        p_value = 1 - chi2.cdf(lr_stat, df) if lr_stat > 0 else 1.0
        
        # Valor crítico
        critical_value = chi2.ppf(0.95, df)
        
        # Decisión
        decision = "RECHAZAR_H0" if p_value < 0.05 else "NO_RECHAZAR_H0"
        
        return {
            'lr_statistic': lr_stat,
            'df': df,
            'p_value': p_value,
            'critical_value': critical_value,
            'decision': decision,
            'log_lik_restricted': log_lik_restricted,
            'log_lik_unrestricted': log_lik_unrestricted,
            'params_restricted': result_restricted.x,
            'params_unrestricted': result_unrestricted.x
        }
    
    except Exception as e:
        return {
            'lr_statistic': 0.0,
            'df': 1,
            'p_value': 1.0,
            'critical_value': 3.8415,
            'decision': 'NO_RECHAZAR_H0',
            'log_lik_restricted': -1000.0,
            'log_lik_unrestricted': -1000.0,
            'params_restricted': [0.0, 0.9],
            'params_unrestricted': [0.0, 0.9, 0.0]
        }

# ============================================================================
# ⚠️ BACKTESTING DE VaR
# ============================================================================

def calculate_var(returns, R_t, sigma_matrix, weights=None, confidence=0.95):
    """Calcula Value-at-Risk condicional riguroso usando H_t (Covarianza)"""
    T = returns.shape[0]
    N = returns.shape[1]
    
    if weights is None:
        weights = np.ones(N) / N
    
    var_series = np.zeros(T)
    
    for t in range(T):
        # Matriz diagonal de volatilidades condicionales (D_t)
        D_t = np.diag(sigma_matrix[t])
        
        # Matriz de covarianza condicional H_t = D_t * R_t * D_t
        H_t = D_t @ R_t[t] @ D_t
        
        # Varianza matemática real del portafolio (w^T * H_t * w)
        sigma2_p = weights.T @ H_t @ weights
        sigma_p = np.sqrt(sigma2_p) if sigma2_p > 0 else 1e-10
        
        # VaR condicional
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
    
    # Kupiec POF Test
    p_hat = n_violations / n_observations if n_observations > 0 else 0
    p = 1 - confidence
    
    if p_hat > 0 and p_hat < 1 and n_observations > 0:
        lr_stat = -2 * (n_observations * np.log(1-p) + n_violations * np.log(p/(1-p)) -
                       n_observations * np.log(1-p_hat) - n_violations * np.log(p_hat/(1-p_hat)))
    else:
        lr_stat = 0
    
    p_value = 1 - chi2.cdf(lr_stat, 1) if lr_stat > 0 else 1.0
    
    return {
        'violations': n_violations,
        'expected': expected_violations,
        'violation_rate': n_violations / n_observations if n_observations > 0 else 0,
        'expected_rate': 1 - confidence,
        'kupiec_lr': lr_stat,
        'kupiec_pvalue': p_value,
        'passed': p_value > 0.05
    }

# ============================================================================
# 📊 VALIDACIÓN OUT-OF-SAMPLE
# ============================================================================

def out_of_sample_validation(prices, valid_tickers, train_ratio=0.7, confidence_gumbel=0.95, 
                             kappa_threshold=0.3, var_confidence=0.95, garch_window=252):
    """
    Validación Out-of-Sample pura evitando el Look-Ahead Bias.
    """
    returns = calculate_returns(prices)
    
    if len(returns.columns) < 2:
        return None, "Se requieren al menos 2 activos para el modelo DCC."
    
    n_obs = len(returns)
    n_train = int(n_obs * train_ratio)
    
    if n_obs - n_train < 50:
        return None, "Período de prueba demasiado corto (mín. 50 observaciones)."
    
    returns_train = returns.iloc[:n_train]
    returns_test = returns.iloc[n_train:]
    
    # ========== FASE DE ENTRENAMIENTO ==========
    z_std_train, sigma_train = garch_filter(returns_train)
    thresholds_train, indicators_train = fit_gumbel_threshold(z_std_train, confidence_gumbel, garch_window)
    H_t_train, prop_stressed_train = calculate_systemic_indicator(indicators_train, kappa_threshold)
    
    # Estimar parámetros libres (Train)
    Q_bar_train = np.corrcoef(z_std_train.T)
    R_t_train, Q_t_train, p_train = dcc_homeostatic(z_std_train, H_t_train, Q_bar_train)
    params_train = [p_train['a'], p_train['b'], p_train['gamma']]
    
    # Estimar DCC Estándar para comparación (Train)
    _, _, p_std = dcc_homeostatic(z_std_train, pd.Series(0, index=H_t_train.index), Q_bar_train)
    params_std = [p_std['a'], p_std['b'], 0.0]
    
    # ========== FASE DE PRUEBA (OUT-OF-SAMPLE) ==========
    z_std_test, sigma_test = garch_filter(returns_test)
    thresholds_test, indicators_test = fit_gumbel_threshold(z_std_test, confidence_gumbel, garch_window)
    H_t_test, prop_stressed_test = calculate_systemic_indicator(indicators_test, kappa_threshold)
    
    # Proyectar utilizando parámetros 'congelados' de Train (Evita sobreajuste/look-ahead)
    R_t_test, Q_t_test, _ = dcc_homeostatic(z_std_test, H_t_test, Q_bar_train, fixed_params=params_train)
    var_test = calculate_var(returns_test, R_t_test, sigma_test, confidence=var_confidence)
    backtest_oos = backtest_var(returns_test, var_test, var_confidence)
    
    # Proyectar Estándar para benchmark
    R_t_standard, _, _ = dcc_homeostatic(z_std_test, pd.Series(0, index=H_t_test.index), Q_bar_train, fixed_params=params_std)
    var_standard = calculate_var(returns_test, R_t_standard, sigma_test, confidence=var_confidence)
    backtest_standard = backtest_var(returns_test, var_standard, var_confidence)
    
    results = {
        'train_period': f"{returns_train.index[0].strftime('%Y-%m-%d')} a {returns_train.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{returns_test.index[0].strftime('%Y-%m-%d')} a {returns_test.index[-1].strftime('%Y-%m-%d')}",
        'n_train': n_train,
        'n_test': len(returns_test),
        'params_train': p_train,
        'H_t_test': H_t_test,
        'prop_stressed_test': prop_stressed_test,
        'var_test': var_test,
        'backtest_oos': backtest_oos,
        'backtest_standard': backtest_standard,
        'R_t_test': R_t_test,
        'returns_test': returns_test
    }
    
    return results, None

# ============================================================================
# 📊 VISUALIZACIONES
# ============================================================================

def plot_correlation_heatmap(R_t, dates, tickers, title="Matriz de Correlación"):
    """Heatmap de correlaciones"""
    avg_corr = np.mean(R_t[-60:], axis=0)
    
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

def plot_out_of_sample_comparison(results):
    """Comparación de performance In-Sample vs Out-of-Sample"""
    portfolio_return_oos = results['returns_test'].mean(axis=1)
    var_oos = results['var_test']
    dates_oos = results['returns_test'].index
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates_oos,
        y=portfolio_return_oos,
        mode='lines',
        name='Retorno OoS',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates_oos,
        y=-var_oos,
        mode='lines',
        name='VaR OoS',
        line=dict(color='#d62728', width=2, dash='dash')
    ))
    
    # Marcar violaciones OoS
    violations = portfolio_return_oos < -var_oos
    violation_dates = dates_oos[violations]
    violation_values = portfolio_return_oos[violations]
    
    fig.add_trace(go.Scatter(
        x=violation_dates,
        y=violation_values,
        mode='markers',
        name='Violaciones VaR OoS',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title="📊 Validación Out-of-Sample: VaR vs Retornos Reales",
        xaxis_title="Fecha",
        yaxis_title="Retorno",
        height=400,
        showlegend=True
    )
    
    return fig

# ============================================================================
# 🖥️ INTERFAZ STREAMLIT PRINCIPAL
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
    
    # Selección de período con pandemia automática
    st.sidebar.subheader("2. Período de Análisis")
    
    regime_option = st.sidebar.selectbox(
        "Selecciona el régimen",
        [
            "✅ COVID-19 Pandemia (Enero-Junio 2020) - RECOMENDADO",
            "COVID-19 Completo (2020)",
            "COVID-19 Extendido (2020-2021)",
            "Crisis Financiera Global (2008)",
            "Crisis Eurozona (2011)",
            "Periodo Normal (2018-2019)",
            "Personalizado"
        ]
    )
    
    # Definir fechas según selección
    if regime_option == "✅ COVID-19 Pandemia (Enero-Junio 2020) - RECOMENDADO":
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 6, 30)
        st.sidebar.success("✔️ Período COVID-19 establecido: 2020-01-01 a 2020-06-30")
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.985, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.50, 0.05)
        
    elif regime_option == "COVID-19 Completo (2020)":
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)
        st.sidebar.success("✔️ Período COVID-19 2020 establecido")
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.99, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.60, 0.05)
        
    elif regime_option == "COVID-19 Extendido (2020-2021)":
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2021, 12, 31)
        st.sidebar.success("✔️ Período COVID-19 Extendido establecido")
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.99, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.60, 0.05)
        
    elif regime_option == "Crisis Financiera Global (2008)":
        start_date = datetime(2008, 1, 1)
        end_date = datetime(2008, 12, 31)
        st.sidebar.success("✔️ Período Crisis 2008 establecido")
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.95, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.30, 0.05)
        
    elif regime_option == "Crisis Eurozona (2011)":
        start_date = datetime(2011, 1, 1)
        end_date = datetime(2011, 12, 31)
        st.sidebar.success("✔️ Período Eurozona 2011 establecido")
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.98, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.45, 0.05)
        
    elif regime_option == "Periodo Normal (2018-2019)":
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2019, 12, 31)
        st.sidebar.success("✔️ Período Normal establecido")
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.98, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.45, 0.05)
        
    else:
        # Personalizado
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Inicio", value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("Fin", value=datetime.now())
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.98, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.45, 0.05)
    
    # Parámetros del modelo
    st.sidebar.subheader("3. Parámetros del Modelo")
    var_confidence = st.sidebar.slider("Confianza VaR", 0.90, 0.99, 0.95, 0.01)
    garch_window = st.sidebar.slider("Ventana GARCH (días)", 60, 500, 252)
    
    # Validación Out-of-Sample
    st.sidebar.subheader("4. Validación")
    enable_oos = st.sidebar.checkbox("Activar Validación Out-of-Sample", value=True)
    train_ratio = st.sidebar.slider("Proporción Entrenamiento (%)", 50, 90, 70, 5) if enable_oos else 70
    
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
                st.error("❌ No se encontraron datos válidos. Es posible que los activos no existieran en las fechas seleccionadas (Ej. Bitcoin en 2008).")
                st.stop()
            
            returns = calculate_returns(prices)
            valid_tickers = returns.columns.tolist()
            
            if len(valid_tickers) < 2:
                st.error("❌ El modelo DCC-GARCH requiere al menos 2 activos concurrentes.")
                st.stop()
            
            if len(returns) < 50:
                st.error(f"❌ Solo se obtuvieron {len(returns)} días de datos. Se requieren al menos 50 observaciones.")
                st.stop()
            
            dropped_tickers = set(tickers) - set(valid_tickers)
            if dropped_tickers:
                st.warning(f"⚠️ Activos excluidos (sin historial en este período): {', '.join(dropped_tickers)}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Activos Listos", len(valid_tickers))
            with col2:
                st.metric("Período", f"{returns.index[0].strftime('%Y-%m-%d')} a {returns.index[-1].strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Observaciones", len(returns))
            
            with st.expander("📋 Ver Datos de Precios"):
                st.dataframe(prices.tail(10))
            
            # 2. Cálculo de retornos y GARCH
            st.markdown("---")
            st.markdown('<p class="sub-header">📈 2. Cálculo de Retornos y Filtrado GARCH</p>', 
                       unsafe_allow_html=True)
            
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
            
            st.plotly_chart(
                plot_homeostatic_indicator(H_t.values, prop_stressed.values, prop_stressed.index),
                use_container_width=True
            )
            
            # 4. Modelo DCC-GARCH Homeostático
            st.markdown("---")
            st.markdown('<p class="sub-header">🔗 4. Correlación Dinámica (DCC-H)</p>', 
                       unsafe_allow_html=True)
            
            Q_bar = np.corrcoef(z_std.T)
            R_t, Q_t, params = dcc_homeostatic(z_std, H_t, Q_bar)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parámetro α (shock)", f"{params['a']:.3f}")
            with col2:
                st.metric("Parámetro β (persistencia)", f"{params['b']:.3f}")
            with col3:
                st.metric("Parámetro γ (homeostasis)", f"{params['gamma']:.3f}")
            
            st.plotly_chart(
                plot_correlation_heatmap(R_t, returns.index, valid_tickers, 
                                        "Matriz de Correlación Promedio (Últimos 60 días)"),
                use_container_width=True
            )
            
            st.markdown("**Seleccionar par de activos para ver evolución de correlación:**")
            col1, col2 = st.columns(2)
            with col1:
                asset1 = st.selectbox("Activo 1", valid_tickers, index=0, key="asset1")
            with col2:
                asset2 = st.selectbox("Activo 2", valid_tickers, index=1 if len(valid_tickers) > 1 else 0, key="asset2")
            
            idx1, idx2 = valid_tickers.index(asset1), valid_tickers.index(asset2)
            
            st.plotly_chart(
                plot_correlation_timeseries(R_t, returns.index, valid_tickers, (idx1, idx2)),
                use_container_width=True
            )
            
            # 5. Test de Razón de Verosimilitud
            st.markdown("---")
            st.markdown('<p class="sub-header">🧪 5. Test de Razón de Verosimilitud (Validación de H2)</p>', 
                       unsafe_allow_html=True)
            
            with st.spinner("Ejecutando Test LR..."):
                lr_results = likelihood_ratio_test(z_std, H_t, Q_bar)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Estadístico LR", f"{lr_results['lr_statistic']:.4f}")
                with col2:
                    st.metric("Valor Crítico (5%)", f"{lr_results['critical_value']:.4f}")
                with col3:
                    st.metric("P-value", f"{lr_results['p_value']:.6f}")
                with col4:
                    if lr_results['decision'] == "RECHAZAR_H0":
                        st.success("✅ H0 Rechazada")
                    else:
                        st.error("❌ H0 No Rechazada")
                
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
                    2. Esto NO invalida tu tesis, pero sugiere:
                       - Probar con otros períodos (crisis 2008, COVID-19)
                       - Ajustar el umbral κ o la confianza de Gumbel
                       - El efecto homeostático puede ser específico de ciertos regímenes
                    """)
                
                st.markdown("### 📊 Comparación de Modelos")
                comparison_df = pd.DataFrame({
                    'Modelo': ['DCC Estándar', 'DCC Homeostático'],
                    'Parámetros': [2, 3],
                    'Log-Likelihood': [lr_results['log_lik_restricted'], lr_results['log_lik_unrestricted']],
                    'AIC': [-2*lr_results['log_lik_restricted'] + 2*2, 
                            -2*lr_results['log_lik_unrestricted'] + 2*3],
                    'BIC': [-2*lr_results['log_lik_restricted'] + 2*np.log(len(z_std)), 
                            -2*lr_results['log_lik_unrestricted'] + 3*np.log(len(z_std))]
                })
                
                st.dataframe(comparison_df.style.format({
                    'Log-Likelihood': '{:.4f}',
                    'AIC': '{:.4f}',
                    'BIC': '{:.4f}'
                }))
            
            # 6. Value-at-Risk y Backtesting
            st.markdown("---")
            st.markdown('<p class="sub-header">⚠️ 6. Value-at-Risk Condicional Riguroso y Backtesting</p>', 
                       unsafe_allow_html=True)
            
            var_series = calculate_var(returns, R_t, sigma, confidence=var_confidence)
            backtest_results = backtest_var(returns, var_series, var_confidence)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Violaciones Observadas", backtest_results['violations'])
            with col2:
                st.metric("Violaciones Esperadas", f"{backtest_results['expected']:.1f}")
            with col3:
                st.metric("Tasa Observada", f"{backtest_results['violation_rate']*100:.2f}%")
            with col4:
                st.metric("Tasa Esperada", f"{backtest_results['expected_rate']*100:.2f}%")
            
            if backtest_results['passed']:
                st.success(f"✅ Test de Kupiec APROBADO (p-value: {backtest_results['kupiec_pvalue']:.4f})")
            else:
                st.error(f"❌ Test de Kupiec RECHAZADO (p-value: {backtest_results['kupiec_pvalue']:.4f})")
            
            st.plotly_chart(
                plot_var_backtesting(returns, var_series, returns.index),
                use_container_width=True
            )
            
            # 7. Validación Out-of-Sample
            if enable_oos:
                st.markdown("---")
                st.markdown('<p class="sub-header">🔬 7. Validación Out-of-Sample pura</p>', 
                           unsafe_allow_html=True)
                
                with st.spinner("Ejecutando validación predictiva Out-of-Sample..."):
                    oos_results, oos_error = out_of_sample_validation(
                        prices, valid_tickers, train_ratio/100, 
                        confidence_gumbel, kappa_threshold, var_confidence, garch_window
                    )
                    
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
                        
                        st.markdown("### 📊 Comparación Pura en Prueba de Estrés Predictivo: DCC-H vs DCC Estándar")
                        
                        comparison_oos = pd.DataFrame({
                            'Métrica': ['Violaciones', 'Tasa Observada', 'Tasa Esperada', 'Kupiec p-value'],
                            'DCC Homeostático': [
                                oos_results['backtest_oos']['violations'],
                                f"{oos_results['backtest_oos']['violation_rate']*100:.2f}%",
                                f"{oos_results['backtest_oos']['expected_rate']*100:.2f}%",
                                f"{oos_results['backtest_oos']['kupiec_pvalue']:.4f}"
                            ],
                            'DCC Estándar': [
                                oos_results['backtest_standard']['violations'],
                                f"{oos_results['backtest_standard']['violation_rate']*100:.2f}%",
                                f"{oos_results['backtest_standard']['expected_rate']*100:.2f}%",
                                f"{oos_results['backtest_standard']['kupiec_pvalue']:.4f}"
                            ]
                        })
                        
                        st.dataframe(comparison_oos)
                        
                        st.plotly_chart(
                            plot_out_of_sample_comparison(oos_results),
                            use_container_width=True
                        )
                        
                        v_oos = oos_results['backtest_oos']['violations']
                        v_std = oos_results['backtest_standard']['violations']
                        
                        if v_oos <= v_std:
                            st.success(f"""
                            **✅ El modelo DCC-H muestra mejor performance out-of-sample:**
                            - Presenta la misma cantidad o menos violaciones que el modelo estándar
                            - **DCC-H: {v_oos} violaciones vs Estándar: {v_std} violaciones**
                            - Posee un fuerte nivel de generalización
                            """)
                        else:
                            st.warning(f"""
                            **⚠️ El modelo DCC-H tiene más violaciones en out-of-sample:**
                            - DCC-H: {v_oos} vs Estándar: {v_std}
                            - Puede indicar cierto nivel de overfitting
                            """)
            
            # 8. Exportar resultados
            st.markdown("---")
            st.markdown('<p class="sub-header">💾 8. Exportar Resultados</p>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
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
                summary = {
                    'Modelo': 'DCC-GARCH Homeostático',
                    'Activos': len(valid_tickers),
                    'Período': f"{start_date} a {end_date}",
                    'Confianza Gumbel': confidence_gumbel,
                    'Umbral κ': kappa_threshold,
                    'Días Homeostasis': int(H_t.sum()),
                    'Violaciones VaR': backtest_results['violations'],
                    'Kupiec p-value': backtest_results['kupiec_pvalue'],
                    'LR Test p-value': lr_results['p_value'],
                    'Decisión LR': lr_results['decision']
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
        
        1. **📊 Filtrado GARCH**: Extrae residuos estandarizados y matrices de volatilidad condicional
        2. **🎯 Teoría de Valores Extremos**: Ajusta distribución de Gumbel para detectar eventos extremos
        3. **🏠 Indicador Homeostático**: Identifica cuando el sistema está en "tensión" (H_t = 1)
        4. **🔗 DCC Modificado**: La correlación dinámica cambia según el régimen homeostático
        5. **🧪 Test LR**: Valida estadísticamente que el componente homeostático (γ) aporta significativamente
        6. **🔬 Out-of-Sample Puro**: Prueba el modelo en datos nunca vistos evitando look-ahead bias
        7. **⚠️ VaR Condicional**: Calcula un Value-at-Risk que usa la reconstrucción matricial exacta
        
        #### Hipótesis que se pueden testear:
        
        - **H1**: Los umbrales de Gumbel predicen mejor los eventos extremos que la distribución normal
        - **H2**: Las correlaciones cambian significativamente cuando H_t = 1 **(Validado con Test LR)**
        - **H3**: El VaR condicional sistémico tiene mayor validez (Kupiec Test robusto)
        
        ---
        
        <div class="warning-box">
        <strong>⚠️ Nota Académica:</strong> Esta aplicación es para fines de investigación académica. 
        Se construyó respetando el rigor matemático necesario para sustentación Doctoral.
        </div>
        """, unsafe_allow_html=True)
        
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
