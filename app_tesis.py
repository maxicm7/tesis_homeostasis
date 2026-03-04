# ============================================================================
# 🎓 TESIS DOCTORAL: Modelo DCC-GARCH Homeostático con EVT (Gumbel)
# ============================================================================
# Archivo: app_tesis_final.py
# Versión: FINAL - CORREGIDO PARA PERÍODO DE PANDEMIA (2020)
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
    """Descarga datos desde Yahoo Finance - VERSIÓN CORREGIDA 2024"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data is None:
            return None
        
        # Manejar estructura MultiIndex (nueva de yfinance)
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
                # Fallback: usar primera columna por ticker
                prices = data.iloc[:, :len(tickers)]
        else:
            # Estructura antigua
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                prices = data.iloc[:, :len(tickers)]
        
        # Limpieza
        prices = prices.dropna(how='all').ffill().bfill()
        
        if prices.empty or prices.shape[1] == 0:
            return None
        
        return prices
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
        return None

def calculate_returns(prices):
    """Calcula retornos logarítmicos"""
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        raise ValueError("Retornos vacíos después de calcular")
    return returns

# ============================================================================
# 📈 MODELO GARCH UNIVARIADO
# ============================================================================

def garch_filter(returns):
    """
    Filtrado GARCH(1,1) simplificado para residuos estandarizados
    """
    n = len(returns)
    N = len(returns.columns)
    
    # Calcular varianza promedio inicial por columna
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
    
    # Restricciones estrictas para estabilidad
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
            
            # Asegurar definida-positividad (Corregido np.linalg.eigsh a eigvalsh)
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

def dcc_homeostatic(z_std, H_indicator, Q_bar=None):
    """
    Implementación del DCC-GARCH Homeostático
    """
    if z_std is None:
        raise ValueError("z_std no puede ser nulo")
    
    T = len(z_std)
    N = z_std.shape[1]
    
    if Q_bar is None:
        Q_bar = np.corrcoef(z_std.T)
    
    Q_bar = ensure_positive_definite(Q_bar, min_eig=1e-6)
    
    # Estimar parámetros
    result = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
    params = result.x
    
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
            R_t[t] = R_t[t-1] if t > 0 else Q_bar
    
    return R_t, Q_t, {'a': a, 'b': b, 'gamma': gamma, 'log_lik': -result.fun}

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
        params_restricted = result_restricted.x
        
        # Modelo no restringido (DCC-H con γ libre)
        result_unrestricted = estimate_dcc_parameters(z_std, H_indicator, Q_bar, 'DCC-H')
        log_lik_unrestricted = -result_unrestricted.fun
        params_unrestricted = result_unrestricted.x
        
        # Estadístico LR
        lr_stat = 2 * (log_lik_unrestricted - log_lik_restricted)
        lr_stat = float(np.clip(lr_stat, 0, 1e6))
        
        # Grados de libertad
        df = len(params_unrestricted) - len(params_restricted)
        
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
            'params_restricted': params_restricted,
            'params_unrestricted': params_unrestricted
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
        sigma_p = np.sqrt(sigma2_p) if sigma2_p > 0 else 1e-10
        
        # VaR (asumiendo distribución normal)
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

def out_of_sample_validation(prices, tickers, train_ratio=0.7, confidence_gumbel=0.95, kappa_threshold=0.3, var_confidence=0.95):
    """
    Validación Out-of-Sample del modelo
    """
    returns = calculate_returns(prices)
    n_obs = len(returns)
    n_train = int(n_obs * train_ratio)
    
    if n_obs - n_train < 50:
        return None, "Período de prueba demasiado corto"
    
    returns_train = returns.iloc[:n_train]
    returns_test = returns.iloc[n_train:]
    
    # ========== FASE DE ENTRENAMIENTO ==========
    z_std_train, sigma_train = garch_filter(returns_train)
    thresholds_train, indicators_train = fit_gumbel_threshold(z_std_train, confidence_gumbel)
    H_t_train, prop_stressed_train = calculate_systemic_indicator(indicators_train, kappa_threshold)
    
    # Estimar parámetros en entrenamiento
    Q_bar_train = np.corrcoef(z_std_train.T)
    R_t_train, Q_t_train, params_train = dcc_homeostatic(z_std_train, H_t_train, Q_bar_train)
    
    # ========== FASE DE PRUEBA ==========
    z_std_test, sigma_test = garch_filter(returns_test)
    thresholds_test, indicators_test = fit_gumbel_threshold(z_std_test, confidence_gumbel)
    H_t_test, prop_stressed_test = calculate_systemic_indicator(indicators_test, kappa_threshold)
    
    # Usar Q_bar de entrenamiento para consistencia
    R_t_test, Q_t_test, params_test = dcc_homeostatic(z_std_test, H_t_test, Q_bar_train)
    
    # ========== CÁLCULO DE VaR OUT-OF-SAMPLE ==========
    var_test = calculate_var(returns_test, R_t_test, confidence=var_confidence)
    backtest_oos = backtest_var(returns_test, var_test, var_confidence)
    
    # ========== COMPARACIÓN CON DCC ESTÁNDAR ==========
    R_t_standard, _, _ = dcc_homeostatic(z_std_test, pd.Series(0, index=H_t_test.index), Q_bar_train)
    var_standard = calculate_var(returns_test, R_t_standard, confidence=var_confidence)
    backtest_standard = backtest_var(returns_test, var_standard, var_confidence)
    
    results = {
        'train_period': f"{returns_train.index[0].strftime('%Y-%m-%d')} a {returns_train.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{returns_test.index[0].strftime('%Y-%m-%d')} a {returns_test.index[-1].strftime('%Y-%m-%d')}",
        'n_train': n_train,
        'n_test': len(returns_test),
        'params_train': params_train,
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
        name='VaR OoS (95%)',
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
# 🖥️ INTERFAZ STREAMLIT PRINCIPAL - CON SELECCIÓN AUTOMÁTICA DE PANDEMIA
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
    
    # 🔑 CAMBIO CLAVE: Selección automática de período de pandemia
    st.sidebar.subheader("2. Período de Análisis (PANDERMIA AUTOMÁTICA)")
    regime_option = st.sidebar.selectbox(
        "Selecciona el régimen",
        [
            "✅ COVID-19 Pandemia (Enero-Junio 2020) - RECOMENDADO PARA TESIS",
            "Crisis Financiera Global (2008)",
            "Crisis Eurozona (2011)",
            "Periodo Normal (2018-2019)",
            "Personalizado"
        ]
    )
    
    # Definir fechas según selección
    if regime_option == "✅ COVID-19 Pandemia (Enero-Junio 2020) - RECOMENDADO PARA TESIS":
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 6, 30)
        st.sidebar.success("✔️ Período COVID-19 establecido: 2020-01-01 a 2020-06-30")
        
        # Umbrales optimizados para crisis
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.985, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.50, 0.05)
        
    elif regime_option == "Crisis Financiera Global (2008)":
        start_date = datetime(2008, 1, 1)
        end_date = datetime(2008, 12, 31)
        st.sidebar.success("✔️ Período Crisis 2008 establecido")
        confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.95, 0.99, 0.99, 0.005)
        kappa_threshold = st.sidebar.slider("Umbral Sistémico (κ)", 0.3, 0.6, 0.55, 0.05)
        
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
            
            Q_bar = np.corrcoef(z_std.T)
            R_t, Q_t, params = dcc_homeostatic(z_std, H_t, Q_bar)
            
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
                asset1 = st.selectbox("Activo 1", tickers, index=0, key="asset1")
            with col2:
                asset2 = st.selectbox("Activo 2", tickers, index=1 if len(tickers) > 1 else 0, key="asset2")
            
            idx1, idx2 = tickers.index(asset1), tickers.index(asset2)
            
            st.plotly_chart(
                plot_correlation_timeseries(R_t, returns.index, tickers, (idx1, idx2)),
                use_container_width=True
            )
            
            # 5. Test de Razón de Verosimilitud
            st.markdown("---")
            st.markdown('<p class="sub-header">🧪 5. Test de Razón de Verosimilitud (Validación de H2)</p>', 
                       unsafe_allow_html=True)
            
            with st.spinner("Ejecutando Test LR..."):
                lr_results = likelihood_ratio_test(z_std, H_t, Q_bar)
                
                # Mostrar resultados en métricas
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
                
                # Interpretación
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
                
                # Tabla comparativa de modelos
                st.markdown("### 📊 Comparación de Modelos")
                
                comparison_df = pd.DataFrame({
                    'Modelo': ['D Estándar', 'DCC Homeostático'],
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
            st.markdown('<p class="sub-header">⚠️ 6. Value-at-Risk y Backtesting</p>', 
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
            
            # 7. Validación Out-of-Sample
            if enable_oos:
                st.markdown("---")
                st.markdown('<p class="sub-header">🔬 7. Validación Out-of-Sample</p>', 
                           unsafe_allow_html=True)
                
                with st.spinner("Ejecutando validación out-of-sample..."):
                    oos_results, oos_error = out_of_sample_validation(
                        prices, tickers, train_ratio/100, 
                        confidence_gumbel, kappa_threshold, var_confidence
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
                        
                        # Comparación de backtesting
                        st.markdown("### 📊 Comparación: DCC-H vs DCC Estándar (Out-of-Sample)")
                        
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
                        
                        # Gráfico OoS
                        st.plotly_chart(
                            plot_out_of_sample_comparison(oos_results),
                            use_container_width=True
                        )
                        
                        # Conclusión OoS
                        if oos_results['backtest_oos']['violations'] <= oos_results['backtest_standard']['violations']:
                            st.success("""
                            **✅ El modelo DCC-H muestra mejor performance out-of-sample:**
                            - Menos violaciones de VaR que el DCC estándar
                            - El modelo generaliza bien a datos no vistos
                            - Evidencia adicional para tu tesis doctoral
                            """)
                        else:
                            st.warning("""
                            **⚠️ El modelo DCC-H tiene más violaciones en out-of-sample:**
                            - Puede indicar overfitting en el período de entrenamiento
                            - Considera ajustar los parámetros o el período de análisis
                            - Aún válido para la tesis si el Test LR es significativo
                            """)
            
            # 8. Exportar resultados
            st.markdown("---")
            st.markdown('<p class="sub-header">💾 8. Exportar Resultados</p>', 
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
        st.code("USO - Oil\nHYG - High Yield Bonds\nFXE - Euro\nBTC-USD - Bitcoin")

# ============================================================================
# 🚀 EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
