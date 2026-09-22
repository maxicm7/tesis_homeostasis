# ============================================================================
# 🎓 TESIS DOCTORAL: Modelo DCC-GARCH Homeostático con EVT (Gumbel)
# ============================================================================
# Archivo: app_tesis.py
# Versión: FINAL CON CLASIFICADOR DE FASES AUTOMÁTICO
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
    .phase-badge {display: inline-block; padding: 8px 16px; border-radius: 20px; 
                  font-weight: bold; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 🎯 NUEVA FUNCIÓN: CLASIFICADOR AUTOMÁTICO DE FASES
# ============================================================================

def clasificar_fase(lr_pvalue, kupiec_pvalue, dias_ht_percentage, volatilidad_anual=None):
    """
    Clasifica automáticamente la fase del mercado basada en múltiples métricas
    
    Parámetros:
    - lr_pvalue: P-value del Test LR (significancia de γ)
    - kupiec_pvalue: P-value del Test de Kupiec (calibración del VaR)
    - dias_ht_percentage: Porcentaje de días con H_t = 1
    - volatilidad_anual: Volatilidad anualizada (opcional)
    
    Retorna:
    - dict con fase, descripcion, color, y recomendaciones
    """
    
    # Fase 1: Estabilidad (Homeostasis Continua)
    if (lr_pvalue < 0.05 and 
        kupiec_pvalue > 0.05 and 
        dias_ht_percentage < 2):
        return {
            'fase': 'FASE 1: ESTABILIDAD',
            'subfase': 'Homeostasis Continua',
            'color': '#28a745',  # Verde
            'descripcion': 'Los mecanismos de corrección homeostática operan de forma continua y predecible. El VaR está bien calibrado.',
            'caracteristicas': [
                '✓ γ altamente significativo',
                '✓ VaR confiable',
                '✓ Tensión sistémica baja',
                '✓ Correlaciones estables'
            ],
            'recomendaciones': [
                '• Mantener estrategia de diversificación tradicional',
                '• VaR paramétrico suficiente para gestión de riesgo',
                '• Monitoreo estándar'
            ]
        }
    
    # Fase 2: Shock Exógeno (Homeostasis Selectiva)
    elif (lr_pvalue < 0.05 and 
          kupiec_pvalue > 0.05 and 
          2 <= dias_ht_percentage <= 10):
        return {
            'fase': 'FASE 2: SHOCK EXÓGENO',
            'subfase': 'Homeostasis Selectiva',
            'color': '#ffc107',  # Amarillo
            'descripcion': 'Shock externo (geopolítico, sanitario) activa mecanismos homeostáticos de forma selectiva. El sistema absorbe el shock.',
            'caracteristicas': [
                '✓ γ significativo',
                '✓ VaR confiable',
                '⚠ Tensión sistémica moderada',
                '✓ Mecanismos de corrección operativos'
            ],
            'recomendaciones': [
                '• Revisar coberturas tail risk',
                '• Mantener VaR paramétrico pero monitorear violaciones',
                '• Activar protocolos de crisis si H_t persiste >5 días'
            ]
        }
    
    # Fase 3: Saturación (Mecanismos Comprometidos)
    elif (lr_pvalue >= 0.05 and 
          kupiec_pvalue < 0.05 and 
          dias_ht_percentage > 10):
        return {
            'fase': 'FASE 3: SATURACIÓN',
            'subfase': 'Mecanismos Comprometidos',
            'color': '#dc3545',  # Rojo
            'descripcion': 'CRISIS SISTÉMICA: Los mecanismos de corrección están saturados. El VaR subestima el riesgo real. Correlaciones colapsan hacia 1.',
            'caracteristicas': [
                '✗ γ NO significativo',
                '✗ VaR subestima riesgo',
                '✗ Tensión sistémica muy alta',
                '✗ Correlaciones → 1 (contagio)'
            ],
            'recomendaciones': [
                '• ACTIVAR PROTOCOLO DE CRISIS',
                '• Complementar VaR con stress tests',
                '• Reducir exposición inmediatamente',
                '• Aumentar liquidez',
                '• Reportar a comité de riesgo/regulador'
            ]
        }
    
    # Fase 4: Transición (Homeostasis Dispersa)
    elif (lr_pvalue >= 0.05 and 
          kupiec_pvalue > 0.05 and 
          5 < dias_ht_percentage <= 15):
        return {
            'fase': 'FASE 4: TRANSICIÓN',
            'subfase': 'Homeostasis Dispersa',
            'color': '#17a2b8',  # Cian/Azul
            'descripcion': 'Período de ajuste post-crisis o búsqueda de nuevo equilibrio. La corrección existe pero de forma dispersa y no sistemática.',
            'caracteristicas': [
                '✗ γ NO significativo',
                '✓ VaR confiable',
                '⚠ Tensión sistémica moderada-alta',
                '⚠ Correlaciones variables'
            ],
            'recomendaciones': [
                '• Mantener flexibilidad en asignación',
                '• Monitorear cambios de régimen',
                '• Evitar posiciones muy concentradas',
                '• Prepararse para posible escalada'
            ]
        }
    
    # Fase Indeterminada
    else:
        return {
            'fase': 'FASE INDETERMINADA',
            'subfase': 'Requiere Análisis Manual',
            'color': '#6c757d',  # Gris
            'descripcion': 'El patrón no coincide claramente con ninguna fase definida. Se requiere análisis cualitativo adicional.',
            'caracteristicas': [
                '? Patrón mixto',
                '? Métricas contradictorias'
            ],
            'recomendaciones': [
                '• Revisar contexto macroeconómico',
                '• Consultar con analistas senior',
                '• Considerar factores exógenos no capturados'
            ]
        }


def mostrar_fase_detectada(fase_info):
    """
    Muestra visualmente la fase detectada en la interfaz
    """
    # Badge de fase con color
    st.markdown(f"""
    <div style='background-color: {fase_info["color"]}; color: white; 
                padding: 15px; border-radius: 10px; margin: 20px 0;'>
        <h2 style='margin: 0; color: white;'>🎯 {fase_info["fase"]}</h2>
        <h4 style='margin: 5px 0; color: white;'>{fase_info["subfase"]}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Descripción
    st.info(f"**Descripción:** {fase_info['descripcion']}")
    
    # Características
    st.markdown("**📋 Características Detectadas:**")
    for caracteristica in fase_info['caracteristicas']:
        st.markdown(f"- {caracteristica}")
    
    # Recomendaciones
    st.markdown("**✅ Recomendaciones:**")
    for recomendacion in fase_info['recomendaciones']:
        st.markdown(f"- {recomendacion}")


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
# 📈 MODELO GARCH UNIVARIADO — ESTIMADO POR MLE, POR ACTIVO (Sección 3.2.3)
# ============================================================================
#
# CORRECCIÓN (punto 4 de la adenda doctoral): los parámetros (omega, alpha,
# beta) ya NO se fijan igual para todos los activos. Se estiman individualmente
# por Máxima Verosimilitud (QMLE) para cada serie, con diagnósticos de
# Ljung-Box y ARCH-LM post-estimación. Fijar los mismos parámetros para
# activos tan distintos como equity, bonos, oro o divisas sesgaría los
# residuos estandarizados z_{i,t} que alimentan Gumbel y el DCC-H.
# ============================================================================

def _garch11_neg_loglik(theta, r):
    """Log-verosimilitud gaussiana negativa de un GARCH(1,1) univariado."""
    omega, alpha, beta = theta
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(r) if np.var(r) > 1e-12 else 1e-6
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t-1] ** 2 + beta * sigma2[t-1]
    sigma2 = np.clip(sigma2, 1e-12, None)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + r ** 2 / sigma2)
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit_garch11_mle(r):
    """
    Estima GARCH(1,1) por QMLE para una serie de retornos individual.
    Escala los retornos (x100) para evitar mal condicionamiento numérico
    con omega muy pequeño, y usa multi-start para evitar óptimos locales
    degenerados (alpha->1, beta->0), un problema frecuente en GARCH con
    optimización de un solo punto de partida.
    """
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    scale = 100.0
    rs = r * scale

    bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]
    cons = ({'type': 'ineq', 'fun': lambda th: 0.999 - (th[1] + th[2])},)
    var_rs = np.var(rs) if np.var(rs) > 1e-8 else 1.0

    starts = [
        (var_rs * 0.05, 0.05, 0.90),
        (var_rs * 0.10, 0.10, 0.85),
        (var_rs * 0.05, 0.02, 0.95),
        (var_rs * 0.20, 0.15, 0.75),
    ]

    best_res = None
    for s0 in starts:
        try:
            res = minimize(_garch11_neg_loglik, x0=np.array(s0), args=(rs,),
                            method='SLSQP', bounds=bounds, constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-12})
        except Exception:
            continue
        if res is None or not np.isfinite(res.fun):
            continue
        if best_res is None or (res.success and not best_res.success) or \
           (res.success == best_res.success and res.fun < best_res.fun):
            best_res = res

    if best_res is None:
        # Fallback conservador si ningún arranque converge
        omega, alpha, beta = 0.05 * var_rs, 0.08, 0.85
        converged = False
        loglik = np.nan
    else:
        omega, alpha, beta = best_res.x
        omega = omega / (scale ** 2)  # des-escalar
        converged = bool(best_res.success)
        loglik = -best_res.fun

    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(r) if np.var(r) > 1e-12 else 1e-6
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t-1] ** 2 + beta * sigma2[t-1]
    sigma = np.sqrt(np.clip(sigma2, 1e-12, None))
    z = r / sigma

    return {
        'omega': float(omega), 'alpha': float(alpha), 'beta': float(beta),
        'persistence': float(alpha + beta),
        'sigma2': sigma2, 'sigma': sigma, 'z': z,
        'loglik': loglik, 'converged': converged,
        'sigma2_last': float(sigma2[-1]), 'resid_last': float(r[-1]),
    }


def ljung_box_test(x, lags=10):
    """Test de Ljung-Box (manual, sin dependencias externas) sobre una serie."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= lags + 1:
        return {'stat': np.nan, 'p_value': np.nan}
    x = x - np.mean(x)
    denom = np.sum(x ** 2)
    if denom < 1e-12:
        return {'stat': np.nan, 'p_value': np.nan}
    stat = 0.0
    for k in range(1, lags + 1):
        rk = np.sum(x[k:] * x[:-k]) / denom
        stat += (rk ** 2) / (n - k)
    stat *= n * (n + 2)
    p_value = 1 - chi2.cdf(stat, lags)
    return {'stat': float(stat), 'p_value': float(p_value)}


def arch_lm_test(x, lags=5):
    """
    Test ARCH-LM (manual): regresión de x_t^2 sobre sus 'lags' rezagos.
    Estadístico = n * R^2 ~ chi2(lags) bajo H0 (sin efectos ARCH residuales).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x2 = x ** 2
    n = len(x2)
    if n <= lags + 5:
        return {'stat': np.nan, 'p_value': np.nan}
    Y = x2[lags:]
    X = np.column_stack([x2[lags - k: n - k] for k in range(1, lags + 1)])
    X = np.column_stack([np.ones(len(Y)), X])
    try:
        beta_hat, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Y_hat = X @ beta_hat
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        stat = len(Y) * r2
        p_value = 1 - chi2.cdf(stat, lags)
        return {'stat': float(stat), 'p_value': float(p_value)}
    except Exception:
        return {'stat': np.nan, 'p_value': np.nan}


def garch_filter(returns):
    """
    Filtrado GARCH(1,1) con parámetros (omega_i, alpha_i, beta_i) estimados
    por MLE de forma individual para cada activo (Sección 3.2.3).

    Retorna:
    - z_std: DataFrame (T x N) de residuos estandarizados
    - sigma_matrix: array (T x N) de volatilidades condicionales
    - garch_params_df: DataFrame con parámetros y diagnósticos por activo
    - garch_state: dict {ticker: {'sigma2_last':..., 'resid_last':..., params...}}
      para continuar el filtro sin re-estimar (usado en validación Out-of-Sample)
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    n = len(returns)
    N = len(returns.columns)

    sigma_matrix = np.zeros((n, N))
    z_std_list = []
    rows = []
    garch_state = {}

    for i, col in enumerate(returns.columns):
        r = returns[col].values
        fit = fit_garch11_mle(r)

        sigma_matrix[:, i] = fit['sigma']
        z_std_list.append(fit['z'])

        lb = ljung_box_test(fit['z'], lags=10)
        lb2 = ljung_box_test(fit['z'] ** 2, lags=10)
        lm = arch_lm_test(fit['z'], lags=5)

        rows.append({
            'Ticker': col,
            'omega': fit['omega'], 'alpha': fit['alpha'], 'beta': fit['beta'],
            'persistencia (a+b)': fit['persistence'],
            'Convergió': fit['converged'],
            'Ljung-Box p (z)': lb['p_value'],
            'Ljung-Box p (z²)': lb2['p_value'],
            'ARCH-LM p': lm['p_value'],
        })

        garch_state[col] = {
            'omega': fit['omega'], 'alpha': fit['alpha'], 'beta': fit['beta'],
            'sigma2_last': fit['sigma2_last'], 'resid_last': fit['resid_last'],
        }

    z_std = pd.DataFrame(np.column_stack(z_std_list), index=returns.index, columns=returns.columns)
    garch_params_df = pd.DataFrame(rows)

    return z_std, sigma_matrix, garch_params_df, garch_state


def garch_filter_apply_fixed(returns, garch_state):
    """
    Aplica parámetros GARCH(1,1) YA ESTIMADOS (congelados) a un nuevo tramo
    de retornos, continuando la recursión de sigma² desde el último estado
    conocido (sigma2_last, resid_last), sin reestimar y sin reinicializar
    con la varianza incondicional del propio tramo nuevo.

    Esto es lo que corresponde usar durante la proyección Out-of-Sample:
    los parámetros y el estado se "congelan" en el momento de entrenamiento
    y se propagan hacia adelante, como ocurriría en un despliegue real.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    n = len(returns)
    N = len(returns.columns)
    sigma_matrix = np.zeros((n, N))
    z_std_list = []

    for i, col in enumerate(returns.columns):
        st_i = garch_state[col]
        omega, alpha, beta = st_i['omega'], st_i['alpha'], st_i['beta']
        r = returns[col].values

        sigma2 = np.empty(n)
        sigma2_prev = st_i['sigma2_last']
        resid_prev = st_i['resid_last']
        for t in range(n):
            sigma2[t] = omega + alpha * resid_prev ** 2 + beta * sigma2_prev
            sigma2_prev = sigma2[t]
            resid_prev = r[t]

        sigma = np.sqrt(np.clip(sigma2, 1e-12, None))
        sigma_matrix[:, i] = sigma
        z_std_list.append(r / sigma)

    z_std = pd.DataFrame(np.column_stack(z_std_list), index=returns.index, columns=returns.columns)
    return z_std, sigma_matrix

# ============================================================================
# 🎯 DISTRIBUCIÓN DE GUMBEL
# ============================================================================

def fit_gumbel_threshold(residuals, confidence=0.95, window=252):
    """
    Ajusta distribución de Gumbel y calcula umbrales de tensión homeostática.

    CORRECCIÓN (punto 2 / anti-look-ahead): el umbral ya NO es un valor único
    promediado sobre TODAS las ventanas de la muestra (lo cual usaba
    implícitamente información futura para clasificar días pasados). Ahora
    el umbral τ_t es estrictamente causal y variable en el tiempo: en cada t
    se ajusta Gumbel únicamente con la ventana [t-window, t-1], y h_{i,t} se
    evalúa contra ESE umbral específico de t (Sección 3.3.2 de la tesis).
    Los primeros `window` días no tienen umbral definido (burn-in) y quedan
    con indicador 0 por defecto.

    Retorna:
    - thresholds_summary: dict {ticker: umbral promedio en el tiempo} — solo
      para fines de visualización resumida en la interfaz.
    - indicators: DataFrame (T x N) binario, causal.
    - threshold_ts: DataFrame (T x N) con el umbral τ_t completo, variable en
      el tiempo (NaN durante el burn-in).
    """
    indicators = pd.DataFrame(0, index=residuals.index, columns=residuals.columns)
    threshold_ts = pd.DataFrame(np.nan, index=residuals.index, columns=residuals.columns)
    thresholds_summary = {}

    for col in residuals.columns:
        abs_res = np.abs(residuals[col]).values
        n = len(abs_res)
        thresh_col = np.full(n, np.nan)

        for t in range(window, n):
            window_data = abs_res[t - window:t]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) < 30:
                continue
            try:
                loc, scale = gumbel_r.fit(window_data)
                thresh_col[t] = gumbel_r.ppf(confidence, loc=loc, scale=scale)
            except Exception:
                continue

        valid = ~np.isnan(thresh_col)
        ind_col = np.zeros(n, dtype=int)
        ind_col[valid] = (abs_res[valid] > thresh_col[valid]).astype(int)

        indicators[col] = ind_col
        threshold_ts[col] = thresh_col
        thresholds_summary[col] = float(np.nanmean(thresh_col)) if valid.any() else np.nan

    return thresholds_summary, indicators, threshold_ts

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


def compute_recursive_Qstress(z_std, H_indicator, Q_bar, min_obs=30, shrink_c=20.0):
    """
    CORRECCIÓN (punto 2 — anti-circularidad, Sección 3.4.5 de la tesis):
    Q^(S) ya NO se estima una única vez con TODA la muestra (lo cual usaba
    información contemporánea y futura respecto de cada t para construir el
    objetivo hacia el cual el propio parámetro gamma empuja — un artefacto
    de sobreajuste in-sample). Ahora Q^(S)_t se estima de forma RECURSIVA,
    usando únicamente observaciones con H_s=1 para s <= t-2 (estrictamente
    pasado respecto del período que se está actualizando).

    Si no hay al menos `min_obs` observaciones de estrés disponibles hasta
    ese punto (período de calentamiento / burn-in), Q^(S)_t colapsa a Q_bar,
    con lo cual el término gamma se anula automáticamente y el modelo se
    comporta como un DCC estándar hasta acumular evidencia suficiente.

    Se aplica además una contracción (shrinkage, en el espíritu de
    Ledoit-Wolf) hacia Q_bar, con peso decreciente a medida que crece el
    número de observaciones de estrés disponibles — mitiga el problema de
    dimensionalidad cuando N es grande respecto de los días de estrés.

    Retorna: array (T, N, N) — Q^(S)_t a usar en la actualización de Q_t.
    """
    Z = z_std.values if hasattr(z_std, 'values') else np.asarray(z_std)
    H = H_indicator.values if hasattr(H_indicator, 'values') else np.asarray(H_indicator)
    T, N = Z.shape

    cum_outer = np.zeros((N, N))
    cum_count = 0
    cum_outer_hist = [None] * T
    cum_count_hist = [0] * T

    for s in range(T):
        if H[s] == 1:
            zs = Z[s]
            cum_outer = cum_outer + np.outer(zs, zs)
            cum_count += 1
        cum_outer_hist[s] = cum_outer
        cum_count_hist[s] = cum_count

    Qs = np.zeros((T, N, N))
    for t in range(T):
        idx = t - 2  # información disponible estrictamente hasta s = t-2
        if idx < 0 or cum_count_hist[idx] < min_obs:
            Qs[t] = Q_bar
            continue
        n_obs = cum_count_hist[idx]
        avg_outer = cum_outer_hist[idx] / n_obs
        d = np.sqrt(np.clip(np.diag(avg_outer), 1e-8, None))
        corr = avg_outer / np.outer(d, d)
        corr = ensure_positive_definite(corr, min_eig=1e-6)
        delta = min(1.0, shrink_c / (n_obs + shrink_c))
        Qs[t] = (1 - delta) * corr + delta * Q_bar

    return Qs


def dcc_likelihood_full(z_std, H_indicator, Q_bar, params, Qs_precomputed,
                          return_contributions=False):
    """
    Calcula la log-verosimilitud completa del modelo DCC, usando Q^(S)_t
    RECURSIVO (Qs_precomputed[t], ver compute_recursive_Qstress) en lugar de
    una única matriz de estrés estática calculada con toda la muestra.
    """
    T = len(z_std)
    N = z_std.shape[1]

    a = max(params[0], 1e-8)
    b = max(params[1], 1e-8)
    gamma = max(params[2], 1e-8) if len(params) > 2 else 0.0

    # CORRECCIÓN: el límite anterior (a+b+gamma >= 0.98) era demasiado
    # restrictivo. En datos financieros reales la persistencia de la
    # correlación suele ser alta (a+b cercano a 1), por lo que el óptimo
    # verdadero a menudo cae justo por encima de 0.98 — y el optimizador
    # (L-BFGS-B, que solo conoce los bounds de caja, no esta restricción
    # conjunta) podía quedar atrapado devolviendo esta penalización
    # constante en lugar de encontrar el máximo real. El margen correcto
    # es el límite teórico de estacionariedad (a+b+gamma < 1), con un
    # pequeño resguardo numérico.
    if a + b + gamma >= 0.999:
        return (-1000.0, None) if return_contributions else -1000.0

    Z = z_std.values if hasattr(z_std, 'values') else np.asarray(z_std)
    H = H_indicator.values if hasattr(H_indicator, 'values') else np.asarray(H_indicator)

    Q_prev = ensure_positive_definite(Q_bar.copy(), min_eig=1e-4)

    log_lik = 0.0
    count_valid = 0
    contributions = np.zeros(T) if return_contributions else None

    for t in range(1, T):
        try:
            Q_stress_t = Qs_precomputed[t]
            if gamma > 0 and H[t-1] == 1:
                Q_t = (1 - a - b - gamma) * Q_bar + \
                      a * np.outer(Z[t-1], Z[t-1]) + \
                      b * Q_prev + \
                      gamma * Q_stress_t
            else:
                Q_t = (1 - a - b) * Q_bar + \
                      a * np.outer(Z[t-1], Z[t-1]) + \
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

            z_vec = Z[t-1]
            R_inv = np.linalg.inv(R_t)
            quadratic = float(z_vec.T @ R_inv @ z_vec)

            if np.isnan(quadratic) or quadratic > 1000:
                continue

            contribution = -0.5 * (logdet + quadratic)
            log_lik += contribution
            if return_contributions:
                contributions[t] = contribution
            count_valid += 1
            Q_prev = R_t

        except Exception:
            continue

    if count_valid < T * 0.8:
        bad = -10000.0 - (T * count_valid)
        return (bad, contributions) if return_contributions else bad

    if return_contributions:
        return float(log_lik), contributions
    return float(log_lik)


def estimate_dcc_parameters(z_std, H_indicator, Q_bar, Qs_precomputed, model_type='DCC-H'):
    """
    Estima parámetros DCC por máxima verosimilitud, usando Q^(S)_t recursivo.

    CORRECCIÓN: se reemplaza L-BFGS-B (que solo respeta bounds de caja
    individuales) por SLSQP con una restricción explícita a+b+gamma<0.999,
    el mismo enfoque ya usado en fit_garch11_mle. Antes, la restricción
    conjunta vivía únicamente como un salto discontinuo dentro de la
    función de verosimilitud (un "muro" que el optimizador no veía venir),
    lo que podía dejarlo atrapado devolviendo la penalización constante en
    lugar de converger al máximo real — visible como un log-likelihood
    sospechosamente redondo (p. ej. exactamente -1000.0) en el modelo
    resultante. Con la restricción declarada explícitamente, el optimizador
    la respeta durante toda la búsqueda, sin discontinuidades.
    """
    def neg_log_lik(params):
        result = dcc_likelihood_full(z_std, H_indicator, Q_bar, params, Qs_precomputed)
        if np.isinf(result) or np.isnan(result):
            return 1e10
        return -result
    
    if model_type == 'DCC-H':
        initial_params = [0.02, 0.92, 0.02]
        bounds = [(1e-8, 0.3), (0.5, 0.999), (0, 0.3)]
        cons = ({'type': 'ineq', 'fun': lambda p: 0.999 - (p[0] + p[1] + p[2])},)
    else:
        initial_params = [0.02, 0.92]
        bounds = [(1e-8, 0.3), (0.5, 0.999)]
        cons = ({'type': 'ineq', 'fun': lambda p: 0.999 - (p[0] + p[1])},)

    result = minimize(
        neg_log_lik,
        initial_params,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-10}
    )

    # Resguardo: si SLSQP no converge desde el punto de partida por defecto,
    # reintentar con un segundo punto de partida más conservador antes de
    # devolver un resultado potencialmente degenerado.
    if not result.success or result.fun >= 999.0:
        alt_initial = [0.05, 0.80, 0.05] if model_type == 'DCC-H' else [0.05, 0.80]
        result_alt = minimize(
            neg_log_lik, alt_initial, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-10}
        )
        if result_alt.fun < result.fun:
            result = result_alt

    return result


def compute_opg_se(z_std, H_indicator, Q_bar, Qs_precomputed, params, h=1e-4):
    """
    Errores estándar robustos tipo "sandwich" (OPG — outer product of
    gradients, en el espíritu de Bollerslev & Wooldridge, 1992) para los
    parámetros de la Etapa 2 (a, b, gamma). Aproxima la matriz de
    información mediante la suma de productos externos de las
    contribuciones de score por observación (derivadas numéricas de la
    log-verosimilitud por t respecto de cada parámetro).

    Nota de honestidad metodológica (ver Sección 3.5.3 de la tesis): esta
    aproximación corrige la sub-estimación de SE que resulta de ignorar por
    completo la incertidumbre de estimación (como hacía la versión anterior
    de esta app, que no reportaba SE en absoluto), pero NO propaga la
    incertidumbre de la Etapa 1 (parámetros GARCH). Para inferencia
    doctoral completa se recomienda complementar con el bootstrap
    paramétrico descrito en la Sección 3.9 de la tesis.
    """
    k = len(params)
    params = np.array(params, dtype=float)
    T = len(z_std)
    scores = np.zeros((T, k))
    degenerate = [False] * k

    for j in range(k):
        p_plus = params.copy(); p_plus[j] += h
        p_minus = params.copy(); p_minus[j] -= h
        _, c_plus = dcc_likelihood_full(z_std, H_indicator, Q_bar, p_plus, Qs_precomputed,
                                          return_contributions=True)
        _, c_minus = dcc_likelihood_full(z_std, H_indicator, Q_bar, p_minus, Qs_precomputed,
                                           return_contributions=True)
        # Si alguna de las dos evaluaciones perturbadas cae en una región
        # inválida/degenerada, el score de ese parámetro es indefinido —
        # se marca como tal (NaN) en vez de asumir cero, que transmitiría
        # una certeza falsa (SE=0.0000) sobre un resultado en realidad no
        # confiable en ese punto.
        if c_plus is None or c_minus is None:
            degenerate[j] = True
            continue
        diff = c_plus - c_minus
        if np.allclose(diff, 0.0):
            degenerate[j] = True
        scores[:, j] = diff / (2 * h)

    se = np.full(k, np.nan)
    valid_idx = [j for j in range(k) if not degenerate[j]]
    if valid_idx:
        try:
            B = scores.T @ scores
            cov = np.linalg.pinv(B)
            diag = np.diag(cov)
            for j in valid_idx:
                if diag[j] > 0:
                    se[j] = np.sqrt(diag[j])
        except Exception:
            pass
    return se


def dcc_homeostatic(z_std, H_indicator, Q_bar=None, fixed_params=None, Qs_precomputed=None):
    """
    Implementación del DCC-GARCH Homeostático, con Q^(S)_t recursivo
    (anti-circularidad). Permite fijar los parámetros para validaciones
    Out-of-Sample genuinas, y opcionalmente recibir un Qs_precomputed ya
    calculado (por ejemplo, sobre la serie combinada train+test) para
    garantizar continuidad causal entre entrenamiento y prueba.
    """
    if z_std is None or z_std.empty:
        raise ValueError("z_std no puede ser nulo o vacío")
    
    T = len(z_std)
    N = z_std.shape[1]
    
    if Q_bar is None:
        Q_bar = np.corrcoef(z_std.T)
    
    Q_bar = ensure_positive_definite(Q_bar, min_eig=1e-6)

    if Qs_precomputed is None:
        Qs_precomputed = compute_recursive_Qstress(z_std, H_indicator, Q_bar)

    # Estimar parámetros o usar fijos
    if fixed_params is None:
        result = estimate_dcc_parameters(z_std, H_indicator, Q_bar, Qs_precomputed, 'DCC-H')
        params = result.x
        log_lik = -result.fun
    else:
        params = fixed_params
        log_lik = None
    
    a = float(np.clip(params[0], 1e-8, 0.3))
    b = float(np.clip(params[1], 0.5, 0.999))
    gamma = float(np.clip(params[2] if len(params) > 2 else 0.0, 0, 0.3))

    Z = z_std.values if hasattr(z_std, 'values') else np.asarray(z_std)
    H = H_indicator.values if hasattr(H_indicator, 'values') else np.asarray(H_indicator)

    # Evolución de Q_t
    Q_t = np.zeros((T, N, N))
    R_t = np.zeros((T, N, N))
    Q_t[0] = Q_bar
    
    for t in range(1, T):
        try:
            Q_stress_t = Qs_precomputed[t]
            if gamma > 0 and H[t-1] == 1:
                Q_t[t] = (1 - a - b - gamma) * Q_bar + \
                         a * np.outer(Z[t-1], Z[t-1]) + \
                         b * Q_t[t-1] + \
                         gamma * Q_stress_t
            else:
                Q_t[t] = (1 - a - b) * Q_bar + \
                         a * np.outer(Z[t-1], Z[t-1]) + \
                         b * Q_t[t-1]
            
            Q_t[t] = ensure_positive_definite(Q_t[t], min_eig=1e-8)
            
            diag_q = np.sqrt(np.diag(Q_t[t]))
            diag_q = np.clip(diag_q, 1e-8, None)
            D_inv = np.diag(1 / diag_q)
            R_t[t] = D_inv @ Q_t[t] @ D_inv
            
            min_eig = np.min(np.linalg.eigvalsh(R_t[t]))
            if min_eig < 1e-5:
                R_t[t] = R_t[t] + (1e-5 - min_eig) * np.eye(N)
                
        except Exception:
            Q_t[t] = Q_t[t-1] if t > 0 else Q_bar
            R_t[t] = R_t[t-1] if t > 0 else Q_bar
    
    # Asignar también el R_t inicial
    diag_q0 = np.sqrt(np.diag(Q_t[0]))
    diag_q0 = np.clip(diag_q0, 1e-8, None)
    D_inv0 = np.diag(1 / diag_q0)
    R_t[0] = D_inv0 @ Q_t[0] @ D_inv0
    
    return R_t, Q_t, {'a': a, 'b': b, 'gamma': gamma, 'log_lik': log_lik,
                       'Qs_precomputed': Qs_precomputed}

# ============================================================================
# 🧪 TEST DE RAZÓN DE VEROSIMILITUD
# ============================================================================

def likelihood_ratio_test(z_std, H_indicator, Q_bar, Qs_precomputed=None, compute_se=True):
    """
    Test de Razón de Verosimilitud: DCC-H vs DCC estándar.

    CORRECCIÓN (punto 3 — Sección 3.6.1 de la tesis): dado que gamma>=0 es
    una restricción de FRONTERA (el modelo exige gamma>=0 por construcción
    económica), bajo H0: gamma=0 el estadístico LR NO sigue una chi2(1)
    estándar. Sigue una mezcla 50/50 entre una masa puntual en 0 y una
    chi2(1) (Self & Liang, 1987; Chernoff, 1954), de modo que:

        p_value_corregido = 0.5 * P(chi2(1) > LR_observado)

    Usar la chi2(1) sin corregir SUBESTIMA sistemáticamente el p-value real,
    inflando la tasa de falsos positivos ("homeostasis detectable" con
    mayor frecuencia de la que el modelo realmente sustenta). Se reporta
    tanto el p-value naive (chi2(1) estándar, solo a modo de referencia)
    como el p-value corregido, que es el que debe usarse para la decisión.

    Además (punto 7), se calculan errores estándar robustos tipo sandwich
    (OPG) para (a, b, gamma) del modelo no restringido — antes esta app no
    reportaba ningún error estándar.
    """
    try:
        if Qs_precomputed is None:
            Qs_precomputed = compute_recursive_Qstress(z_std, H_indicator, Q_bar)

        # Modelo restringido (DCC estándar, γ = 0)
        result_restricted = estimate_dcc_parameters(z_std, H_indicator, Q_bar, Qs_precomputed, 'DCC')
        log_lik_restricted = -result_restricted.fun
        
        # Modelo no restringido (DCC-H con γ libre)
        result_unrestricted = estimate_dcc_parameters(z_std, H_indicator, Q_bar, Qs_precomputed, 'DCC-H')
        log_lik_unrestricted = -result_unrestricted.fun
        
        # Estadístico LR
        lr_stat = 2 * (log_lik_unrestricted - log_lik_restricted)
        lr_stat = float(np.clip(lr_stat, 0, 1e6))
        
        # Grados de libertad
        df = len(result_unrestricted.x) - len(result_restricted.x)
        
        # P-value naive (chi2 estándar) y corregido (mezcla de frontera)
        p_value_naive = 1 - chi2.cdf(lr_stat, df) if lr_stat > 0 else 1.0
        p_value = 0.5 * p_value_naive  # <- p-value a usar para la decisión

        # Valores críticos: naive (3.84 para df=1) y corregido (2.71 para df=1)
        critical_value_naive = chi2.ppf(0.95, df)
        critical_value = chi2.ppf(0.90, df)  # equivalente, tras la corrección, a LR>2.71
        
        # Decisión (basada en el p-value CORREGIDO)
        decision = "RECHAZAR_H0" if p_value < 0.05 else "NO_RECHAZAR_H0"

        se_unrestricted = None
        if compute_se:
            try:
                se_unrestricted = compute_opg_se(z_std, H_indicator, Q_bar, Qs_precomputed,
                                                   result_unrestricted.x)
            except Exception:
                se_unrestricted = None
        
        return {
            'lr_statistic': lr_stat,
            'df': df,
            'p_value': p_value,
            'p_value_naive': p_value_naive,
            'critical_value': critical_value,
            'critical_value_naive': critical_value_naive,
            'decision': decision,
            'log_lik_restricted': log_lik_restricted,
            'log_lik_unrestricted': log_lik_unrestricted,
            'params_restricted': result_restricted.x,
            'params_unrestricted': result_unrestricted.x,
            'se_unrestricted': se_unrestricted,
            'Qs_precomputed': Qs_precomputed,
        }
    
    except Exception as e:
        return {
            'lr_statistic': 0.0,
            'df': 1,
            'p_value': 1.0,
            'p_value_naive': 1.0,
            'critical_value': 2.7055,
            'critical_value_naive': 3.8415,
            'decision': 'NO_RECHAZAR_H0',
            'log_lik_restricted': -1000.0,
            'log_lik_unrestricted': -1000.0,
            'params_restricted': [0.0, 0.9],
            'params_unrestricted': [0.0, 0.9, 0.0],
            'se_unrestricted': None,
            'Qs_precomputed': None,
        }


def benjamini_hochberg(pvalues, alpha=0.05):
    """
    Corrección por comparaciones múltiples de Benjamini-Hochberg (FDR).
    Punto 8 de la adenda doctoral (Sección 4.6): al explorar una grilla de
    especificaciones (distintos alpha de Gumbel, distintos kappa, distintas
    ventanas), reportar el mejor resultado sin corrección expone a la tesis
    a una crítica clásica de data snooping (White, 2000). Se prefiere BH
    sobre Bonferroni por ser menos conservadora cuando los períodos/
    especificaciones no son independientes entre sí.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    if m == 0:
        return {'adjusted_pvalues': np.array([]), 'significant': np.array([]), 'cutoff': 0.0}

    order = np.argsort(pvalues)
    ranked = pvalues[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= thresh

    if passed.any():
        k_max = np.max(np.where(passed)[0])
        cutoff = ranked[k_max]
    else:
        cutoff = 0.0

    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    adj_pvalues = np.empty(m)
    adj_pvalues[order] = adj

    significant = pvalues <= cutoff if cutoff > 0 else np.zeros(m, dtype=bool)
    return {'adjusted_pvalues': adj_pvalues, 'significant': significant, 'cutoff': float(cutoff)}


def run_robustness_panel(returns, alpha_grid, kappa_grid, garch_window=252):
    """
    Panel de robustez (punto 8): re-ejecuta GARCH -> Gumbel -> H_t -> DCC-H
    -> Test LR (corregido) para cada combinación (alpha, kappa) de la
    grilla, y aplica la corrección de Benjamini-Hochberg sobre el conjunto
    completo de p-values obtenidos, dejando explícito el número total de
    especificaciones evaluadas.
    """
    z_std, sigma, garch_params_df, garch_state = garch_filter(returns)
    Q_bar_full = np.corrcoef(z_std.T)
    Q_bar_full = ensure_positive_definite(Q_bar_full, min_eig=1e-6)

    rows = []
    for a_g in alpha_grid:
        for k_g in kappa_grid:
            _, indicators, _ = fit_gumbel_threshold(z_std, a_g, garch_window)
            H_t_g, prop_g = calculate_systemic_indicator(indicators, k_g)
            lr_res = likelihood_ratio_test(z_std, H_t_g, Q_bar_full, compute_se=False)
            rows.append({
                'alpha_gumbel': a_g,
                'kappa': k_g,
                'dias_Ht': int(H_t_g.sum()),
                'pct_Ht': float(H_t_g.mean() * 100),
                'LR_stat': lr_res['lr_statistic'],
                'p_value_corregido': lr_res['p_value'],
                'gamma': float(lr_res['params_unrestricted'][2]) if len(lr_res['params_unrestricted']) > 2 else np.nan,
            })

    df = pd.DataFrame(rows)
    bh = benjamini_hochberg(df['p_value_corregido'].values, alpha=0.05)
    df['p_value_ajustado_BH'] = bh['adjusted_pvalues']
    df['significativo_BH'] = bh['significant']
    df['n_especificaciones_evaluadas'] = len(df)
    return df

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

def out_of_sample_validation(prices, valid_tickers, train_ratio=0.7, confidence_gumbel=0.95, 
                             kappa_threshold=0.3, var_confidence=0.95, garch_window=252):
    """
    Validación Out-of-Sample pura, evitando el Look-Ahead Bias.

    CORRECCIÓN respecto de la versión anterior: se detectaron y corrigieron
    DOS fuentes de fuga de información desde el período de prueba hacia el
    de entrenamiento:

    1) GARCH re-estimado de forma independiente sobre el propio test set
       (incluyendo su inicialización con la varianza incondicional DEL
       TEST). Ahora el GARCH se estima SOLO en entrenamiento, y se aplica
       congelado sobre el test, continuando la recursión de sigma² desde el
       último estado observado en entrenamiento (garch_filter_apply_fixed).

    2) Q^(S) se recalculaba dentro de dcc_homeostatic() usando z_std_test y
       H_t_test — es decir, usando información del propio período de
       prueba (incluyendo días *posteriores* a cada t pronosticado). Ahora
       Q^(S)_t se calcula de forma recursiva sobre la serie COMBINADA
       (entrenamiento + prueba, en ese orden temporal), de modo que al
       proyectar sobre el test set solo se usa información estrictamente
       pasada respecto de cada t (que puede incluir historia de
       entrenamiento, pero nunca del futuro del propio test).
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
    z_std_train, sigma_train, garch_params_train, garch_state = garch_filter(returns_train)
    _, indicators_train, _ = fit_gumbel_threshold(z_std_train, confidence_gumbel, garch_window)
    H_t_train, prop_stressed_train = calculate_systemic_indicator(indicators_train, kappa_threshold)
    
    Q_bar_train = np.corrcoef(z_std_train.T)
    Q_bar_train = ensure_positive_definite(Q_bar_train, min_eig=1e-6)

    # Qs recursivo calculado SOLO con el tramo de entrenamiento, para estimar
    # (a, b, gamma) sin ninguna información del test.
    Qs_train = compute_recursive_Qstress(z_std_train, H_t_train, Q_bar_train)
    R_t_train, Q_t_train, p_train = dcc_homeostatic(z_std_train, H_t_train, Q_bar_train,
                                                      Qs_precomputed=Qs_train)
    params_train = [p_train['a'], p_train['b'], p_train['gamma']]
    
    # DCC Estándar para comparación (Train) — gamma=0, Qs irrelevante pero se pasa por consistencia
    H_zero_train = pd.Series(0, index=H_t_train.index)
    Qs_zero_train = compute_recursive_Qstress(z_std_train, H_zero_train, Q_bar_train)
    _, _, p_std = dcc_homeostatic(z_std_train, H_zero_train, Q_bar_train, Qs_precomputed=Qs_zero_train)
    params_std = [p_std['a'], p_std['b'], 0.0]
    
    # ========== FASE DE PRUEBA (OUT-OF-SAMPLE) ==========
    # GARCH: parámetros CONGELADOS desde entrenamiento, continuando la
    # recursión de sigma² (sin reestimar, sin reinicializar con datos del test)
    z_std_test, sigma_test = garch_filter_apply_fixed(returns_test, garch_state)

    # H_t del test: se calcula causalmente sobre la serie COMBINADA
    # train+test para que los primeros días de test tengan historia
    # suficiente (ventana de Gumbel), sin usar ningún dato futuro del test.
    z_std_full = pd.concat([z_std_train, z_std_test], axis=0)
    _, indicators_full, _ = fit_gumbel_threshold(z_std_full, confidence_gumbel, garch_window)
    H_t_full, prop_stressed_full = calculate_systemic_indicator(indicators_full, kappa_threshold)
    H_t_test = H_t_full.iloc[n_train:]
    prop_stressed_test = prop_stressed_full.iloc[n_train:]

    # Q^(S) recursivo sobre la serie COMBINADA (causal): al llegar al test,
    # "sabe" la historia de estrés de entrenamiento, pero nada del futuro
    # dentro del propio test.
    Qs_full = compute_recursive_Qstress(z_std_full, H_t_full, Q_bar_train)

    # Proyección con parámetros congelados de Train, sobre la serie completa,
    # para que Q_{t-1} tenga continuidad real entre train y test; solo se
    # reporta el tramo de test.
    R_t_full, Q_t_full, _ = dcc_homeostatic(z_std_full, H_t_full, Q_bar_train,
                                              fixed_params=params_train, Qs_precomputed=Qs_full)
    R_t_test = R_t_full[n_train:]
    var_test = calculate_var(returns_test, R_t_test, sigma_test, confidence=var_confidence)
    backtest_oos = backtest_var(returns_test, var_test, var_confidence)

    # Benchmark DCC estándar, misma lógica, gamma=0
    H_zero_full = pd.Series(0, index=H_t_full.index)
    Qs_zero_full = compute_recursive_Qstress(z_std_full, H_zero_full, Q_bar_train)
    R_t_standard_full, _, _ = dcc_homeostatic(z_std_full, H_zero_full, Q_bar_train,
                                                fixed_params=params_std, Qs_precomputed=Qs_zero_full)
    R_t_standard = R_t_standard_full[n_train:]
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

def plot_tension_financiera(H_t, dates):
    """
    Calcula y grafica los días acumulados sin un evento sistémico (Latencia).
    Es el equivalente financiero a la brecha entre números primos.
    """
    # Calcular días desde el último H_t = 1
    # Si H_t es 1, se resetea a 0. Si es 0, suma 1 al día anterior.
    tension_acumulada = H_t.groupby((H_t == 1).cumsum()).cumcount()
    
    fig = go.Figure()
    
    # Área de acumulación de tensión
    fig.add_trace(go.Scatter(
        x=dates, 
        y=tension_acumulada,
        mode='lines', 
        name='Tensión Acumulada (Días)',
        fill='tozeroy',
        line=dict(color='#ff4b4b', width=2)
    ))
    
    # Marcar los días de reseteo (H_t = 1)
    reseteos = dates[H_t == 1]
    fig.add_trace(go.Scatter(
        x=reseteos, 
        y=[0]*len(reseteos),
        mode='markers', 
        name='Reseteo Homeostático (H_t=1)',
        marker=dict(color='#00d4ff', size=8, line=dict(width=1, color='white'))
    ))
    
    fig.update_layout(
        title="📈 Curva de Presión del Mercado (Equivalente a Brechas de Primos)",
        xaxis_title="Fecha",
        yaxis_title="Días sin Crisis Sistémica (Latencia)",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)"
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
            
            # Mostrar datos
            with st.expander("📋 Ver Datos de Precios"):
                st.dataframe(prices.tail(10))
            
            # 2. Cálculo de retornos y GARCH
            st.markdown("---")
            st.markdown('<p class="sub-header">📈 2. Cálculo de Retornos y Filtrado GARCH</p>', 
                       unsafe_allow_html=True)
            
            z_std, sigma, garch_params_df, garch_state = garch_filter(returns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Retorno Medio Anual", f"{returns.mean().mean()*252:.2%}")
            with col2:
                st.metric("Volatilidad Anual", f"{returns.std().std()*np.sqrt(252):.2%}")

            with st.expander("📐 Parámetros GARCH(1,1) estimados por activo (MLE) y diagnósticos", expanded=False):
                st.caption(
                    "Los parámetros (ω, α, β) ya NO están fijos: se estiman individualmente por "
                    "Máxima Verosimilitud para cada activo (Sección 3.2.3). Ljung-Box y ARCH-LM "
                    "verifican que no quede autocorrelación ni heterocedasticidad residual en z_t."
                )
                st.dataframe(garch_params_df.style.format({
                    'omega': '{:.2e}', 'alpha': '{:.4f}', 'beta': '{:.4f}',
                    'persistencia (a+b)': '{:.4f}',
                    'Ljung-Box p (z)': '{:.4f}', 'Ljung-Box p (z²)': '{:.4f}',
                    'ARCH-LM p': '{:.4f}',
                }), use_container_width=True)
                n_no_converge = int((~garch_params_df['Convergió']).sum())
                if n_no_converge > 0:
                    st.warning(f"⚠️ {n_no_converge} activo(s) no convergieron en la optimización GARCH. "
                               f"Revisar la calidad de esos ajustes antes de interpretar resultados.")
                n_lb_reject = int((garch_params_df['Ljung-Box p (z²)'] < 0.05).sum())
                if n_lb_reject > 0:
                    st.info(f"ℹ️ {n_lb_reject} activo(s) muestran autocorrelación residual en z² "
                            f"(Ljung-Box p<0.05): el GARCH(1,1) podría no estar capturando toda la "
                            f"heterocedasticidad condicional en esos casos.")
            
            # 3. Modelo de Valores Extremos (Gumbel)
            st.markdown("---")
            st.markdown('<p class="sub-header">🎯 3. Distribución de Gumbel y Umbrales</p>', 
                       unsafe_allow_html=True)
            
            thresholds, indicators, threshold_ts = fit_gumbel_threshold(z_std, confidence_gumbel, garch_window)
            H_t, prop_stressed = calculate_systemic_indicator(indicators, kappa_threshold)
            
            # Mostrar umbrales
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 Umbrales por Activo (Gumbel) — promedio temporal**")
                st.caption("El umbral τ_t es causal y varía en el tiempo (ventana móvil estrictamente "
                           "pasada); se muestra aquí su promedio solo a fines de referencia.")
                threshold_df = pd.DataFrame({
                    'Ticker': list(thresholds.keys()),
                    'Umbral (promedio)': list(thresholds.values())
                })
                st.dataframe(threshold_df.style.format({'Umbral (promedio)': '{:.4f}'}))
            
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
                plot_correlation_heatmap(R_t, returns.index, valid_tickers, 
                                        "Matriz de Correlación Promedio (Últimos 60 días)"),
                use_container_width=True
            )
            
            # Selector de par para serie temporal de correlación
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
                
                # Mostrar resultados en métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Estadístico LR", f"{lr_results['lr_statistic']:.4f}")
                
                with col2:
                    st.metric("Valor Crítico corregido (5%)", f"{lr_results['critical_value']:.4f}",
                               help="γ≥0 es una restricción de frontera: bajo H0 el LR sigue una "
                                    "mezcla 50/50 chi2(0)+chi2(1) (Self & Liang, 1987), no una chi2(1) "
                                    "estándar. El valor crítico corregido es ≈2.71 en vez de 3.84.")
                
                with col3:
                    st.metric("P-value corregido", f"{lr_results['p_value']:.6f}",
                               help=f"P-value naive (chi2(1) sin corregir, NO usar para decisión): "
                                    f"{lr_results['p_value_naive']:.6f}")
                
                with col4:
                    if lr_results['decision'] == "RECHAZAR_H0":
                        st.success("✅ H0 Rechazada")
                    else:
                        st.error("❌ H0 No Rechazada")

                st.caption(
                    f"⚠️ Corrección de frontera aplicada (Sección 3.6.1): p-value corregido = "
                    f"0.5 × p-value naive = 0.5 × {lr_results['p_value_naive']:.6f} = "
                    f"{lr_results['p_value']:.6f}. La decisión se basa en el p-value **corregido**."
                )

                if lr_results.get('se_unrestricted') is not None:
                    se = lr_results['se_unrestricted']
                    a_hat, b_hat, g_hat = lr_results['params_unrestricted'][:3]
                    se_df = pd.DataFrame({
                        'Parámetro': ['a (shock)', 'b (persistencia)', 'γ (homeostasis)'],
                        'Estimación': [a_hat, b_hat, g_hat],
                        'SE robusto (sandwich/OPG)': se[:3] if len(se) >= 3 else [np.nan]*3,
                    })
                    se_df['t-stat'] = se_df['Estimación'] / se_df['SE robusto (sandwich/OPG)'].replace(0, np.nan)
                    st.markdown("**Errores estándar robustos (Etapa 2, aproximación OPG/sandwich):**")
                    st.dataframe(se_df.style.format({
                        'Estimación': '{:.4f}', 'SE robusto (sandwich/OPG)': '{:.4f}', 't-stat': '{:.2f}'
                    }), use_container_width=True)
                    st.caption(
                        "Nota: esta aproximación robustece los errores estándar de la Etapa 2 (DCC), "
                        "pero no propaga la incertidumbre de la Etapa 1 (parámetros GARCH). Para "
                        "inferencia doctoral completa, complementar con bootstrap paramétrico "
                        "(Sección 3.9 de la tesis)."
                    )
                
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

            # ========================================================================
            # 🧪 PANEL DE ROBUSTEZ Y CORRECCIÓN POR COMPARACIONES MÚLTIPLES (punto 8)
            # ========================================================================
            st.markdown("---")
            st.markdown('<p class="sub-header">🧪 5b. Panel de Robustez (Corrección por Comparaciones Múltiples)</p>',
                       unsafe_allow_html=True)
            st.caption(
                "Explorar varias combinaciones de α (Gumbel) y κ (umbral sistémico) y reportar solo "
                "la mejor expone a la tesis a data snooping (White, 2000): con suficientes "
                "combinaciones, es esperable encontrar alguna 'significativa' por azar. Este panel "
                "corre la grilla completa y aplica corrección de Benjamini-Hochberg (FDR) sobre el "
                "conjunto de p-values (Sección 4.6)."
            )

            colr1, colr2 = st.columns(2)
            with colr1:
                alpha_options = sorted(set([0.90, 0.95, 0.97, 0.98, 0.99, round(float(confidence_gumbel), 3)]))
                alpha_default = sorted(set([round(float(confidence_gumbel), 3), 0.95, 0.99]) & set(alpha_options))
                alpha_grid_sel = st.multiselect(
                    "Valores de α (Gumbel) a evaluar",
                    alpha_options,
                    default=alpha_default
                )
            with colr2:
                kappa_options = sorted(set([0.30, 0.45, 0.60, round(float(kappa_threshold), 3)]))
                kappa_default = sorted(set([round(float(kappa_threshold), 3), 0.30, 0.45]) & set(kappa_options))
                kappa_grid_sel = st.multiselect(
                    "Valores de κ (umbral sistémico) a evaluar",
                    kappa_options,
                    default=kappa_default
                )

            run_robustness = st.button("▶️ Ejecutar panel de robustez", key="run_robustness_btn")
            if run_robustness:
                if not alpha_grid_sel or not kappa_grid_sel:
                    st.warning("Seleccioná al menos un valor de α y uno de κ.")
                else:
                    n_specs = len(alpha_grid_sel) * len(kappa_grid_sel)
                    with st.spinner(f"Evaluando {n_specs} especificaciones (α × κ)..."):
                        robustness_df = run_robustness_panel(returns, alpha_grid_sel, kappa_grid_sel, garch_window)
                    st.info(f"ℹ️ Se evaluaron **{n_specs} especificaciones** en total "
                            f"(declarado explícitamente para evitar data snooping).")
                    st.dataframe(robustness_df.style.format({
                        'pct_Ht': '{:.2f}', 'LR_stat': '{:.4f}',
                        'p_value_corregido': '{:.6f}', 'gamma': '{:.4f}',
                        'p_value_ajustado_BH': '{:.6f}',
                    }), use_container_width=True)
                    n_sig_raw = int((robustness_df['p_value_corregido'] < 0.05).sum())
                    n_sig_bh = int(robustness_df['significativo_BH'].sum())
                    st.markdown(
                        f"**Resultados significativos sin corrección (p<0.05):** {n_sig_raw} de {n_specs}  \n"
                        f"**Resultados significativos tras corrección BH-FDR:** {n_sig_bh} de {n_specs}"
                    )
                    if n_sig_raw > n_sig_bh:
                        st.warning(
                            "⚠️ La corrección por comparaciones múltiples reduce el número de "
                            "especificaciones que se consideran significativas. Reportar en la tesis "
                            "los resultados **ajustados**, no los crudos."
                        )
            
            # ========================================================================
            # 🎯 NUEVA SECCIÓN: CLASIFICACIÓN AUTOMÁTICA DE FASE DEL MERCADO
            # ========================================================================
            st.markdown("---")
            st.markdown('<p class="sub-header">🎯 6. Clasificación Automática de Fase del Mercado</p>', 
                       unsafe_allow_html=True)
            
            # Calcular fase detectada
            dias_ht_pct = H_t.mean() * 100
            volatilidad_anual = returns.std().std() * np.sqrt(252)
            
            # Obtener resultados de Kupiec
            var_series = calculate_var(returns, R_t, sigma, confidence=var_confidence)
            backtest_results = backtest_var(returns, var_series, var_confidence)
            
            fase_info = clasificar_fase(
                lr_pvalue=lr_results['p_value'],
                kupiec_pvalue=backtest_results['kupiec_pvalue'],
                dias_ht_percentage=dias_ht_pct,
                volatilidad_anual=volatilidad_anual
            )
            
            # Mostrar fase detectada
            mostrar_fase_detectada(fase_info)
            
            # 6. Value-at-Risk y Backtesting (ahora es sección 7)
            st.markdown("---")
            st.markdown('<p class="sub-header">⚠️ 7. Value-at-Risk Condicional Riguroso y Backtesting</p>', 
                       unsafe_allow_html=True)
            
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
            
            # 7. Validación Out-of-Sample (ahora es sección 8)
            if enable_oos:
                st.markdown("---")
                st.markdown('<p class="sub-header">🔬 8. Validación Out-of-Sample pura</p>', 
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
                        
                        # Comparación de backtesting
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
                        
                        # Gráfico OoS
                        st.plotly_chart(
                            plot_out_of_sample_comparison(oos_results),
                            use_container_width=True
                        )
                        
                        # Conclusión OoS con Variables Dinámicas
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
            
            # ========================================================================
            # 📊 NUEVA SECCIÓN: TABLA COMPARATIVA DE TODOS LOS PERÍODOS ANALIZADOS
            # ========================================================================
            st.markdown("---")
            st.markdown('<p class="sub-header">📊 9. Tabla Comparativa de Todos los Períodos Analizados</p>', 
                       unsafe_allow_html=True)
            
            with st.expander("Ver tabla comparativa de fases detectadas en todos los períodos", expanded=True):
                # Crear tabla comparativa con datos históricos
                comparativa_df = pd.DataFrame({
                    'Período': ['2018-2019', '2020 (COVID agudo)', '2020-2021 (COVID ext.)', 
                               '2008 (Crisis)', '2022-2026 (Transición)', 'Actual'],
                    'Tipo de Régimen': ['Normal', 'Exógeno', 'Exógeno+Recuperación', 
                                       'Endógeno', 'Transicional', regime_option],
                    'α Gumbel': [0.98, 0.99, 0.99, 0.95, 0.95, confidence_gumbel],
                    'κ Sistémico': [0.45, 0.60, 0.60, 0.30, 0.30, kappa_threshold],
                    'Días H_t': ['-', '2', '5-10', '0-29', '115', int(H_t.sum())],
                    '% Tiempo': ['<2%', '0.55%', '0.7%', '0-11.5%', '7.7%', f'{dias_ht_pct:.1f}%'],
                    'LR p-value': ['<0.0001', '0.0', '1.0', '1.0', '1.0', f"{lr_results['p_value']:.6f}"],
                    'γ Significativo': ['✅ SÍ', '✅ SÍ', '❌ NO', '❌ NO', '❌ NO', 
                                       '✅ SÍ' if lr_results['p_value'] < 0.05 else '❌ NO'],
                    'Kupiec p-value': ['-', '0.25', '0.79', '0.0008-0.0078', '0.816', 
                                      f"{backtest_results['kupiec_pvalue']:.4f}"],
                    'VaR Confiable': ['✅', '✅', '✅', '❌', '✅', 
                                     '✅' if backtest_results['passed'] else '❌'],
                    'Fase Detectada': ['FASE 1: Estabilidad', 'FASE 2: Shock Exógeno', 
                                      'FASE 4: Transición', 'FASE 3: Saturación', 
                                      'FASE 4: Transición', fase_info['fase']]
                })
                
                st.dataframe(comparativa_df, use_container_width=True)
                
                st.info("""
                **Leyenda:**
                - **FASE 1**: Estabilidad (γ significativo, VaR confiable, H_t bajo)
                - **FASE 2**: Shock Exógeno (γ significativo, VaR confiable, H_t moderado)
                - **FASE 3**: Saturación (γ NO significativo, VaR NO confiable, H_t alto)
                - **FASE 4**: Transición (γ NO significativo, VaR confiable, H_t moderado-alto)
                """)
            
            # 8. Exportar resultados (ahora es sección 10)
            st.markdown("---")
            st.markdown('<p class="sub-header">💾 10. Exportar Resultados</p>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Exportar series temporales
                results_df = pd.DataFrame({
                    'Date': returns.index,
                    'H_Indicator': H_t.values,
                    'Prop_Stressed': prop_stressed.values,
                    'VaR': var_series,
                    'Fase_Detectada': fase_info['fase']
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
                    'Activos': len(valid_tickers),
                    'Período': f"{start_date} a {end_date}",
                    'Confianza Gumbel': confidence_gumbel,
                    'Umbral κ': kappa_threshold,
                    'Días Homeostasis': int(H_t.sum()),
                    'Porcentaje H_t': f'{dias_ht_pct:.1f}%',
                    'Fase Detectada': fase_info['fase'],
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
        8. **🎯 Clasificador Automático**: Detecta automáticamente la fase del mercado (Estabilidad/Shock/Saturación/Transición)
        
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
