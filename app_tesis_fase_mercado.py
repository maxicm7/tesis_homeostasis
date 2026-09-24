# ============================================================================
# 🎓 TESIS DOCTORAL: Modelo DCC-GARCH Homeostático con EVT (Gumbel)
# ============================================================================
# Archivo: app_tesis.py
# Versión: 2.6 — bootstrap de robustez con corrección de Romano-Wolf (septiembre 2026)
# Ejecutar: streamlit run app_tesis.py
#
# CAMBIOS RESPECTO DE LA VERSIÓN ANTERIOR (el detalle está en cada función):
#  [1] Verosimilitud DCC: con R_t (que usa info hasta t-1) se evalúa z_t, no
#      z_{t-1}; y la recursión usa Q_{t-1} SIN normalizar. Estimación y
#      filtrado comparten ahora una única función de recursión.
#  [2] Penalización de región inválida = -1e10 (antes -1000, que podía ser
#      "mejor" que la verosimilitud real y atraer al optimizador).
#  [3] Historia previa (burn-in) separada de la ventana de análisis, y
#      diagnóstico explícito de identificación de γ ("días informativos").
#  [4] Kupiec corregido para 0 violaciones; se agrega Christoffersen.
#  [5] Streamlit: resultados persistentes (session_state + caché). El
#      selector de pares y el panel de robustez ahora funcionan.
#  [6] EVT: Gumbel ajustada sobre MÁXIMOS POR BLOQUE (L-momentos).
#  [7] Out-of-Sample: entrenamiento = historia previa, prueba = ventana;
#      evaluación por cobertura y pérdida cuantílica + Diebold-Mariano.
#  [8] Tabla comparativa construida con corridas reales de la sesión
#      (se eliminan los valores fijos de versiones anteriores).
#  [9] Una única especificación principal (α, κ) para todos los períodos.
# [10] Errores estándar: sandwich real (A⁻¹BA⁻¹) y OPG, con aviso de frontera.
# [11] Clasificador de fases sin huecos y con chequeo de identificación.
# [12] Detalles: volatilidad, fallback GARCH, fines de semana (BTC), errores
#      que antes se ocultaban, etiquetas.
# [13] v2.1: recursión DCC-H vectorizada (idéntica, ~30x más rápida).
# [14] v2.2: la tensión puede medirse como SORPRESA (residuos z del GARCH,
#      umbral con ventana móvil — especificación original) o como ESTADO
#      (retornos brutos, umbral con ventana expansiva). Con la especificación
#      original, cada activo supera su umbral ≈1% de los días en CUALQUIER
#      régimen (doble normalización), por lo que H_t no distingue crisis.
# [15] v2.2: el clasificador ya no etiqueta "Estabilidad" cuando γ no es
#      significativo por falta de información: informa "sin evidencia
#      concluyente". Aviso si la ventana es demasiado larga para una fase.
# [16] v2.2: tablas sin "None", nombre para períodos personalizados, preset
#      P6 (2022–hoy) y portafolio base con historia larga (^GDAXI, DX-Y.NYB).
# [17] v2.3: el selector de fechas "Personalizado" permitía ir solo 10 años
#      hacia atrás (límite por defecto de st.date_input); ahora desde 1990.
# [18] v2.3: con γ no significativo, la Fase 4 exige al menos
#      MIN_INFORMATIVOS_POTENCIA días informativos; con menos, el resultado
#      es "sin evidencia concluyente" (el Monte Carlo mostró potencia ≈0 con
#      12–20 días informativos). La Fase 3 se mantiene (VaR fallido y tensión
#      alta son observables), con una advertencia sobre γ.
# [19] v2.4: GARCH vectorizado (idéntico, ~14x más rápido).
# [20] v2.4: bootstrap paramétrico del test LR bajo H0: γ=0 (Sección 3.6.1),
#      que reestima ambas etapas en cada réplica.
# [21] v2.5: el bootstrap guarda cada réplica al terminarla; si la ejecución
#      se interrumpe (un clic en la página, la pestaña suspendida), al volver
#      a presionar el botón continúa desde donde quedó.
# [22] v2.6: el panel de robustez descarta especificaciones equivalentes (con
#      N activos, dos κ que exigen la misma cantidad de activos en tensión dan
#      exactamente el mismo H_t y no deben contarse dos veces en la corrección).
# [23] v2.6: bootstrap de robustez con corrección de Romano-Wolf (stepdown
#      sobre el máximo LR), que respeta la correlación entre especificaciones.
# ============================================================================

import time
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from scipy.stats import norm, chi2
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.signal import lfilter
from scipy.special import xlogy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DCC-GARCH Homeostático - Tesis Doctoral",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# ⚙️ CONSTANTES
# ============================================================================
EULER_GAMMA = 0.5772156649015329
PENALTY = -1e10               # [2] log-verosimilitud de región inválida
ALPHA_TEST = 0.05             # nivel de significancia de los tests
UMBRAL_H_BAJO = 2.0           # % de días con H_t=1 (clasificador de fases)
UMBRAL_H_ALTO = 10.0
# [18] PROVISIONAL: días informativos mínimos para interpretar un γ NO
# significativo como "corrección no sistemática" (Fase 4). El Monte Carlo
# con T=1000 (~12 días informativos) dio potencia ≈0 para γ ≤ 0.07. Ajustar
# este valor cuando el Monte Carlo indique con cuántos días la potencia es
# aceptable (p. ej. ≥ 80%).
MIN_INFORMATIVOS_POTENCIA = 60
DCC_H_BOUNDS = [(1e-6, 0.3), (0.5, 0.998), (0.0, 0.3)]   # a, b, γ
DCC_STATIONARITY = 0.998      # a + b + γ <= 0.998

# ============================================================================
# 🎨 ESTILOS CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin-top: 30px;}
    .warning-box {background-color: #fff3cd; padding: 15px; border-radius: 5px;
                  border-left: 5px solid #ffc107;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 🎯 CLASIFICADOR AUTOMÁTICO DE FASES  [11]
# ============================================================================

def _fase(fase, subfase, color, descripcion, caracteristicas, recomendaciones, nota=None):
    return {'fase': fase, 'subfase': subfase, 'color': color, 'descripcion': descripcion,
            'caracteristicas': caracteristicas, 'recomendaciones': recomendaciones,
            'nota': nota}


def clasificar_fase(lr_pvalue, kupiec_pvalue, dias_ht_percentage, gamma_identificado=True,
                    n_informativos=None, min_informativos=MIN_INFORMATIVOS_POTENCIA,
                    umbral_bajo=UMBRAL_H_BAJO, umbral_alto=UMBRAL_H_ALTO, alpha_test=ALPHA_TEST):
    """
    Clasifica la fase del mercado. CORRECCIÓN [11]: la versión anterior tenía
    combinaciones sin fase asignada (p. ej. γ significativo con H_t > 10%) y
    exigía γ significativo en Fase 1 con H_t < 2%, una combinación casi
    imposible porque con tan pocos días de estrés γ no está identificado.

    Árbol de decisión (cubre todas las combinaciones):
      0) γ no identificado  -> INDETERMINADA (no hay información para evaluar γ)
      1) VaR NO confiable (Kupiec p <= α):
           H% > umbral_alto -> FASE 3 (Saturación)
           H% <= umbral_alto -> INDETERMINADA (VaR mal calibrado sin tensión
                                sistémica: posible falla de especificación)
      2) VaR confiable:
           γ significativo:  H% < umbral_bajo -> FASE 1 ; resto -> FASE 2
           γ no significativo:
             con < min_informativos días informativos o H% <= umbral_bajo
                 -> SIN EVIDENCIA CONCLUYENTE [15]
             resto -> FASE 4 (Transición)

    CORRECCIÓN [15]: la v2.0 etiquetaba "FASE 1: ESTABILIDAD" a períodos con γ
    no significativo y poca información (p. ej. COVID o 2008). El estudio
    Monte Carlo muestra que con pocos días informativos el test no tiene
    potencia, así que un γ no significativo no permite afirmar nada.

    NOTA PARA LA TESIS: las reglas que completan los huecos de la versión
    anterior (casos marcados con 'nota') son una propuesta a validar contra
    la definición teórica de las fases; los umbrales son configurables.
    """
    h = dias_ht_percentage

    if (not gamma_identificado) or lr_pvalue is None or not np.isfinite(lr_pvalue):
        return _fase(
            'FASE INDETERMINADA', 'γ no identificado', '#6c757d',
            'En la muestra de estimación no hubo días con H_t=1 en los que Q^(S) difiera de Q̄, '
            'por lo que el término homeostático no puede estimarse ni testearse.',
            ['? Sin días informativos para γ', '? Test LR no aplicable'],
            ['• Ampliar la historia previa (burn-in) o la ventana de análisis',
             '• Revisar si α/κ generan suficientes días de estrés (declarar el cambio)'])

    gamma_sig = lr_pvalue < alpha_test
    var_ok = kupiec_pvalue > alpha_test
    poca_info = n_informativos is not None and n_informativos < min_informativos

    if not var_ok:
        if h > umbral_alto:
            if gamma_sig:
                nota = ('γ es significativo: la corrección homeostática opera pero resulta '
                        'insuficiente frente a la magnitud del shock.')
            elif poca_info:
                nota = (f'γ no significativo con solo {n_informativos} días informativos: no es '
                        'evidencia de mecanismos comprometidos, solo de tensión alta con VaR '
                        'mal calibrado.')
            else:
                nota = None
            return _fase(
                'FASE 3: SATURACIÓN', 'Mecanismos Comprometidos', '#dc3545',
                'Tensión sistémica muy alta y VaR que no cubre el riesgo realizado.',
                [('⚠ γ significativo pero insuficiente' if gamma_sig else '✗ γ NO significativo'),
                 '✗ VaR mal calibrado (Kupiec)', '✗ Tensión sistémica muy alta',
                 '✗ Correlaciones tienden a 1 (contagio)'],
                ['• Activar protocolo de crisis', '• Complementar VaR con stress tests',
                 '• Reducir exposición y aumentar liquidez', '• Reportar a comité de riesgo'],
                nota)
        return _fase(
            'FASE INDETERMINADA', 'VaR mal calibrado sin tensión sistémica alta', '#6c757d',
            'El VaR falla el test de Kupiec aunque la tensión sistémica no es alta: '
            'probable problema de especificación del VaR (distribución, pesos, volatilidad).',
            ['✗ VaR mal calibrado', f'? H_t = {h:.1f}% (≤ {umbral_alto:.0f}%)'],
            ['• Revisar supuesto de normalidad del VaR', '• Revisar ajuste GARCH por activo'],
            'Caso no contemplado en la versión anterior del clasificador.')

    if gamma_sig:
        if h < umbral_bajo:
            return _fase(
                'FASE 1: ESTABILIDAD', 'Homeostasis Continua', '#28a745',
                'Los mecanismos de corrección operan de forma continua y el VaR está bien calibrado.',
                ['✓ γ significativo', '✓ VaR confiable', '✓ Tensión sistémica baja'],
                ['• Diversificación tradicional', '• VaR paramétrico suficiente', '• Monitoreo estándar'])
        nota = (f'H_t = {h:.1f}% supera {umbral_alto:.0f}%: tensión alta pero absorbida por el sistema.'
                if h > umbral_alto else None)
        return _fase(
            'FASE 2: SHOCK EXÓGENO', 'Homeostasis Selectiva', '#ffc107',
            'Un shock activa los mecanismos homeostáticos de forma selectiva y el sistema lo absorbe.',
            ['✓ γ significativo', '✓ VaR confiable', '⚠ Tensión sistémica moderada',
             '✓ Mecanismos de corrección operativos'],
            ['• Revisar coberturas de riesgo de cola', '• Monitorear violaciones del VaR',
             '• Activar protocolos si H_t persiste > 5 días'],
            nota)

    if poca_info or h <= umbral_bajo:
        motivo = (f'solo {n_informativos} días informativos, sin potencia para detectar γ'
                  if poca_info else f'tensión detectada baja (H_t = {h:.1f}% de los días)')
        return _fase(
            'SIN EVIDENCIA CONCLUYENTE', 'γ no significativo con información insuficiente', '#5a6b7b',
            f'No se puede afirmar ni descartar un efecto homeostático: {motivo}. Un γ no '
            'significativo en este caso no equivale a ausencia de homeostasis ni a estabilidad '
            '(ver estudio Monte Carlo, Sección 3.9).',
            ['– γ no significativo', f'– {motivo.capitalize()}', '✓ VaR confiable'],
            ['• No interpretar como "estabilidad" ni como "ausencia de homeostasis"',
             '• Ampliar la muestra o revisar la definición de tensión (Bloque 2)'],
            'Reemplaza a la etiqueta "Fase 1: Estabilidad (sin activación)" de la v2.0, que era engañosa.')
    return _fase(
        'FASE 4: TRANSICIÓN', 'Homeostasis Dispersa', '#17a2b8',
        'Hay tensión sistémica pero la corrección no opera de forma sistemática.',
        ['✗ γ NO significativo', '✓ VaR confiable', '⚠ Tensión sistémica moderada-alta',
         '⚠ Correlaciones variables'],
        ['• Mantener flexibilidad en la asignación', '• Monitorear cambios de régimen',
         '• Evitar posiciones concentradas'])


def mostrar_fase_detectada(fase_info):
    st.markdown(f"""
    <div style='background-color: {fase_info["color"]}; color: white;
                padding: 15px; border-radius: 10px; margin: 20px 0;'>
        <h2 style='margin: 0; color: white;'>🎯 {fase_info["fase"]}</h2>
        <h4 style='margin: 5px 0; color: white;'>{fase_info["subfase"]}</h4>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"**Descripción:** {fase_info['descripcion']}")
    if fase_info.get('nota'):
        st.caption(f"ℹ️ {fase_info['nota']}")
    st.markdown("**📋 Características Detectadas:**")
    for c in fase_info['caracteristicas']:
        st.markdown(f"- {c}")
    st.markdown("**✅ Recomendaciones:**")
    for r in fase_info['recomendaciones']:
        st.markdown(f"- {r}")


# ============================================================================
# 📦 DATOS
# ============================================================================

def _extract_close(data, tickers):
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(data.columns.get_level_values(0))
        if 'Close' in lvl0:
            prices = data['Close']
        elif 'Adj Close' in lvl0:
            prices = data['Adj Close']
        else:
            prices = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        col = 'Close' if 'Close' in data.columns else ('Adj Close' if 'Adj Close' in data.columns else None)
        prices = data[[col]].copy() if col else data.copy()
        if prices.shape[1] == 1:
            prices.columns = [tickers[0]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


def clean_prices(prices):
    """
    Limpieza de precios. CORRECCIÓN [12]: se eliminan sábados y domingos.
    Con BTC-USD en el portafolio, yfinance devuelve también fines de semana y
    el forward-fill generaba retornos exactamente iguales a cero para los
    índices bursátiles esos días, sesgando el GARCH y las correlaciones.
    """
    prices = prices.copy()
    if getattr(prices.index, 'tz', None) is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.dropna(axis=1, how='all')
    prices = prices[prices.index.dayofweek < 5]
    prices = prices.ffill().dropna(how='any')
    return prices


@st.cache_data(ttl=7200, show_spinner=False)
def download_data(tickers, start_date, end_date):
    """Descarga desde Yahoo Finance. Devuelve (precios, mensaje_error)."""
    tickers = list(tickers)
    try:
        data = yf.download(tickers, start=start_date, end=end_date,
                           progress=False, auto_adjust=True)
    except Exception as e:
        return None, f"Error descargando datos: {e}"
    if data is None or data.empty:
        return None, "Yahoo Finance no devolvió datos para esos activos y fechas."
    prices = clean_prices(_extract_close(data, tickers))
    if prices.empty or prices.shape[0] < 10:
        return None, "No quedaron suficientes fechas con datos para todos los activos."
    return prices, None


def calculate_returns(prices):
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    return returns


# ============================================================================
# 📈 GARCH(1,1) UNIVARIADO POR MLE, POR ACTIVO (Sección 3.2.3)
# ============================================================================

def _garch11_sigma2(omega, alpha, beta, r, sigma2_0):
    """
    σ²_t = ω + α r²_{t−1} + β σ²_{t−1}, con σ²_0 dado. [19] Es una recursión
    lineal, así que se calcula con un filtro IIR (idéntico al bucle, ~50x más
    rápido); esto hace viable reestimar el GARCH en cada réplica del bootstrap.
    """
    x = np.empty(len(r))
    x[0] = sigma2_0
    x[1:] = omega + alpha * r[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], x)


def _garch11_neg_loglik(theta, r):
    omega, alpha, beta = theta
    sigma2 = _garch11_sigma2(omega, alpha, beta, r, np.var(r) if np.var(r) > 1e-12 else 1e-6)
    sigma2 = np.clip(sigma2, 1e-12, None)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + r ** 2 / sigma2)
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit_garch11_mle(r):
    """QMLE de GARCH(1,1) con reescalado (x100) y multi-start."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    scale = 100.0
    rs = r * scale
    bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]
    cons = ({'type': 'ineq', 'fun': lambda th: 0.999 - (th[1] + th[2])},)
    var_rs = np.var(rs) if np.var(rs) > 1e-8 else 1.0
    starts = [(var_rs * 0.05, 0.05, 0.90), (var_rs * 0.10, 0.10, 0.85),
              (var_rs * 0.05, 0.02, 0.95), (var_rs * 0.20, 0.15, 0.75)]

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
        # CORRECCIÓN [12]: antes omega quedaba en escala x100² (no se des-escalaba)
        omega, alpha, beta = 0.05 * np.var(r), 0.08, 0.85
        converged, loglik = False, np.nan
    else:
        omega, alpha, beta = best_res.x
        omega = omega / (scale ** 2)
        converged = bool(best_res.success)
        loglik = -best_res.fun

    sigma2 = _garch11_sigma2(omega, alpha, beta, r, np.var(r) if np.var(r) > 1e-12 else 1e-6)
    sigma = np.sqrt(np.clip(sigma2, 1e-12, None))

    return {'omega': float(omega), 'alpha': float(alpha), 'beta': float(beta),
            'persistence': float(alpha + beta), 'sigma2': sigma2, 'sigma': sigma,
            'z': r / sigma, 'loglik': loglik, 'converged': converged,
            'sigma2_last': float(sigma2[-1]), 'resid_last': float(r[-1])}


def ljung_box_test(x, lags=10):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= lags + 1:
        return {'stat': np.nan, 'p_value': np.nan}
    x = x - np.mean(x)
    denom = np.sum(x ** 2)
    if denom < 1e-12:
        return {'stat': np.nan, 'p_value': np.nan}
    stat = sum((np.sum(x[k:] * x[:-k]) / denom) ** 2 / (n - k) for k in range(1, lags + 1))
    stat *= n * (n + 2)
    return {'stat': float(stat), 'p_value': float(chi2.sf(stat, lags))}


def arch_lm_test(x, lags=5):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x2 = x ** 2
    n = len(x2)
    if n <= lags + 5:
        return {'stat': np.nan, 'p_value': np.nan}
    Y = x2[lags:]
    X = np.column_stack([np.ones(len(Y))] + [x2[lags - k: n - k] for k in range(1, lags + 1)])
    try:
        beta_hat, *_ = np.linalg.lstsq(X, Y, rcond=None)
        ss_res = np.sum((Y - X @ beta_hat) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        stat = len(Y) * r2
        return {'stat': float(stat), 'p_value': float(chi2.sf(stat, lags))}
    except Exception:
        return {'stat': np.nan, 'p_value': np.nan}


def garch_filter(returns):
    """GARCH(1,1) por activo. Devuelve z_std, sigma (T x N), tabla de diagnósticos y estado."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    n, N = returns.shape
    sigma_matrix = np.zeros((n, N))
    z_list, rows, garch_state = [], [], {}

    for i, col in enumerate(returns.columns):
        fit = fit_garch11_mle(returns[col].values)
        sigma_matrix[:, i] = fit['sigma']
        z_list.append(fit['z'])
        lb = ljung_box_test(fit['z'], lags=10)
        lb2 = ljung_box_test(fit['z'] ** 2, lags=10)
        lm = arch_lm_test(fit['z'], lags=5)
        rows.append({'Ticker': col, 'omega': fit['omega'], 'alpha': fit['alpha'],
                     'beta': fit['beta'], 'persistencia (a+b)': fit['persistence'],
                     'Convergió': fit['converged'], 'Ljung-Box p (z)': lb['p_value'],
                     'Ljung-Box p (z²)': lb2['p_value'], 'ARCH-LM p': lm['p_value']})
        garch_state[col] = {'omega': fit['omega'], 'alpha': fit['alpha'], 'beta': fit['beta'],
                            'sigma2_last': fit['sigma2_last'], 'resid_last': fit['resid_last']}

    z_std = pd.DataFrame(np.column_stack(z_list), index=returns.index, columns=returns.columns)
    return z_std, sigma_matrix, pd.DataFrame(rows), garch_state


def garch_filter_apply_fixed(returns, garch_state):
    """Aplica parámetros GARCH congelados continuando la recursión desde el último estado."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    n, N = returns.shape
    sigma_matrix = np.zeros((n, N))
    z_list = []
    for i, col in enumerate(returns.columns):
        s = garch_state[col]
        r = returns[col].values
        sigma2 = np.empty(n)
        s2_prev, r_prev = s['sigma2_last'], s['resid_last']
        for t in range(n):
            sigma2[t] = s['omega'] + s['alpha'] * r_prev ** 2 + s['beta'] * s2_prev
            s2_prev, r_prev = sigma2[t], r[t]
        sigma = np.sqrt(np.clip(sigma2, 1e-12, None))
        sigma_matrix[:, i] = sigma
        z_list.append(r / sigma)
    z_std = pd.DataFrame(np.column_stack(z_list), index=returns.index, columns=returns.columns)
    return z_std, sigma_matrix


# ============================================================================
# 🎯 TEORÍA DE VALORES EXTREMOS: GUMBEL SOBRE MÁXIMOS POR BLOQUE  [6]
# ============================================================================

def _gumbel_lmoments(x):
    """
    Estimación de (loc, scale) de Gumbel por L-momentos (Hosking, 1990):
    scale = λ2 / ln 2,  loc = λ1 − γ_E · scale.
    Es robusta con muestras chicas (≈50 máximos) y es cerrada, lo que
    permite re-estimar en cada t de la ventana móvil sin costo.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    b0 = x.mean()
    b1 = np.sum(np.arange(n) * x) / (n * (n - 1))
    l2 = 2 * b1 - b0
    if l2 <= 0:
        return np.nan, np.nan
    scale = l2 / np.log(2.0)
    return b0 - EULER_GAMMA * scale, scale


def gumbel_rolling_params(z_std, window=252, block=5, method='bloques', min_points=10,
                          expanding=False):
    """
    Parámetros de Gumbel causales y variables en el tiempo: en cada t se usa
    solo la ventana [t-window, t-1] de |z|.

    CORRECCIÓN [6]: la distribución de Gumbel es el límite de los MÁXIMOS
    por bloque (Fisher-Tippett-Gnedenko), no de todas las observaciones.
    Con method='bloques', la ventana se divide en bloques de `block` días
    (5 = semanal) y Gumbel se ajusta a los máximos de cada bloque. El umbral
    τ_t resultante es el nivel que el máximo semanal supera solo con
    probabilidad 1-α. method='todas' reproduce el criterio anterior (Gumbel
    sobre todas las |z|) únicamente para comparación.

    Con expanding=True [14] el umbral en t usa TODO el pasado disponible
    [0, t-1] (la ventana `window` pasa a ser solo el mínimo de datos para
    empezar). El umbral no "se acostumbra" a una crisis: si los extremos se
    vuelven frecuentes, H_t lo registra. Los bloques se alinean desde el
    inicio de la muestra y solo se usan bloques completos anteriores a t.

    Devuelve dos DataFrames (T x N): loc y scale (NaN durante el burn-in).
    Separar parámetros de umbral permite evaluar cualquier α sin re-ajustar.
    """
    A = np.abs(z_std.values)
    T, N = A.shape
    loc = np.full((T, N), np.nan)
    scale = np.full((T, N), np.nan)
    for j in range(N):
        a = A[:, j]
        if expanding and method == 'bloques':
            nb = T // block
            bm = a[:nb * block].reshape(nb, block).max(axis=1)
            for t in range(window, T):
                data = bm[:t // block]          # bloques completos antes de t
                data = data[~np.isnan(data)]
                if len(data) < min_points:
                    continue
                loc[t, j], scale[t, j] = _gumbel_lmoments(data)
            continue
        for t in range(window, T):
            w = a[:t] if expanding else a[t - window:t]
            w = w[~np.isnan(w)]
            if method == 'bloques':
                m = len(w) // block
                if m < min_points:
                    continue
                data = w[len(w) - m * block:].reshape(m, block).max(axis=1)
            else:
                if len(w) < 30:
                    continue
                data = w
            loc[t, j], scale[t, j] = _gumbel_lmoments(data)
    return (pd.DataFrame(loc, index=z_std.index, columns=z_std.columns),
            pd.DataFrame(scale, index=z_std.index, columns=z_std.columns))


def gumbel_thresholds(loc_df, scale_df, confidence):
    """Cuantil α de Gumbel: τ = loc − scale · ln(−ln α)."""
    return loc_df - scale_df * np.log(-np.log(confidence))


def stress_indicators(z_std, threshold_df):
    """h_{i,t} = 1 si |z_{i,t}| > τ_{i,t}; 0 durante el burn-in (τ no definido)."""
    thr = threshold_df.values
    a = np.abs(z_std.values)
    ind = np.where(np.isnan(thr), 0, (a > np.nan_to_num(thr, nan=np.inf)).astype(int))
    return pd.DataFrame(ind, index=z_std.index, columns=z_std.columns)


def calculate_systemic_indicator(indicators, kappa=0.3):
    prop_stressed = indicators.mean(axis=1)
    return (prop_stressed >= kappa).astype(int), prop_stressed


# ============================================================================
# 🔗 DCC-GARCH HOMEOSTÁTICO
# ============================================================================

def ensure_positive_definite(matrix, min_eig=1e-6):
    sym = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sym)
    return eigvecs @ np.diag(np.maximum(eigvals, min_eig)) @ eigvecs.T


def compute_recursive_Qstress(z_std, H_indicator, Q_bar, min_obs=30, shrink_c=20.0):
    """
    Q^(S)_t recursivo (anti-circularidad, Sección 3.4.5): usa solo días con
    H_s=1 y s <= t-2, con contracción hacia Q̄ (peso c/(n+c)).

    Devuelve (Qs, active):
      - Qs: array (T, N, N)
      - active: bool (T,) — True si en t ya hay >= min_obs días de estrés y,
        por lo tanto, Q^(S)_t ≠ Q̄.

    ADVERTENCIA DE IDENTIFICACIÓN [3]: si Q^(S)_t = Q̄, entonces
    (1−a−b−γ)Q̄ + γQ̄ = (1−a−b)Q̄ y γ desaparece de la recursión. γ solo es
    identificable a partir de los días t con H_{t−1}=1 y active[t]=True
    ("días informativos"). Si no hay ninguno, el test LR no es aplicable.
    """
    Z = z_std.values if hasattr(z_std, 'values') else np.asarray(z_std)
    H = H_indicator.values if hasattr(H_indicator, 'values') else np.asarray(H_indicator)
    T, N = Z.shape
    Qs = np.empty((T, N, N))
    active = np.zeros(T, dtype=bool)
    cum_outer = np.zeros((N, N))
    cum_count = 0
    for t in range(T):
        s = t - 2   # se incorpora la información disponible hasta s = t-2
        if s >= 0 and H[s] == 1:
            cum_outer += np.outer(Z[s], Z[s])
            cum_count += 1
        if cum_count < min_obs:
            Qs[t] = Q_bar
            continue
        avg = cum_outer / cum_count
        d = np.sqrt(np.clip(np.diag(avg), 1e-8, None))
        corr = ensure_positive_definite(avg / np.outer(d, d), min_eig=1e-6)
        delta = min(1.0, shrink_c / (cum_count + shrink_c))
        Qs[t] = (1 - delta) * corr + delta * Q_bar
        active[t] = True
    return Qs, active


def count_informative_days(H, active, t_start):
    """Días t >= t_start con H_{t-1}=1 y Q^(S)_t activo: la información que identifica γ."""
    H = np.asarray(H)
    idx = np.arange(max(t_start, 1), len(H))
    if len(idx) == 0:
        return 0
    return int(np.sum((H[idx - 1] == 1) & active[idx]))


def _dcc_recursion_loop(Z, H, Q_bar, Qs, a, b, gamma, store=False):
    """
    Recursión ÚNICA del DCC-H, usada en estimación y en filtrado.  [1]

    Para t >= 1:
      Q_t = (1−a−b−γH_{t−1})Q̄ + a z_{t−1}z'_{t−1} + b Q_{t−1} + γH_{t−1} Q^(S)_t
      R_t = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}
      ℓ_t = −½ [ ln|R_t| + z_t' R_t⁻¹ z_t ]

    CORRECCIÓN [1]: la versión anterior evaluaba z_{t−1} con una R_t que ya
    contenía z_{t−1}z'_{t−1} (la observación "se predecía a sí misma", lo que
    inflaba la verosimilitud y sesgaba a hacia arriba) y además continuaba la
    recursión con R_t normalizada, mientras que el filtrado usaba Q_t. Ahora
    se evalúa z_t y se arrastra Q_t, idéntico en estimación y en filtrado.

    Q_t es definida positiva por construcción (combinación convexa de
    matrices DP más un término semidefinido), por eso no se "fuerza" la DP
    en cada paso; si Cholesky falla por redondeo se regulariza mínimamente.
    """
    T, N = Z.shape
    contrib = np.zeros(T)
    Q_store = np.empty((T, N, N)) if store else None
    R_store = np.empty((T, N, N)) if store else None
    eye = np.eye(N)
    ok = True
    Q_prev = Q_bar
    w_q = 1 - a - b
    for t in range(T):
        if t == 0:
            Q_t = Q_bar.copy()
        else:
            z1 = Z[t - 1]
            if gamma > 0 and H[t - 1] == 1:
                Q_t = (w_q - gamma) * Q_bar + a * np.outer(z1, z1) + b * Q_prev + gamma * Qs[t]
            else:
                Q_t = w_q * Q_bar + a * np.outer(z1, z1) + b * Q_prev
        d = np.sqrt(np.clip(np.diag(Q_t), 1e-12, None))
        R_t = Q_t / np.outer(d, d)
        try:
            L = np.linalg.cholesky(R_t)
        except np.linalg.LinAlgError:
            R_t = (R_t + 1e-6 * eye) / (1 + 1e-6)
            try:
                L = np.linalg.cholesky(R_t)
            except np.linalg.LinAlgError:
                L, ok = None, False
        if L is not None:
            v = solve_triangular(L, Z[t], lower=True, check_finite=False)
            contrib[t] = -0.5 * (2.0 * np.sum(np.log(np.diag(L))) + v @ v)
        else:
            contrib[t] = np.nan
        if store:
            Q_store[t], R_store[t] = Q_t, R_t
        Q_prev = Q_t
    return contrib, ok, Q_store, R_store


def _dcc_recursion_vec(Z, H, Q_bar, Qs, a, b, gamma):
    """
    Versión vectorizada de la MISMA recursión (resultados idénticos a la de
    bucle, hasta redondeo). Como Q_t = C_t + b·Q_{t−1} es lineal, se calcula
    con un filtro IIR (scipy.signal.lfilter) sobre el eje temporal, y la
    factorización de Cholesky y los sistemas lineales se resuelven en lote.
    Es ~10-30 veces más rápida; si alguna R_t no es numéricamente DP, se
    recurre a la versión de bucle, que regulariza paso a paso.
    """
    T, N = Z.shape
    C = np.empty((T, N, N))
    C[0] = Q_bar
    if T > 1:
        Z1 = Z[:-1]
        C[1:] = (1 - a - b) * Q_bar + a * (Z1[:, :, None] * Z1[:, None, :])
        if gamma > 0:
            g = gamma * (np.asarray(H[:-1]) == 1)
            C[1:] += g[:, None, None] * (Qs[1:] - Q_bar)
    Q = lfilter([1.0], [1.0, -b], C, axis=0)
    d = np.sqrt(np.clip(np.einsum('tii->ti', Q), 1e-12, None))
    R = Q / (d[:, :, None] * d[:, None, :])
    L = np.linalg.cholesky(R)                       # LinAlgError si alguna falla
    logdet = 2.0 * np.sum(np.log(np.einsum('tii->ti', L)), axis=1)
    quad = np.einsum('ti,ti->t', Z, np.linalg.solve(R, Z[:, :, None])[:, :, 0])
    return -0.5 * (logdet + quad), Q, R


def dcc_recursion(Z, H, Q_bar, Qs, a, b, gamma, store=False):
    """Recursión DCC-H (ver _dcc_recursion_loop): vía rápida vectorizada con respaldo en bucle."""
    try:
        contrib, Q, R = _dcc_recursion_vec(Z, H, Q_bar, Qs, a, b, gamma)
        if np.all(np.isfinite(contrib)):
            return contrib, True, (Q if store else None), (R if store else None)
    except np.linalg.LinAlgError:
        pass
    return _dcc_recursion_loop(Z, H, Q_bar, Qs, a, b, gamma, store)


def _unpack(params):
    a, b = float(params[0]), float(params[1])
    gamma = float(params[2]) if len(params) > 2 else 0.0
    return a, b, gamma


def dcc_loglik(params, Z, H, Q_bar, Qs, t_start=1, return_contributions=False):
    """
    Log-verosimilitud (parte de correlación) sumada desde t_start.
    CORRECCIÓN [2]: cualquier región inválida devuelve −1e10 (antes −1000,
    valor que podía ser MAYOR que la verosimilitud real — p. ej. ≈ −1500
    con 6 activos y 500 días — y atraer al optimizador hacia ella).
    También se eliminó el descarte selectivo de observaciones: todos los
    parámetros se evalúan sobre exactamente la misma muestra.
    """
    a, b, gamma = _unpack(params)
    if a < 0 or b < 0 or gamma < 0 or a + b + gamma >= 0.999:
        return (PENALTY, None) if return_contributions else PENALTY
    contrib, ok, _, _ = dcc_recursion(Z, H, Q_bar, Qs, a, b, gamma)
    c = contrib[t_start:]
    if not ok or not np.all(np.isfinite(c)):
        return (PENALTY, None) if return_contributions else PENALTY
    ll = float(np.sum(c))
    if return_contributions:
        contrib[:t_start] = 0.0
        return ll, contrib
    return ll


def estimate_dcc_parameters(Z, H, Q_bar, Qs, model_type='DCC-H', t_start=1,
                            restricted_params=None):
    """
    MLE de (a, b[, γ]) con SLSQP, restricción explícita de estacionariedad
    y multi-start. Para el DCC-H se incluye como punto de partida el óptimo
    del modelo restringido con γ=0: como el DCC-H anida al DCC, esto
    garantiza log L(DCC-H) >= log L(DCC) y, por lo tanto, LR >= 0.
    """
    def neg(p):
        ll = dcc_loglik(p, Z, H, Q_bar, Qs, t_start)
        return 1e10 if (not np.isfinite(ll) or ll <= PENALTY) else -ll

    if model_type == 'DCC-H':
        bounds = DCC_H_BOUNDS
        cons = ({'type': 'ineq', 'fun': lambda p: DCC_STATIONARITY - (p[0] + p[1] + p[2])},)
        starts = [[0.02, 0.93, 0.02], [0.05, 0.85, 0.05]]
        if restricted_params is not None:
            ar, br = float(restricted_params[0]), float(restricted_params[1])
            g0 = max(0.0, min(0.01, DCC_STATIONARITY - 0.001 - ar - br))
            starts = [[ar, br, 0.0], [ar, br, g0]] + starts
    else:
        bounds = DCC_H_BOUNDS[:2]
        cons = ({'type': 'ineq', 'fun': lambda p: DCC_STATIONARITY - (p[0] + p[1])},)
        starts = [[0.02, 0.95], [0.05, 0.90], [0.01, 0.97]]

    best = None
    for x0 in starts:
        try:
            res = minimize(neg, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-9})
        except Exception:
            continue
        if not np.isfinite(res.fun) or res.fun >= 1e9:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError(f"La estimación {model_type} no convergió desde ningún punto de partida.")
    return best


def compute_robust_se(Z, H, Q_bar, Qs, params, t_start=1, h=1e-4):
    """
    Errores estándar de la Etapa 2 por diferencias finitas.  [10]

    - OPG:      Cov = B⁻¹,          B = Σ_t s_t s_t'
    - Sandwich: Cov = A⁻¹ B A⁻¹,    A = −∂²ℓ/∂θ∂θ'  (Bollerslev-Wooldridge)

    CORRECCIÓN: la versión anterior rotulaba como "sandwich" lo que era OPG.
    Parámetros a menos de 2h de una cota (típicamente γ≈0) o con a+b+γ cerca
    del límite de estacionariedad se marcan "en frontera": allí el SE y el
    estadístico t no tienen distribución normal y se reportan como NaN; los
    SE del resto se calculan condicionales a esos parámetros.
    Limitación: no propaga la incertidumbre de la Etapa 1 (GARCH);
    complementar con bootstrap paramétrico (Sección 3.9).
    """
    params = np.asarray(params, dtype=float)
    k = len(params)
    lower = [bd[0] for bd in DCC_H_BOUNDS[:k]]
    upper = [bd[1] for bd in DCC_H_BOUNDS[:k]]
    frontera = np.array([(params[j] - 2 * h < lower[j]) or (params[j] + 2 * h > upper[j])
                         for j in range(k)])
    if params.sum() + 2 * h >= 0.999:
        frontera[:] = True
    out = {'se_sandwich': np.full(k, np.nan), 'se_opg': np.full(k, np.nan), 'frontera': frontera}
    J = [j for j in range(k) if not frontera[j]]
    if not J:
        return out

    def f(p):
        return dcc_loglik(p, Z, H, Q_bar, Qs, t_start, return_contributions=True)

    f0, _ = f(params)
    n = len(J)
    scores = np.zeros((len(Z), n))
    f_plus, f_minus = np.zeros(n), np.zeros(n)
    for m, j in enumerate(J):
        pp, pm = params.copy(), params.copy()
        pp[j] += h
        pm[j] -= h
        fp, cp = f(pp)
        fm, cm = f(pm)
        if cp is None or cm is None:
            return out
        scores[:, m] = (cp - cm) / (2 * h)
        f_plus[m], f_minus[m] = fp, fm

    Hs = np.zeros((n, n))
    for m in range(n):
        Hs[m, m] = (f_plus[m] - 2 * f0 + f_minus[m]) / h ** 2
    for m in range(n):
        for l in range(m + 1, n):
            j, i = J[m], J[l]
            vals = []
            for sj, si in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                p = params.copy()
                p[j] += sj * h
                p[i] += si * h
                vals.append(f(p)[0])
            Hs[m, l] = Hs[l, m] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h ** 2)

    B = scores.T @ scores
    A = -Hs
    try:
        cov_opg = np.linalg.inv(B)
        out['se_opg'][J] = np.sqrt(np.clip(np.diag(cov_opg), 0, None))
    except np.linalg.LinAlgError:
        pass
    try:
        if np.all(np.linalg.eigvalsh((A + A.T) / 2) > 0):
            A_inv = np.linalg.inv(A)
            cov_s = A_inv @ B @ A_inv
            out['se_sandwich'][J] = np.sqrt(np.clip(np.diag(cov_s), 0, None))
    except np.linalg.LinAlgError:
        pass
    return out


def dcc_filter(Z, H, Q_bar, Qs, params):
    """Filtra R_t y Q_t con parámetros dados (misma recursión que la estimación)."""
    a, b, gamma = _unpack(params)
    _, _, Q_t, R_t = dcc_recursion(Z, H, Q_bar, Qs, a, b, gamma, store=True)
    return R_t, Q_t


# ============================================================================
# 🧪 TEST DE RAZÓN DE VEROSIMILITUD Y COMPARACIONES MÚLTIPLES
# ============================================================================

def likelihood_ratio_test(Z, H, Q_bar, Qs, active, t_start=1, compute_se=True):
    """
    LR: DCC-H vs DCC estándar, con corrección de frontera (γ >= 0):
    bajo H0 el LR se distribuye como ½χ²(0) + ½χ²(1) (Self & Liang, 1987),
    por lo que p = ½·P(χ²(1) > LR) y el valor crítico al 5% es 2.71.

    CORRECCIÓN [3]: si no hay días informativos (ver count_informative_days)
    γ no está identificado y el test NO se calcula (antes devolvía LR=0 y
    p=0.5, que se leía como "no significativo" cuando en realidad no había
    información). Tampoco se ocultan errores con valores ficticios: las
    excepciones se propagan y se muestran en la interfaz.
    """
    H = np.asarray(H)
    n_inf = count_informative_days(H, active, t_start)
    n_obs = len(H) - max(t_start, 1)

    res_r = estimate_dcc_parameters(Z, H, Q_bar, Qs, 'DCC', t_start)
    ll_r = -res_r.fun
    out = {'n_informativos': n_inf, 'n_obs_lik': n_obs, 'params_restricted': res_r.x,
           'log_lik_restricted': ll_r, 'df': 1,
           'critical_value': chi2.ppf(0.90, 1), 'critical_value_naive': chi2.ppf(0.95, 1)}

    if n_inf == 0:
        out.update({'identificado': False, 'lr_statistic': np.nan, 'p_value': np.nan,
                    'p_value_naive': np.nan, 'decision': 'NO_IDENTIFICADO',
                    'params_unrestricted': np.r_[res_r.x, 0.0], 'log_lik_unrestricted': ll_r,
                    'se': None})
        return out

    res_u = estimate_dcc_parameters(Z, H, Q_bar, Qs, 'DCC-H', t_start, restricted_params=res_r.x)
    ll_u = -res_u.fun
    lr = max(0.0, 2 * (ll_u - ll_r))
    p_naive = float(chi2.sf(lr, 1)) if lr > 1e-10 else 1.0
    p_corr = 0.5 * p_naive if lr > 1e-10 else 1.0

    out.update({'identificado': True, 'lr_statistic': lr, 'p_value': p_corr,
                'p_value_naive': p_naive,
                'decision': 'RECHAZAR_H0' if p_corr < ALPHA_TEST else 'NO_RECHAZAR_H0',
                'params_unrestricted': res_u.x, 'log_lik_unrestricted': ll_u,
                'se': compute_robust_se(Z, H, Q_bar, Qs, res_u.x, t_start) if compute_se else None})
    return out


def benjamini_hochberg(pvalues, alpha=0.05):
    """Corrección de Benjamini-Hochberg (FDR) sobre un conjunto de p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    if m == 0:
        return {'adjusted_pvalues': np.array([]), 'significant': np.array([], dtype=bool), 'cutoff': 0.0}
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    passed = ranked <= (np.arange(1, m + 1) / m) * alpha
    cutoff = ranked[np.max(np.where(passed)[0])] if passed.any() else 0.0
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj_p = np.empty(m)
    adj_p[order] = np.clip(adj, 0, 1)
    significant = pvalues <= cutoff if cutoff > 0 else np.zeros(m, dtype=bool)
    return {'adjusted_pvalues': adj_p, 'significant': significant, 'cutoff': float(cutoff)}


def distinct_specs(alpha_grid, kappa_grid, N):
    """
    [22] Especificaciones (α, κ) DISTINTAS. Como H_t=1 si (activos en tensión)/N >= κ,
    lo que importa es k = ⌈κ·N⌉: con N=6, κ=0.20 y κ=0.30 exigen ambos ≥2 activos y
    producen exactamente el mismo H_t. Devuelve ([(α, κ, k), ...], [(α, κ) descartados]).
    """
    specs, seen, dup = [], set(), []
    for a in sorted(set(float(x) for x in alpha_grid)):
        # De mayor a menor: entre κ equivalentes se conserva el mayor (p. ej. 0.30 y no 0.20)
        for k in sorted(set(float(x) for x in kappa_grid), reverse=True):
            cnt = int(np.ceil(k * N - 1e-9))
            if (a, cnt) in seen:
                dup.append((a, k))
                continue
            seen.add((a, cnt))
            specs.append((a, k, cnt))
    specs.sort(key=lambda x: (x[0], x[1]))
    return specs, dup


def run_robustness_panel(z_std, stress_series, loc_df, scale_df, Q_bar, t0, t_start, alpha_grid,
                         kappa_grid, min_obs=30):
    """
    Panel de robustez (Sección 4.6): re-ejecuta Gumbel → H_t → Q^(S) → LR para
    cada (α, κ) y aplica Benjamini-Hochberg SOLO sobre las especificaciones
    en las que γ está identificado (las otras no tienen p-value).
    """
    Z = z_std.values
    rows = []
    specs, dup = distinct_specs(alpha_grid, kappa_grid, Z.shape[1])
    ind_cache = {}
    for a_g, k_g, cnt in specs:
        if a_g not in ind_cache:
            ind_cache[a_g] = stress_indicators(stress_series, gumbel_thresholds(loc_df, scale_df, a_g))
        ind = ind_cache[a_g]
        H_g, _ = calculate_systemic_indicator(ind, k_g)
        Hv = H_g.values
        Qs_g, act_g = compute_recursive_Qstress(Z, Hv, Q_bar, min_obs)
        lr = likelihood_ratio_test(Z, Hv, Q_bar, Qs_g, act_g, t_start, compute_se=False)
        rows.append({
            'alpha_gumbel': a_g, 'kappa': k_g, 'activos_en_tension_min': cnt,
            'dias_Ht_ventana': int(Hv[t0:].sum()),
            'pct_Ht_ventana': float(Hv[t0:].mean() * 100),
            'dias_informativos': lr['n_informativos'],
            'LR_stat': lr['lr_statistic'],
            'p_value_corregido': lr['p_value'],
            'gamma': float(lr['params_unrestricted'][2]),
        })
    df = pd.DataFrame(rows)
    df['p_value_ajustado_BH'] = np.nan
    df['significativo_BH'] = False
    ident = df['p_value_corregido'].notna()
    if ident.any():
        bh = benjamini_hochberg(df.loc[ident, 'p_value_corregido'].values)
        df.loc[ident, 'p_value_ajustado_BH'] = bh['adjusted_pvalues']
        df.loc[ident, 'significativo_BH'] = bh['significant']
    df['n_especificaciones_evaluadas'] = len(df)
    df['n_especificaciones_identificadas'] = int(ident.sum())
    df.attrs['descartadas'] = dup
    return df


# ============================================================================
# 🔁 BOOTSTRAP PARAMÉTRICO DEL TEST LR  [20]  (Sección 3.6.1)
# ============================================================================
#
# Pregunta que responde: si γ fuera 0, ¿con qué frecuencia el procedimiento
# COMPLETO (GARCH → tensión → Q^(S) → LR) produciría un LR tan grande como el
# observado en estos datos? Evita depender de la aproximación asintótica
# ½χ²(0)+½χ²(1), que puede fallar con pocos días informativos, con residuos
# de colas pesadas y con estimación en dos etapas.
#
# Diseño (bootstrap paramétrico bajo H0 con innovaciones empíricas):
#   1. Se toman los parámetros estimados en los datos reales bajo H0:
#      GARCH(1,1) de cada activo, Q̄ y (a, b) del DCC restringido (γ = 0).
#   2. Innovaciones: e_t = L_t⁻¹ z_t, con L_t la factorización de Cholesky de
#      la R_t del DCC restringido, blanqueadas para que tengan covarianza
#      identidad. Se remuestrean filas completas con reposición: se conservan
#      las colas pesadas y la dependencia contemporánea de las colas.
#   3. Se simula la muestra completa (historia previa + ventana) con γ = 0.
#   4. Sobre cada muestra simulada se repite exactamente el procedimiento de
#      la app: se REESTIMA el GARCH (opcional), se recalcula H_t con la misma
#      definición de tensión, Q^(S) y el test LR con la misma verosimilitud.
#   5. p_bootstrap = (1 + #{LR* >= LR_obs}) / (B + 1)   (Davison & Hinkley, 1997)
#      Las réplicas sin días informativos cuentan como LR* = 0, que es lo que
#      el procedimiento produce en ese caso.
# ============================================================================

def whitened_innovations(z, R_t):
    """Innovaciones e_t = L_t⁻¹ z_t, centradas y blanqueadas (covarianza identidad)."""
    L = np.linalg.cholesky(R_t)
    E = np.linalg.solve(L, z[:, :, None])[:, :, 0]
    E = E - E.mean(axis=0)
    Lc = np.linalg.cholesky(np.cov(E.T, bias=True))
    return E @ np.linalg.inv(Lc).T


def simulate_restricted(rng, E, n_total, a, b, Q_bar, omega, alpha, beta, sigma2_0):
    """Simula (z, r) de longitud n_total desde el DCC restringido (γ=0) + GARCH por activo."""
    N = Q_bar.shape[0]
    idx = rng.integers(0, len(E), n_total)
    Z = np.empty((n_total, N))
    r = np.empty((n_total, N))
    Q_prev, s2, w = Q_bar, np.asarray(sigma2_0, dtype=float).copy(), 1 - a - b
    for t in range(n_total):
        if t == 0:
            Q = Q_bar
        else:
            Q = w * Q_bar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q_prev
            s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
        d = np.sqrt(np.diag(Q))
        Z[t] = np.linalg.cholesky(Q / np.outer(d, d)) @ E[idx[t]]
        r[t] = np.sqrt(s2) * Z[t]
        Q_prev = Q
    return Z, r


def bootstrap_context(res, cfg, reestimar_garch=True):
    """Todo lo necesario para simular bajo H0, extraído de una corrida de la app."""
    gdf = res['garch_df'].set_index('Ticker').loc[res['tickers']]
    a_r, b_r = [float(x) for x in res['lr']['params_restricted'][:2]]
    return {
        'E': whitened_innovations(res['z_std'].values, res['R_s']),
        'a': a_r, 'b': b_r, 'Q_bar': res['Q_bar'],
        'omega': gdf['omega'].values.astype(float), 'alpha': gdf['alpha'].values.astype(float),
        'beta': gdf['beta'].values.astype(float),
        'sigma2_0': res['sigma'][0] ** 2,
        'index': res['returns'].index, 'columns': list(res['returns'].columns),
        't_start': res['t_start'], 'cfg': dict(cfg), 'reestimar_garch': reestimar_garch,
    }


def bootstrap_replica(seed, ctx):
    """Una réplica bajo H0: simula y aplica el procedimiento completo de la app."""
    rng = np.random.default_rng(seed)
    n = len(ctx['index'])
    Zs, rs = simulate_restricted(rng, ctx['E'], n, ctx['a'], ctx['b'], ctx['Q_bar'],
                                 ctx['omega'], ctx['alpha'], ctx['beta'], ctx['sigma2_0'])
    ret = pd.DataFrame(rs, index=ctx['index'], columns=ctx['columns'])
    if ctx['reestimar_garch']:
        z_df, _, _, _ = garch_filter(ret)
    else:
        z_df = pd.DataFrame(Zs, index=ctx['index'], columns=ctx['columns'])
    Q_bar_h = ensure_positive_definite(np.corrcoef(z_df.values.T), min_eig=1e-6)
    _, _, _, _, H, _, Qs, active, _ = _stress_block(z_df, ret, ctx['cfg'], Q_bar_h)
    lr = likelihood_ratio_test(z_df.values, H.values, Q_bar_h, Qs, active, ctx['t_start'],
                               compute_se=False)
    return {'seed': seed, 'identificado': bool(lr['identificado']),
            'dias_informativos': int(lr['n_informativos']),
            'LR': float(lr['lr_statistic']) if lr['identificado'] else 0.0,
            'gamma_hat': float(lr['params_unrestricted'][2])}


def bootstrap_pvalue(lr_obs, lr_boot):
    lr_boot = np.asarray(lr_boot, dtype=float)
    return (1 + np.sum(lr_boot >= lr_obs - 1e-12)) / (len(lr_boot) + 1)


# ============================================================================
# 🔁 BOOTSTRAP DE ROBUSTEZ CON CORRECCIÓN DE ROMANO-WOLF  [23]
# ============================================================================
#
# Pregunta: considerando TODAS las especificaciones (α, κ) evaluadas, ¿cuáles
# rechazan H0: γ=0 controlando la probabilidad de cometer AL MENOS UN falso
# positivo (FWER)? A diferencia de Bonferroni o Benjamini-Hochberg, Romano-Wolf
# usa la distribución CONJUNTA de los LR bajo H0 (todas las especificaciones
# se evalúan sobre la misma muestra simulada), así que no penaliza de más
# cuando las especificaciones comparten la mayoría de los días de tensión.
#
# Algoritmo (Romano & Wolf, 2005; stepdown sobre el máximo, estadístico LR):
#   1. Ordenar las especificaciones por LR observado, de mayor a menor.
#   2. Para la j-ésima: p_RW = (1 + #{b : max_{i en las restantes} LR*_{b,i} ≥ LR_obs,j})/(B+1)
#   3. Imponer monotonía: p_RW(j) = max(p_RW(j), p_RW(j-1)).
# Además se reportan los p-values bootstrap individuales, sin corrección.
#
# Bajo H0 (γ=0) el DGP no depende de H_t, así que es el mismo para todas las
# especificaciones, y el DCC restringido se estima una sola vez por muestra.
# ============================================================================

def multi_spec_lr(z_df, stress_series, Q_bar, cfg, specs, t_start):
    """LR de varias especificaciones (α, κ) sobre los MISMOS datos."""
    Z = z_df.values
    T = len(Z)
    loc, scale = gumbel_rolling_params(stress_series, cfg['gumbel_window'], cfg['block_size'],
                                       cfg['gumbel_method'],
                                       expanding=(cfg['threshold_window'] == 'expansiva'))
    # El DCC restringido (γ=0) no usa H_t ni Q^(S): se estima una vez.
    H0 = np.zeros(T, dtype=int)
    Qs0 = np.broadcast_to(Q_bar, (T,) + Q_bar.shape)
    res_r = estimate_dcc_parameters(Z, H0, Q_bar, Qs0, 'DCC', t_start)
    ll_r = -res_r.fun
    out, ind_cache = [], {}
    for a_g, k_g, cnt in specs:
        if a_g not in ind_cache:
            ind_cache[a_g] = stress_indicators(stress_series, gumbel_thresholds(loc, scale, a_g))
        H, _ = calculate_systemic_indicator(ind_cache[a_g], k_g)
        Hv = H.values
        Qs, act = compute_recursive_Qstress(Z, Hv, Q_bar, cfg['min_obs_qs'])
        n_inf = count_informative_days(Hv, act, t_start)
        if n_inf == 0:
            out.append({'LR': 0.0, 'n_inf': 0, 'gamma': 0.0, 'identificado': False})
            continue
        res_u = estimate_dcc_parameters(Z, Hv, Q_bar, Qs, 'DCC-H', t_start, restricted_params=res_r.x)
        out.append({'LR': max(0.0, 2 * (-res_u.fun - ll_r)), 'n_inf': n_inf,
                    'gamma': float(res_u.x[2]), 'identificado': True})
    return out


def romano_wolf_replica(seed, ctx, specs):
    """Una réplica bajo H0: simula una muestra y calcula el LR de TODAS las especificaciones."""
    rng = np.random.default_rng(seed)
    n = len(ctx['index'])
    Zs, rs = simulate_restricted(rng, ctx['E'], n, ctx['a'], ctx['b'], ctx['Q_bar'],
                                 ctx['omega'], ctx['alpha'], ctx['beta'], ctx['sigma2_0'])
    ret = pd.DataFrame(rs, index=ctx['index'], columns=ctx['columns'])
    if ctx['reestimar_garch']:
        z_df, _, _, _ = garch_filter(ret)
    else:
        z_df = pd.DataFrame(Zs, index=ctx['index'], columns=ctx['columns'])
    Q_bar_h = ensure_positive_definite(np.corrcoef(z_df.values.T), min_eig=1e-6)
    X = ret.loc[z_df.index] if ctx['cfg']['stress_base'] == 'retornos' else z_df
    res = multi_spec_lr(z_df, X, Q_bar_h, ctx['cfg'], specs, ctx['t_start'])
    row = {'seed': seed}
    for i, r in enumerate(res):
        row[f'LR_{i}'] = r['LR']
    return row


def romano_wolf_pvalues(lr_obs, LRb):
    """p-values bootstrap individuales y ajustados por Romano-Wolf (stepdown, máximo LR)."""
    lr_obs = np.asarray(lr_obs, dtype=float)
    LRb = np.asarray(LRb, dtype=float)
    B, S = LRb.shape
    p_ind = (1 + (LRb >= lr_obs - 1e-12).sum(axis=0)) / (B + 1)
    order = np.argsort(-lr_obs)
    p_rw = np.empty(S)
    prev = 0.0
    for step, j in enumerate(order):
        mx = LRb[:, order[step:]].max(axis=1)
        p = (1 + np.sum(mx >= lr_obs[j] - 1e-12)) / (B + 1)
        prev = max(prev, p)
        p_rw[j] = prev
    return p_ind, p_rw


# ============================================================================
# ⚠️ VaR Y BACKTESTING
# ============================================================================

def calculate_var(R_t, sigma_matrix, weights=None, confidence=0.95):
    """VaR paramétrico del portafolio con Σ_t = D_t R_t D_t (vectorizado)."""
    T, N = sigma_matrix.shape
    w = np.ones(N) / N if weights is None else np.asarray(weights)
    ws = sigma_matrix * w
    sigma2_p = np.einsum('ti,tij,tj->t', ws, R_t, ws)
    sigma_p = np.sqrt(np.clip(sigma2_p, 1e-20, None))
    return -sigma_p * norm.ppf(1 - confidence)


def kupiec_test(violations, confidence):
    """
    Kupiec POF. CORRECCIÓN [4]: con 0 violaciones la versión anterior
    asignaba LR=0 (p=1, "aprobado"), cuando 0 violaciones en cientos de días
    indica un VaR excesivamente conservador. Con xlogy (0·log 0 = 0) la
    fórmula es válida para cualquier x, incluidos 0 y n.
    """
    v = np.asarray(violations, dtype=int)
    n, x = len(v), int(v.sum())
    p = 1 - confidence
    if n == 0:
        return np.nan, np.nan
    p_hat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = max(0.0, -2 * (ll0 - ll1))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_test(violations):
    """Test de independencia de Christoffersen (1998) sobre la secuencia de violaciones."""
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        return np.nan, np.nan
    v0, v1 = v[:-1], v[1:]
    n00 = np.sum((v0 == 0) & (v1 == 0)); n01 = np.sum((v0 == 0) & (v1 == 1))
    n10 = np.sum((v0 == 1) & (v1 == 0)); n11 = np.sum((v0 == 1) & (v1 == 1))
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = max(0.0, -2 * (ll_null - ll_alt))
    return float(lr), float(chi2.sf(lr, 1))


def backtest_var(portfolio_return, var_series, confidence=0.95):
    r = np.asarray(portfolio_return, dtype=float)
    var_series = np.asarray(var_series, dtype=float)
    violations = (r < -var_series).astype(int)
    n = len(violations)
    kup_lr, kup_p = kupiec_test(violations, confidence)
    ind_lr, ind_p = christoffersen_test(violations)
    cc_lr = kup_lr + ind_lr
    return {'violations': int(violations.sum()), 'n': n,
            'expected': n * (1 - confidence),
            'violation_rate': violations.mean() if n else 0.0,
            'expected_rate': 1 - confidence,
            'kupiec_lr': kup_lr, 'kupiec_pvalue': kup_p,
            'christ_ind_lr': ind_lr, 'christ_ind_pvalue': ind_p,
            'christ_cc_lr': cc_lr, 'christ_cc_pvalue': float(chi2.sf(cc_lr, 2)),
            'passed': kup_p > ALPHA_TEST, 'violation_series': violations}


def quantile_loss(portfolio_return, var_series, confidence):
    """Pérdida cuantílica (tick loss) del cuantil q_t = −VaR_t, con τ = 1−confianza."""
    r = np.asarray(portfolio_return, dtype=float)
    q = -np.asarray(var_series, dtype=float)
    tau = 1 - confidence
    return (tau - (r < q).astype(float)) * (r - q)


def diebold_mariano(loss_a, loss_b, lag=None):
    """DM con varianza de largo plazo Newey-West. Estadístico < 0 ⇒ 'a' tiene menor pérdida."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    if T < 10:
        return np.nan, np.nan, np.nan
    lag = int(np.floor(T ** (1 / 3))) if lag is None else lag
    dbar = d.mean()
    dc = d - dbar
    lrv = dc @ dc / T
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k] / T)
    if lrv <= 0:
        return np.nan, np.nan, float(dbar)
    stat = dbar / np.sqrt(lrv / T)
    return float(stat), float(2 * norm.sf(abs(stat))), float(dbar)


# ============================================================================
# 🔄 PIPELINE COMPLETO (sin dependencias de la interfaz)
# ============================================================================

def _stress_block(z_std, returns, cfg, Q_bar):
    """
    Bloque 2 + Q^(S). [14] La serie sobre la que se detecta la tensión es:
      - 'residuos': |z_t| del GARCH (tensión = SORPRESA respecto de la
        volatilidad reciente; especificación original de la tesis)
      - 'retornos': |r_t| brutos (tensión = ESTADO de alta volatilidad)
    Q^(S) y el DCC siempre usan z_t; solo cambia cómo se define H_t.
    """
    X = returns.loc[z_std.index] if cfg['stress_base'] == 'retornos' else z_std
    loc_df, scale_df = gumbel_rolling_params(X, cfg['gumbel_window'], cfg['block_size'],
                                             cfg['gumbel_method'],
                                             expanding=(cfg['threshold_window'] == 'expansiva'))
    thr_df = gumbel_thresholds(loc_df, scale_df, cfg['confidence_gumbel'])
    indicators = stress_indicators(X, thr_df)
    H_t, prop = calculate_systemic_indicator(indicators, cfg['kappa'])
    Qs, active = compute_recursive_Qstress(z_std.values, H_t.values, Q_bar, cfg['min_obs_qs'])
    return loc_df, scale_df, thr_df, indicators, H_t, prop, Qs, active, X


def out_of_sample_validation(returns, t0, cfg):
    """
    Out-of-Sample [7]: entrenamiento = historia previa, prueba = ventana de
    análisis. GARCH se estima solo en entrenamiento y se aplica congelado;
    Gumbel y Q^(S) se calculan causalmente sobre la serie combinada (en cada
    t solo usan información pasada); (a, b, γ) se estiman solo con
    entrenamiento.

    CORRECCIÓN: antes se concluía "mejor performance" si el DCC-H tenía
    MENOS violaciones, pero un VaR con muy pocas violaciones está mal
    calibrado. Ahora se evalúa cobertura (Kupiec, Christoffersen) y precisión
    (pérdida cuantílica media, test de Diebold-Mariano).
    """
    min_train = cfg['gumbel_window'] + 100
    if t0 < min_train:
        return None, (f"Historia previa insuficiente para el Out-of-Sample: {t0} días "
                      f"(se necesitan al menos {min_train}). Aumentá los años de historia previa.")
    n_test = len(returns) - t0
    if n_test < 50:
        return None, f"Ventana de prueba demasiado corta: {n_test} días (mínimo 50)."

    r_train, r_test = returns.iloc[:t0], returns.iloc[t0:]
    z_train, _, _, state = garch_filter(r_train)
    z_test, sigma_test = garch_filter_apply_fixed(r_test, state)
    z_full = pd.concat([z_train, z_test], axis=0)

    Q_bar_tr = ensure_positive_definite(np.corrcoef(z_train.values.T), min_eig=1e-6)
    _, _, _, _, H_full, prop_full, Qs_full, act_full, _ = _stress_block(z_full, returns, cfg,
                                                                         Q_bar_tr)
    Z_full, H_arr = z_full.values, H_full.values

    Z_tr, H_tr, Qs_tr = Z_full[:t0], H_arr[:t0], Qs_full[:t0]
    res_s = estimate_dcc_parameters(Z_tr, H_tr, Q_bar_tr, Qs_tr, 'DCC', 1)
    n_inf_train = count_informative_days(H_tr, act_full[:t0], 1)
    if n_inf_train > 0:
        res_h = estimate_dcc_parameters(Z_tr, H_tr, Q_bar_tr, Qs_tr, 'DCC-H', 1,
                                        restricted_params=res_s.x)
        params_h = res_h.x
    else:
        params_h = np.r_[res_s.x, 0.0]
    params_s = np.r_[res_s.x, 0.0]

    R_h, _ = dcc_filter(Z_full, H_arr, Q_bar_tr, Qs_full, params_h)
    R_s, _ = dcc_filter(Z_full, H_arr, Q_bar_tr, Qs_full, params_s)
    conf = cfg['var_confidence']
    var_h = calculate_var(R_h[t0:], sigma_test, confidence=conf)
    var_s = calculate_var(R_s[t0:], sigma_test, confidence=conf)
    port = r_test.mean(axis=1).values
    loss_h, loss_s = quantile_loss(port, var_h, conf), quantile_loss(port, var_s, conf)
    dm_stat, dm_p, dm_dbar = diebold_mariano(loss_h, loss_s)

    return {
        'train_period': f"{r_train.index[0]:%Y-%m-%d} a {r_train.index[-1]:%Y-%m-%d}",
        'test_period': f"{r_test.index[0]:%Y-%m-%d} a {r_test.index[-1]:%Y-%m-%d}",
        'n_train': t0, 'n_test': n_test,
        'params_h': params_h, 'params_s': params_s,
        'n_inf_train': n_inf_train,
        'n_inf_test': count_informative_days(H_arr, act_full, t0),
        'H_t_test': H_full.iloc[t0:], 'dates': r_test.index, 'port': port,
        'var_h': var_h, 'var_s': var_s,
        'bt_h': backtest_var(port, var_h, conf), 'bt_s': backtest_var(port, var_s, conf),
        'loss_h': float(loss_h.mean()), 'loss_s': float(loss_s.mean()),
        'dm_stat': dm_stat, 'dm_p': dm_p, 'dm_dbar': dm_dbar,
    }, None


def run_pipeline_core(prices, cfg):
    """
    Pipeline in-sample completo [3]. La muestra descargada = historia previa
    (burn-in) + ventana de análisis. La historia previa inicializa los
    umbrales de Gumbel, Q^(S)_t y la recursión de Q_t; la verosimilitud del
    DCC se suma sobre la ventana (o sobre toda la muestra, según cfg).
    """
    returns = calculate_returns(prices)
    if returns.shape[1] < 2:
        raise ValueError("El modelo DCC requiere al menos 2 activos con datos concurrentes.")
    t0 = int(returns.index.searchsorted(pd.Timestamp(cfg['analysis_start'])))
    n_window = len(returns) - t0
    if n_window < 30:
        raise ValueError(f"La ventana de análisis tiene solo {n_window} observaciones (mínimo 30).")

    z_std, sigma, garch_df, _ = garch_filter(returns)
    Q_bar = ensure_positive_definite(np.corrcoef(z_std.values.T), min_eig=1e-6)
    (loc_df, scale_df, thr_df, indicators, H_t, prop, Qs, active,
     stress_series) = _stress_block(z_std, returns, cfg, Q_bar)

    t_start = max(t0, 1) if cfg['lik_scope'] == 'ventana' else 1
    Z = z_std.values
    lr = likelihood_ratio_test(Z, H_t.values, Q_bar, Qs, active, t_start, compute_se=True)
    params_h = lr['params_unrestricted']
    params_s = np.r_[lr['params_restricted'][:2], 0.0]
    R_h, _ = dcc_filter(Z, H_t.values, Q_bar, Qs, params_h)
    R_s, _ = dcc_filter(Z, H_t.values, Q_bar, Qs, params_s)

    conf = cfg['var_confidence']
    var_h = calculate_var(R_h, sigma, confidence=conf)
    port = returns.mean(axis=1).values
    bt = backtest_var(port[t0:], var_h[t0:], conf)

    oos, oos_err = (None, None)
    if cfg['enable_oos']:
        try:
            oos, oos_err = out_of_sample_validation(returns, t0, cfg)
        except Exception as e:
            oos, oos_err = None, f"Error en la validación Out-of-Sample: {e}"

    return {
        'prices': prices, 'returns': returns, 't0': t0, 't_start': t_start,
        'tickers': list(returns.columns), 'z_std': z_std, 'sigma': sigma,
        'garch_df': garch_df, 'loc_df': loc_df, 'scale_df': scale_df, 'thr_df': thr_df,
        'stress_series': stress_series,
        'indicators': indicators, 'H_t': H_t, 'prop': prop, 'Q_bar': Q_bar,
        'n_active_window': int(active[t0:].sum()),
        'lr': lr, 'R_h': R_h, 'R_s': R_s, 'var_h': var_h, 'port': port, 'bt': bt,
        'oos': oos, 'oos_err': oos_err,
    }


@st.cache_data(show_spinner=False, max_entries=10)
def cached_pipeline(tickers, start, end, burn_years, confidence_gumbel, kappa, var_confidence,
                    gumbel_window, block_size, gumbel_method, min_obs_qs, lik_scope, enable_oos,
                    stress_base='residuos', threshold_window='movil'):
    dl_start = start - timedelta(days=int(round(365.25 * burn_years)))
    prices, err = download_data(tickers, dl_start, end + timedelta(days=1))
    if prices is None:
        return {'error': err}
    cfg = dict(analysis_start=start, confidence_gumbel=confidence_gumbel, kappa=kappa,
               var_confidence=var_confidence, gumbel_window=gumbel_window,
               block_size=block_size, gumbel_method=gumbel_method, min_obs_qs=min_obs_qs,
               lik_scope=lik_scope, enable_oos=enable_oos, stress_base=stress_base,
               threshold_window=threshold_window)
    try:
        res = run_pipeline_core(prices, cfg)
    except Exception as e:
        return {'error': f"Error al ejecutar el modelo: {e}"}
    res['dropped'] = sorted(set(tickers) - set(res['tickers']))
    return res


# ============================================================================
# 📊 VISUALIZACIONES
# ============================================================================

def plot_correlation_heatmap(R_window, tickers, title):
    avg_corr = np.mean(R_window[-60:], axis=0)
    fig = go.Figure(data=go.Heatmap(z=avg_corr, x=tickers, y=tickers, colorscale='RdBu',
                                    zmid=0, zmin=-1, zmax=1, text=np.round(avg_corr, 2),
                                    texttemplate="%{text}", textfont={"size": 10}))
    fig.update_layout(title=title, height=500)
    return fig


def plot_correlation_timeseries(R_h, R_s, dates, tickers, pair):
    i, j = pair
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=R_h[:, i, j], mode='lines', name='DCC-H',
                             line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=R_s[:, i, j], mode='lines', name='DCC estándar',
                             line=dict(color='#7f7f7f', width=1, dash='dot')))
    fig.update_layout(title=f"Correlación Dinámica: {tickers[i]} vs {tickers[j]}",
                      xaxis_title="Fecha", yaxis_title="Correlación",
                      yaxis=dict(range=[-1, 1]), height=400)
    return fig


def plot_homeostatic_indicator(H_t, prop_stressed, dates, kappa):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.3, 0.7])
    fig.add_trace(go.Scatter(x=dates, y=prop_stressed, mode='lines', name='Proporción en Tensión',
                             line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    # CORRECCIÓN [12]: la línea usa el κ elegido (antes estaba fija en 0.3)
    fig.add_hline(y=kappa, line_dash="dash", line_color="red",
                  annotation_text=f"Umbral κ={kappa:.2f}", row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=H_t, mode='lines', name='H_t (Homeostasis Activa)',
                             line=dict(color='#2ca02c', width=3), fill='tozeroy'), row=2, col=1)
    fig.update_layout(title="🏠 Indicador de Tensión Homeostática del Sistema", height=500)
    return fig


def plot_var_backtesting(port, var_h, dates, var_s=None, title="⚠️ Backtesting de Value-at-Risk"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=port, mode='lines', name='Retorno Portafolio',
                             line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(x=dates, y=-var_h, mode='lines', name='VaR DCC-H',
                             line=dict(color='#d62728', width=2, dash='dash')))
    if var_s is not None:
        fig.add_trace(go.Scatter(x=dates, y=-var_s, mode='lines', name='VaR DCC estándar',
                                 line=dict(color='#7f7f7f', width=1, dash='dot')))
    viol = port < -var_h
    fig.add_trace(go.Scatter(x=dates[viol], y=port[viol], mode='markers', name='Violaciones (DCC-H)',
                             marker=dict(color='red', size=8, symbol='x')))
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Retorno", height=400)
    return fig


def plot_tension_financiera(H_t, dates):
    """Días acumulados sin evento sistémico (latencia entre activaciones de H_t)."""
    tension = H_t.groupby((H_t == 1).cumsum()).cumcount()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=tension, mode='lines', name='Tensión Acumulada (Días)',
                             fill='tozeroy', line=dict(color='#ff4b4b', width=2)))
    reseteos = dates[H_t.values == 1]
    fig.add_trace(go.Scatter(x=reseteos, y=[0] * len(reseteos), mode='markers',
                             name='Reseteo Homeostático (H_t=1)',
                             marker=dict(color='#00d4ff', size=8, line=dict(width=1, color='white'))))
    fig.update_layout(title="📈 Curva de Presión del Mercado (Equivalente a Brechas de Primos)",
                      xaxis_title="Fecha", yaxis_title="Días sin Crisis Sistémica (Latencia)",
                      height=400, plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ============================================================================
# 🖥️ INTERFAZ
# ============================================================================

PRESETS = {
    "COVID-19 agudo (ene–jun 2020)": (date(2020, 1, 1), date(2020, 6, 30)),
    "COVID-19 completo (2020)": (date(2020, 1, 1), date(2020, 12, 31)),
    "COVID-19 extendido (2020–2021)": (date(2020, 1, 1), date(2021, 12, 31)),
    "Crisis Financiera Global (2008)": (date(2008, 1, 1), date(2008, 12, 31)),
    "Crisis Eurozona (2011)": (date(2011, 1, 1), date(2011, 12, 31)),
    "Período normal (2018–2019)": (date(2018, 1, 1), date(2019, 12, 31)),
    "Post-pandemia P6 (2022–hoy)": (date(2022, 1, 1), date.today()),
    "Personalizado": None,
}
TICKERS_MINIMUM = ['^GSPC', '^STOXX50E', 'TLT', 'GLD', 'UUP', 'EEM']
# Mismo diseño con historia larga en Yahoo (^STOXX50E arranca ~2007-03, UUP 2007-02)
TICKERS_MINIMUM_LONG = ['^GSPC', '^GDAXI', 'TLT', 'GLD', 'DX-Y.NYB', 'EEM']
TICKERS_COMPLETE = ['^GSPC', '^STOXX50E', '^N225', '^VIX', 'TLT', 'HYG',
                    'GLD', 'USO', 'FXE', 'UUP', 'EEM', 'BTC-USD']
PARAM_KEYS = ['tickers', 'start', 'end', 'burn_years', 'confidence_gumbel', 'kappa',
              'var_confidence', 'gumbel_window', 'block_size', 'gumbel_method',
              'min_obs_qs', 'lik_scope', 'enable_oos', 'stress_base', 'threshold_window']


def sidebar_config():
    st.sidebar.header("⚙️ Configuración del Modelo")

    st.sidebar.subheader("1. Activos")
    choice = st.sidebar.selectbox("Portafolio predefinido",
                                  ["Mínimo con historia larga (6 activos: ^GDAXI, DX-Y.NYB)",
                                   "Mínimo original (6 activos: ^STOXX50E, UUP)",
                                   "Completo (12 activos)", "Personalizado"])
    if choice.startswith("Completo"):
        default = TICKERS_COMPLETE
    elif choice.startswith("Mínimo original"):
        default = TICKERS_MINIMUM
    else:
        default = TICKERS_MINIMUM_LONG
    tickers_input = st.sidebar.text_area("Tickers (separados por coma)", value=", ".join(default))
    tickers = tuple(dict.fromkeys(t.strip() for t in tickers_input.split(",") if t.strip()))

    st.sidebar.subheader("2. Período")
    regime = st.sidebar.selectbox("Ventana de análisis", list(PRESETS.keys()))
    if PRESETS[regime] is None:
        c1, c2 = st.sidebar.columns(2)
        # [17] min_value explícito: por defecto Streamlit solo permite 10 años hacia atrás
        start = c1.date_input("Inicio", value=date(2020, 1, 1), min_value=date(1990, 1, 1),
                              max_value=date.today())
        end = c2.date_input("Fin", value=date.today(), min_value=date(1990, 1, 1),
                            max_value=date.today())
        nombre = st.sidebar.text_input("Nombre del período (para la comparativa)", value="")
        regime = nombre.strip() or f"Personalizado ({start:%Y-%m} a {end:%Y-%m})"
    else:
        start, end = PRESETS[regime]
        st.sidebar.caption(f"Ventana: {start} a {end}")
    burn_years = st.sidebar.slider(
        "Historia previa (años)", 1, 10, 3,
        help="Datos anteriores a la ventana que inicializan los umbrales de Gumbel, Q^(S) y la "
             "recursión del DCC. Sin historia previa, γ no se puede identificar en ventanas cortas.")

    st.sidebar.subheader("3. Especificación EVT / sistémica")
    st.sidebar.caption("Fijá α y κ ANTES de mirar resultados: es tu especificación principal "
                       "para todos los períodos. Otras combinaciones van al panel de robustez (5b).")
    base_label = st.sidebar.selectbox(
        "Tensión medida como",
        ["Sorpresa: residuos z del GARCH (especificación original)",
         "Estado: retornos brutos |r|"],
        help="Sorpresa = extremo respecto de la volatilidad reciente (el GARCH ya la descuenta). "
             "Estado = extremo respecto de la historia del activo: las crisis generan muchos días "
             "de tensión.")
    stress_base = 'residuos' if base_label.startswith("Sorpresa") else 'retornos'
    win_label = st.sidebar.selectbox(
        "Ventana del umbral de Gumbel",
        ["Móvil (últimos N días; especificación original)", "Expansiva (todo el pasado disponible)"],
        help="Con ventana móvil el umbral se adapta a la crisis y deja de marcarla después de unos "
             "meses. Con ventana expansiva recuerda las crisis anteriores.")
    threshold_window = 'movil' if win_label.startswith("Móvil") else 'expansiva'
    method_label = st.sidebar.selectbox("Método Gumbel",
                                        ["Máximos por bloque (recomendado)",
                                         "Todas las |z| (versión anterior, solo comparación)"])
    gumbel_method = 'bloques' if method_label.startswith("Máximos") else 'todas'
    block_size = st.sidebar.slider("Tamaño de bloque (días)", 2, 22, 5,
                                   disabled=(gumbel_method != 'bloques'),
                                   help="5 = máximos semanales.")
    gumbel_window = st.sidebar.slider(
        "Ventana de Gumbel (días)" if threshold_window == 'movil' else "Mínimo de días para el primer umbral",
        60, 500, 252)
    confidence_gumbel = st.sidebar.slider("Confianza Gumbel (α)", 0.90, 0.99, 0.95, 0.005)
    kappa = st.sidebar.slider("Umbral sistémico (κ)", 0.15, 0.60, 0.30, 0.05)
    min_obs_qs = st.sidebar.slider(
        "Mínimo de días de estrés para Q^(S)", 5, 60, 10,
        help="Q^(S) se contrae hacia Q̄ con peso 20/(n+20), lo que ya protege con pocas "
             "observaciones. Un mínimo alto deja a γ sin identificar en la mayoría de los períodos.")

    st.sidebar.subheader("4. Estimación y VaR")
    scope_label = st.sidebar.radio("Verosimilitud DCC sobre",
                                   ["Solo la ventana de análisis", "Historia previa + ventana"])
    lik_scope = 'ventana' if scope_label.startswith("Solo") else 'completa'
    var_confidence = st.sidebar.slider("Confianza VaR", 0.90, 0.99, 0.95, 0.01)

    st.sidebar.subheader("5. Validación")
    enable_oos = st.sidebar.checkbox("Out-of-Sample (entrena con la historia, prueba en la ventana)",
                                     value=True)

    cfg = dict(tickers=tickers, start=start, end=end, burn_years=burn_years,
               confidence_gumbel=round(float(confidence_gumbel), 4), kappa=round(float(kappa), 4),
               var_confidence=round(float(var_confidence), 4), gumbel_window=int(gumbel_window),
               block_size=int(block_size), gumbel_method=gumbel_method, min_obs_qs=int(min_obs_qs),
               lik_scope=lik_scope, enable_oos=bool(enable_oos), stress_base=stress_base,
               threshold_window=threshold_window)
    return cfg, regime


def _fmt_df(df, formats, na_rep='—'):
    """
    Formatea columnas como texto [16]. st.dataframe ignora el na_rep de un
    Styler y muestra 'None' en los valores faltantes; así se evita.
    """
    out = df.copy()
    for col, f in formats.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v: na_rep if pd.isna(v) else f.format(v))
    return out


def _fmt(x, f='{:.4f}'):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def render_bootstrap(res, cfg, run_key):
    """Interfaz del bootstrap paramétrico del test LR (Sección 3.6.1) [20]."""
    lr = res['lr']
    with st.expander("🔁 Bootstrap paramétrico del test LR (Sección 3.6.1)", expanded=True):
        st.caption(
            "Simula muestras completas bajo H0 (γ = 0) con los parámetros estimados en TUS datos "
            "(GARCH de cada activo, Q̄, a y b del DCC restringido) y remuestreo de las innovaciones "
            "empíricas. En cada réplica repite el procedimiento completo (GARCH → tensión → Q^(S) → "
            "LR). El p-value bootstrap es la proporción de réplicas con un LR al menos tan grande "
            "como el observado: no depende de la aproximación asintótica ½χ²(0)+½χ²(1).")
        c1, c2, c3 = st.columns(3)
        B = int(c1.selectbox("Réplicas (B)", [49, 99, 199, 499], index=2, key="boot_B",
                             help="199 da un p-value mínimo de 0,005; 499 es más preciso pero tarda "
                                  "más del doble."))
        reest = c2.checkbox("Reestimar el GARCH en cada réplica", value=True, key="boot_reest",
                            help="Recomendado: incorpora la incertidumbre de la Etapa 1.")
        seed = int(c3.number_input("Semilla", 0, 2**31 - 1, 20260923, key="boot_seed"))
        seg = B * 0.75 * len(res['returns']) / 1000
        st.caption(f"Tiempo estimado ≈ {max(1, round(seg / 60))}–{max(1, round(1.5 * seg / 60))} min. "
                   "No cierres ni recargues la pestaña mientras corre.")
        boot_key = f"boot::{run_key}::{B}::{reest}::{seed}"

        part_key = boot_key + "::parcial"
        parcial = st.session_state.get(part_key)
        if parcial and boot_key not in st.session_state:
            st.info(f"⏸️ Hay {len(parcial)} de {B} réplicas ya calculadas de una ejecución interrumpida. "
                    "Presioná *Ejecutar bootstrap* para continuar desde ahí.")

        if st.button("▶️ Ejecutar bootstrap", key="boot_btn"):
            ctx = bootstrap_context(res, cfg, reestimar_garch=reest)
            seeds = np.random.SeedSequence(seed).generate_state(B)
            # [21] La lista vive en session_state: cada réplica terminada queda guardada
            # aunque Streamlit interrumpa el script, y el próximo clic continúa.
            rows = st.session_state.setdefault(part_key, [])
            hechas = len(rows)
            bar, status = st.progress(hechas / B), st.empty()
            t_ini = time.time()
            for i in range(hechas, B):
                sd = int(seeds[i])
                try:
                    rows.append(bootstrap_replica(sd, ctx))
                except Exception as e:
                    rows.append({'seed': sd, 'error': f"{type(e).__name__}: {e}"})
                el, k = time.time() - t_ini, i + 1 - hechas
                bar.progress((i + 1) / B)
                status.caption(f"{i + 1}/{B} réplicas — {el / 60:.1f} min en esta ejecución, "
                               f"≈ {el / k * (B - i - 1) / 60:.1f} min restantes")
            st.session_state[boot_key] = pd.DataFrame(rows)
            del st.session_state[part_key]

        if boot_key not in st.session_state:
            return
        bdf = st.session_state[boot_key]
        ok = bdf[bdf['error'].isna()] if 'error' in bdf.columns else bdf
        n_fail = len(bdf) - len(ok)
        if len(ok) == 0:
            st.error(f"Todas las réplicas fallaron. Primer error: {bdf['error'].dropna().iloc[0]}")
            return
        LRb = ok['LR'].values.astype(float)
        lr_obs = float(lr['lr_statistic'])
        p_boot = bootstrap_pvalue(lr_obs, LRb)
        crit95 = float(np.quantile(LRb, 0.95))
        st.session_state[f"boot_p::{run_key}"] = (p_boot, len(ok))

        c = st.columns(4)
        c[0].metric("p-value bootstrap", f"{p_boot:.4f}")
        c[1].metric("p-value asintótico", f"{lr['p_value']:.4f}")
        c[2].metric("LR observado", f"{lr_obs:.3f}")
        c[3].metric("Valor crítico bootstrap (5%)", f"{crit95:.3f}",
                    help="Percentil 95 de los LR simulados bajo H0 (el asintótico es 2,71).")
        if n_fail:
            st.warning(f"{n_fail} réplica(s) fallaron y se excluyeron. Primer error: "
                       f"{bdf['error'].dropna().iloc[0]}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=LRb, nbinsx=40, name='LR* bajo H0', marker_color='#7f7f7f'))
        fig.add_vline(x=lr_obs, line_color='#d62728', line_width=3,
                      annotation_text=f"LR observado = {lr_obs:.2f}")
        fig.update_layout(title="Distribución bootstrap del LR bajo H0: γ = 0",
                          xaxis_title="LR", yaxis_title="Réplicas", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"{len(ok)} réplicas válidas · γ identificado en {ok['identificado'].mean():.0%} · "
                   f"LR* > 0 en {np.mean(LRb > 1e-8):.0%} · γ̂* medio bajo H0 = "
                   f"{ok['gamma_hat'].mean():.4f} (γ̂ observado = {lr['params_unrestricted'][2]:.4f}) · "
                   f"resolución: el p-value mínimo posible es {1 / (len(ok) + 1):.4f}")
        if p_boot < 0.05:
            st.success(f"Un LR como el observado aparece en menos del 5% de las muestras simuladas sin "
                       f"efecto homeostático (p = {p_boot:.4f}): evidencia a favor de γ > 0 con esta "
                       "especificación. Documentar en la tesis cuándo y por qué se eligió la definición "
                       "de tensión usada.")
        elif p_boot < 0.10:
            st.info(f"Evidencia débil (p = {p_boot:.4f}): significativo al 10% pero no al 5%.")
        else:
            st.warning(f"Un LR como el observado es frecuente sin efecto homeostático (p = {p_boot:.4f}): "
                       "los datos son compatibles con γ = 0.")
        st.download_button("📥 Réplicas del bootstrap (CSV)", bdf.to_csv(index=False),
                           file_name=f"bootstrap_lr_{datetime.now():%Y%m%d}.csv", mime="text/csv",
                           key="boot_dl")


def render_romano_wolf(res, cfg, run_key, a_sel, k_sel):
    """Interfaz del bootstrap de robustez con corrección de Romano-Wolf [23]."""
    lr = res['lr']
    N = len(res['tickers'])
    with st.expander("🔁 Bootstrap de robustez con corrección de Romano-Wolf", expanded=False):
        st.caption(
            "Evalúa TODAS las especificaciones (α, κ) seleccionadas arriba sobre cada muestra "
            "simulada bajo H0 (γ=0). Da, para cada especificación, el p-value bootstrap individual y "
            "el ajustado por Romano-Wolf, que controla la probabilidad de al menos un falso positivo "
            "respetando la correlación entre especificaciones. Incluye la especificación principal: "
            "su p-value individual sirve además como confirmación del bootstrap de la sección 5.")
        specs, dup = distinct_specs(a_sel, k_sel, N)
        if not specs:
            st.warning("Seleccioná valores de α y κ en el panel de arriba.")
            return
        main_cnt = int(np.ceil(cfg['kappa'] * N - 1e-9))
        if not any(abs(a - cfg['confidence_gumbel']) < 1e-9 and c == main_cnt for a, _, c in specs):
            st.warning("La especificación principal no está en la grilla: agregá su α y su κ arriba.")
        st.markdown(f"**{len(specs)} especificaciones distintas:** " +
                    " · ".join(f"α={a:.2f}, ≥{c} activos" for a, _, c in specs))
        if dup:
            st.caption("Equivalentes descartadas: " + ", ".join(f"α={a}, κ={k}" for a, k in dup))

        c1, c2, c3 = st.columns(3)
        B = int(c1.selectbox("Réplicas (B)", [199, 499, 999], index=2, key="rw_B",
                             help="999 da un p-value mínimo de 0,001 y confirma el bootstrap de la "
                                  "sección 5 con más precisión."))
        reest = c2.checkbox("Reestimar el GARCH en cada réplica", value=True, key="rw_reest")
        seed = int(c3.number_input("Semilla", 0, 2**31 - 1, 12345, key="rw_seed",
                                   help="Usá una semilla distinta de la del bootstrap de la sección 5."))
        seg = B * len(res['returns']) / 1000 * (0.9 + 0.18 * len(specs))
        st.caption(f"Tiempo estimado en una computadora ≈ {seg / 3600:.1f}–{1.5 * seg / 3600:.1f} h. "
                   "Se puede interrumpir: al volver a presionar el botón continúa desde donde quedó.")

        spec_sig = tuple((round(a, 4), c) for a, _, c in specs)
        rw_key = f"rw::{run_key}::{spec_sig}::{B}::{reest}::{seed}"
        part_key = rw_key + "::parcial"
        parcial = st.session_state.get(part_key)
        if parcial and rw_key not in st.session_state:
            st.info(f"⏸️ Hay {len(parcial)} de {B} réplicas ya calculadas. Presioná el botón para continuar.")

        if st.button("▶️ Ejecutar bootstrap Romano-Wolf", key="rw_btn"):
            with st.spinner("Calculando los LR observados de cada especificación..."):
                obs = multi_spec_lr(res['z_std'], res['stress_series'], res['Q_bar'], cfg, specs,
                                    res['t_start'])
            ctx = bootstrap_context(res, cfg, reestimar_garch=reest)
            seeds = np.random.SeedSequence(seed).generate_state(B)
            rows = st.session_state.setdefault(part_key, [])
            hechas = len(rows)
            bar, status = st.progress(hechas / B), st.empty()
            t_ini = time.time()
            for i in range(hechas, B):
                sd = int(seeds[i])
                try:
                    rows.append(romano_wolf_replica(sd, ctx, specs))
                except Exception as e:
                    rows.append({'seed': sd, 'error': f"{type(e).__name__}: {e}"})
                el, k = time.time() - t_ini, i + 1 - hechas
                bar.progress((i + 1) / B)
                status.caption(f"{i + 1}/{B} réplicas — {el / 60:.1f} min en esta ejecución, "
                               f"≈ {el / k * (B - i - 1) / 60:.0f} min restantes")
            st.session_state[rw_key] = {'obs': obs, 'rows': pd.DataFrame(rows)}
            del st.session_state[part_key]

        if rw_key not in st.session_state:
            return
        data = st.session_state[rw_key]
        obs, bdf = data['obs'], data['rows']
        ok = bdf[bdf['error'].isna()] if 'error' in bdf.columns else bdf
        if len(ok) == 0:
            st.error(f"Todas las réplicas fallaron. Primer error: {bdf['error'].dropna().iloc[0]}")
            return
        if len(ok) < len(bdf):
            st.warning(f"{len(bdf) - len(ok)} réplica(s) fallaron y se excluyeron. Primer error: "
                       f"{bdf['error'].dropna().iloc[0]}")
        LRb = ok[[f'LR_{i}' for i in range(len(specs))]].values.astype(float)
        lr_obs = np.array([o['LR'] for o in obs])
        p_ind, p_rw = romano_wolf_pvalues(lr_obs, LRb)
        p_asym = [0.5 * float(chi2.sf(x, 1)) if x > 1e-10 else 1.0 for x in lr_obs]
        tab = pd.DataFrame({
            'α Gumbel': [a for a, _, _ in specs], 'κ': [k for _, k, _ in specs],
            'Activos en tensión (mín.)': [c for _, _, c in specs],
            'Principal': ['★' if abs(a - cfg['confidence_gumbel']) < 1e-9 and c == main_cnt else ''
                          for a, _, c in specs],
            'Días informativos': [o['n_inf'] for o in obs], 'γ̂': [o['gamma'] for o in obs],
            'LR': lr_obs, 'p asintótico': p_asym, 'p bootstrap individual': p_ind,
            'p Romano-Wolf': p_rw})
        st.dataframe(_fmt_df(tab, {'α Gumbel': '{:.2f}', 'κ': '{:.2f}', 'γ̂': '{:.4f}', 'LR': '{:.3f}',
                                   'p asintótico': '{:.4f}', 'p bootstrap individual': '{:.4f}',
                                   'p Romano-Wolf': '{:.4f}'}), use_container_width=True)
        n_rw = int((p_rw < 0.05).sum())
        st.caption(f"{len(ok)} réplicas válidas · p-value mínimo posible: {1 / (len(ok) + 1):.4f}")
        main_idx = [i for i, (a, _, c) in enumerate(specs)
                    if abs(a - cfg['confidence_gumbel']) < 1e-9 and c == main_cnt]
        if main_idx:
            i = main_idx[0]
            st.session_state[f"boot_p::{run_key}"] = (float(p_ind[i]), len(ok))
            msg = (f"Especificación principal: p bootstrap individual = {p_ind[i]:.4f} · "
                   f"p Romano-Wolf = {p_rw[i]:.4f}.")
            (st.success if p_rw[i] < 0.05 else st.info)(msg)
        if n_rw:
            st.success(f"{n_rw} de {len(specs)} especificaciones rechazan H0 al 5% después de corregir "
                       "por todas las especificaciones evaluadas (Romano-Wolf).")
        else:
            st.warning("Ninguna especificación rechaza H0 al 5% después de corregir por todas las "
                       "especificaciones evaluadas (Romano-Wolf).")
        st.download_button("📥 Tabla Romano-Wolf (CSV)", tab.to_csv(index=False),
                           file_name=f"romano_wolf_{datetime.now():%Y%m%d}.csv", mime="text/csv",
                           key="rw_dl_tab")
        st.download_button("📥 Réplicas del bootstrap Romano-Wolf (CSV)", bdf.to_csv(index=False),
                           file_name=f"romano_wolf_replicas_{datetime.now():%Y%m%d}.csv",
                           mime="text/csv", key="rw_dl_rep")


def render_results(res, cfg, regime, run_key):
    returns, t0, tickers = res['returns'], res['t0'], res['tickers']
    dates = returns.index[t0:]
    r_win = returns.iloc[t0:]
    H_win, prop_win = res['H_t'].iloc[t0:], res['prop'].iloc[t0:]
    lr, bt = res['lr'], res['bt']

    # ---------------------------------------------------------------- 1. Datos
    st.markdown('<p class="sub-header">📥 1. Datos</p>', unsafe_allow_html=True)
    if res.get('dropped'):
        st.warning(f"⚠️ Activos excluidos (sin datos en el período): {', '.join(res['dropped'])}")
    if '^VIX' in tickers:
        st.info("ℹ️ ^VIX es un índice no negociable: su inclusión en un VaR de portafolio es "
                "discutible. Considerá justificarla o excluirlo.")
    c = st.columns(4)
    c[0].metric("Activos", len(tickers))
    c[1].metric("Ventana de análisis", f"{dates[0]:%Y-%m-%d} a {dates[-1]:%Y-%m-%d}")
    c[2].metric("Obs. ventana", len(dates))
    c[3].metric("Obs. historia previa", t0)
    if t0 < cfg['gumbel_window']:
        st.warning(f"⚠️ La historia previa efectiva ({t0} días) es menor que la ventana de Gumbel "
                   f"({cfg['gumbel_window']}): los primeros días de la ventana no tienen umbral.")
    with st.expander("📋 Ver precios (últimas filas)"):
        st.dataframe(res['prices'].tail(10))

    # ---------------------------------------------------------------- 2. GARCH
    st.markdown("---")
    st.markdown('<p class="sub-header">📈 2. Retornos y Filtrado GARCH</p>', unsafe_allow_html=True)
    port_win = res['port'][t0:]
    c = st.columns(3)
    c[0].metric("Retorno medio anual (ventana)", f"{r_win.mean().mean() * 252:.2%}")
    # CORRECCIÓN [12]: antes se calculaba returns.std().std() (desvío de los desvíos)
    c[1].metric("Volatilidad media individual (anual)", f"{r_win.std().mean() * np.sqrt(252):.2%}")
    c[2].metric("Volatilidad portafolio EW (anual)", f"{np.std(port_win, ddof=1) * np.sqrt(252):.2%}")
    with st.expander("📐 Parámetros GARCH(1,1) por activo (MLE) y diagnósticos"):
        gdf = res['garch_df']
        st.dataframe(_fmt_df(gdf, {
            'omega': '{:.2e}', 'alpha': '{:.4f}', 'beta': '{:.4f}', 'persistencia (a+b)': '{:.4f}',
            'Ljung-Box p (z)': '{:.4f}', 'Ljung-Box p (z²)': '{:.4f}', 'ARCH-LM p': '{:.4f}'}),
            use_container_width=True)
        n_nc = int((~gdf['Convergió']).sum())
        if n_nc:
            st.warning(f"⚠️ {n_nc} activo(s) no convergieron en la optimización GARCH.")
        n_lb = int((gdf['Ljung-Box p (z²)'] < 0.05).sum())
        if n_lb:
            st.info(f"ℹ️ {n_lb} activo(s) con autocorrelación residual en z² (Ljung-Box p<0.05).")

    # ---------------------------------------------------------------- 3. Gumbel
    st.markdown("---")
    st.markdown('<p class="sub-header">🎯 3. Valores Extremos (Gumbel) e Indicador H_t</p>',
                unsafe_allow_html=True)
    serie_txt = "|z| (residuos del GARCH)" if cfg['stress_base'] == 'residuos' else "|r| (retornos brutos)"
    ventana_txt = (f"una ventana móvil estrictamente pasada de {cfg['gumbel_window']} días"
                   if cfg['threshold_window'] == 'movil'
                   else f"una ventana expansiva (todo el pasado disponible, mínimo {cfg['gumbel_window']} días)")
    concepto = ("SORPRESA respecto de la volatilidad reciente" if cfg['stress_base'] == 'residuos'
                else "ESTADO de alta volatilidad respecto de la historia del activo")
    st.markdown(f"**Definición de tensión:** {concepto}.")
    if cfg['gumbel_method'] == 'bloques':
        st.caption(f"Gumbel ajustada por L-momentos sobre máximos de bloques de {cfg['block_size']} "
                   f"días de {serie_txt}, en {ventana_txt}. Un activo está en tensión si su valor "
                   f"supera el cuantil α={cfg['confidence_gumbel']} de esos máximos.")
        if cfg['stress_base'] == 'residuos' and cfg['threshold_window'] == 'movil':
            tasa = 100 * (1 - cfg['confidence_gumbel'] ** (1 / cfg['block_size']))
            st.info(f"ℹ️ Con esta construcción, cada activo supera su umbral en ≈{tasa:.1f}% de los días "
                    "**por construcción, en cualquier régimen**: el GARCH descuenta la volatilidad y la "
                    "ventana móvil adapta el umbral. H_t mide coincidencia de sorpresas, no crisis. "
                    "Para medir estado de tensión, usá retornos brutos con ventana expansiva.")
    else:
        st.caption("⚠️ Método de comparación: Gumbel sobre todas las |z| (criterio de la versión "
                   "anterior, no consistente con la teoría de valores extremos).")
    c1, c2 = st.columns(2)
    with c1:
        thr_win = res['thr_df'].iloc[t0:]
        tdf = pd.DataFrame({'Ticker': tickers,
                            'Umbral promedio (ventana)': thr_win.mean().values,
                            '% días en tensión': res['indicators'].iloc[t0:].mean().values * 100})
        st.dataframe(_fmt_df(tdf, {'Umbral promedio (ventana)': '{:.4f}',
                                   '% días en tensión': '{:.1f}'}))
    with c2:
        st.metric("Días con H_t=1 (ventana)", int(H_win.sum()))
        st.metric("Porcentaje del tiempo", f"{H_win.mean() * 100:.1f}%")
    st.plotly_chart(plot_homeostatic_indicator(H_win.values, prop_win.values, dates, cfg['kappa']),
                    use_container_width=True)
    with st.expander("📈 Latencia entre eventos sistémicos"):
        st.plotly_chart(plot_tension_financiera(H_win, dates), use_container_width=True)

    # ---------------------------------------------------------------- 4. DCC-H
    st.markdown("---")
    st.markdown('<p class="sub-header">🔗 4. Correlación Dinámica (DCC-H)</p>', unsafe_allow_html=True)
    a_h, b_h, g_h = lr['params_unrestricted'][:3]
    c = st.columns(4)
    c[0].metric("a (shock)", f"{a_h:.4f}")
    c[1].metric("b (persistencia)", f"{b_h:.4f}")
    c[2].metric("γ (homeostasis)", f"{g_h:.4f}")
    c[3].metric("Días informativos para γ", lr['n_informativos'],
                help="Días de la muestra de estimación con H_{t-1}=1 y Q^(S)_t distinto de Q̄. "
                     "Si es 0, γ no está identificado.")
    if not lr['identificado']:
        st.warning("⚠️ **γ no está identificado** en esta corrida: no hubo días con H_t=1 en los "
                   "que Q^(S) ya estuviera estimado (se necesitan al menos "
                   f"{cfg['min_obs_qs']} días de estrés previos). Se muestra el DCC estándar.")
    elif lr['n_informativos'] < 10:
        st.info(f"ℹ️ Identificación débil: solo {lr['n_informativos']} días informativos para γ. "
                "Los errores estándar pueden ser muy grandes; considerá estimar sobre historia "
                "previa + ventana (Sección 4 de la barra lateral).")
    st.plotly_chart(plot_correlation_heatmap(res['R_h'][t0:], tickers,
                                             "Correlación promedio DCC-H (últimos 60 días de la ventana)"),
                    use_container_width=True)
    st.markdown("**Seleccionar par de activos:**")
    c1, c2 = st.columns(2)
    a1 = c1.selectbox("Activo 1", tickers, index=0, key="asset1")
    a2 = c2.selectbox("Activo 2", tickers, index=1 if len(tickers) > 1 else 0, key="asset2")
    st.plotly_chart(plot_correlation_timeseries(res['R_h'][t0:], res['R_s'][t0:], dates, tickers,
                                                (tickers.index(a1), tickers.index(a2))),
                    use_container_width=True)

    # ---------------------------------------------------------------- 5. LR
    st.markdown("---")
    st.markdown('<p class="sub-header">🧪 5. Test de Razón de Verosimilitud (H2)</p>',
                unsafe_allow_html=True)
    scope_txt = "la ventana de análisis" if cfg['lik_scope'] == 'ventana' else "historia previa + ventana"
    st.caption(f"Verosimilitud evaluada sobre {scope_txt} ({lr['n_obs_lik']} observaciones).")
    if not lr['identificado']:
        st.warning("El test LR no es aplicable porque γ no está identificado (ver Sección 4). "
                   "Esto no equivale a 'γ no significativo': significa que la muestra no contiene "
                   "información para evaluarlo.")
    else:
        c = st.columns(4)
        c[0].metric("Estadístico LR", f"{lr['lr_statistic']:.4f}")
        c[1].metric("Valor crítico corregido (5%)", f"{lr['critical_value']:.4f}",
                    help="Mezcla ½χ²(0)+½χ²(1) por la restricción de frontera γ≥0 (Self & Liang, 1987).")
        c[2].metric("p-value corregido", f"{lr['p_value']:.6f}",
                    help=f"p-value naive χ²(1), solo referencia: {lr['p_value_naive']:.6f}")
        if lr['decision'] == "RECHAZAR_H0":
            c[3].success("✅ H0 rechazada")
        else:
            c[3].error("❌ H0 no rechazada")

        se = lr['se']
        if se is not None:
            est = np.asarray(lr['params_unrestricted'][:3])
            t_stat = np.where(se['frontera'], np.nan, est / se['se_sandwich'])
            se_df = pd.DataFrame({
                'Parámetro': ['a (shock)', 'b (persistencia)', 'γ (homeostasis)'],
                'Estimación': est, 'SE sandwich': se['se_sandwich'], 'SE OPG': se['se_opg'],
                't (sandwich)': t_stat,
                'En frontera': ['sí' if f else 'no' for f in se['frontera']]})
            st.markdown("**Errores estándar de la Etapa 2:**")
            st.dataframe(_fmt_df(se_df, {'Estimación': '{:.4f}', 'SE sandwich': '{:.4f}',
                                         'SE OPG': '{:.4f}', 't (sandwich)': '{:.2f}'}),
                         use_container_width=True)
            st.caption("En frontera (p. ej. γ≈0) el SE y el t no tienen distribución normal y no se "
                       "reportan: la inferencia sobre γ se basa en el test LR corregido. Estos SE no "
                       "propagan la incertidumbre de la Etapa 1 (GARCH); complementar con bootstrap "
                       "paramétrico (Sección 3.9).")

        if lr['decision'] == "RECHAZAR_H0":
            st.success("La evidencia es compatible con H2 en este período y especificación: incluir γ "
                       "mejora significativamente el ajuste respecto del DCC estándar. Reportar junto "
                       "con el panel de robustez (5b).")
        else:
            st.warning("No hay evidencia suficiente de γ>0 en este período con esta especificación. "
                       "Es un resultado informativo en sí mismo. Si después de verlo se cambian α o κ, "
                       "el cambio debe declararse y corregirse por comparaciones múltiples (5b).")

    # ---------------------------------------------------------------- 5c. Bootstrap [20]
    if lr['identificado']:
        render_bootstrap(res, cfg, run_key)

    n_lik = lr['n_obs_lik']
    k_u = 3 if lr['identificado'] else 2
    comp = pd.DataFrame({
        'Modelo': ['DCC estándar', 'DCC homeostático'], 'Parámetros': [2, k_u],
        'Log-Likelihood': [lr['log_lik_restricted'], lr['log_lik_unrestricted']]})
    comp['AIC'] = -2 * comp['Log-Likelihood'] + 2 * comp['Parámetros']
    comp['BIC'] = -2 * comp['Log-Likelihood'] + comp['Parámetros'] * np.log(n_lik)
    st.markdown("### 📊 Comparación de modelos")
    st.dataframe(_fmt_df(comp, {'Log-Likelihood': '{:.4f}', 'AIC': '{:.4f}', 'BIC': '{:.4f}'}))

    # ---------------------------------------------------------------- 5b. Robustez
    st.markdown("---")
    st.markdown('<p class="sub-header">🧪 5b. Panel de Robustez (Benjamini-Hochberg)</p>',
                unsafe_allow_html=True)
    st.caption("Evalúa una grilla de α y κ y corrige por comparaciones múltiples (FDR). Las "
               "especificaciones sin días informativos no tienen p-value y quedan fuera de la corrección.")
    c1, c2 = st.columns(2)
    a_opts = sorted({0.90, 0.95, 0.97, 0.99, cfg['confidence_gumbel']})
    k_opts = sorted({0.20, 0.30, 0.45, 0.60, cfg['kappa']})
    a_sel = c1.multiselect("Valores de α (Gumbel)", a_opts,
                           default=sorted({cfg['confidence_gumbel'], 0.90, 0.95, 0.97, 0.99}),
                           help="Por defecto: la grilla de la Sección 4.4.3 más α=0.90.")
    k_sel = c2.multiselect("Valores de κ", k_opts, default=sorted({cfg['kappa'], 0.30, 0.45, 0.60}),
                           help=f"Con {len(tickers)} activos, lo que importa es cuántos activos exige "
                                f"cada κ (⌈κ·N⌉); los κ equivalentes se evalúan una sola vez.")
    rob_key = f"robustez::{run_key}"
    if st.button("▶️ Ejecutar panel de robustez", key="run_robustness_btn"):
        if not a_sel or not k_sel:
            st.warning("Seleccioná al menos un valor de α y uno de κ.")
        else:
            n_specs = len(a_sel) * len(k_sel)
            with st.spinner(f"Evaluando {n_specs} especificaciones..."):
                try:
                    st.session_state[rob_key] = run_robustness_panel(
                        res['z_std'], res['stress_series'], res['loc_df'], res['scale_df'],
                        res['Q_bar'], t0,
                        res['t_start'], a_sel, k_sel, cfg['min_obs_qs'])
                except Exception as e:
                    st.error(f"Error en el panel de robustez: {e}")
    if rob_key in st.session_state:
        rdf = st.session_state[rob_key]
        n_specs = int(rdf['n_especificaciones_evaluadas'].iloc[0])
        n_id = int(rdf['n_especificaciones_identificadas'].iloc[0])
        st.info(f"Se evaluaron **{n_specs} especificaciones distintas**; γ identificado en **{n_id}**.")
        if rdf.attrs.get('descartadas'):
            st.caption("Descartadas por ser equivalentes a otra (mismo número mínimo de activos en "
                       "tensión): " + ", ".join(f"α={a}, κ={k}" for a, k in rdf.attrs['descartadas']))
        st.dataframe(_fmt_df(rdf, {'pct_Ht_ventana': '{:.2f}', 'LR_stat': '{:.4f}',
                                   'p_value_corregido': '{:.6f}', 'gamma': '{:.4f}',
                                   'p_value_ajustado_BH': '{:.6f}'}, na_rep='no identificado'),
                     use_container_width=True)
        n_raw = int((rdf['p_value_corregido'] < 0.05).sum())
        n_bh = int(rdf['significativo_BH'].sum())
        st.markdown(f"**Significativos sin corrección (p<0.05):** {n_raw} de {n_id}  \n"
                    f"**Significativos tras BH-FDR:** {n_bh} de {n_id}")
        if n_raw > n_bh:
            st.warning("La corrección reduce las especificaciones significativas: reportar los "
                       "resultados ajustados.")

    render_romano_wolf(res, cfg, run_key, a_sel, k_sel)

    # ---------------------------------------------------------------- 6. Fase
    st.markdown("---")
    st.markdown('<p class="sub-header">🎯 6. Clasificación Automática de Fase</p>',
                unsafe_allow_html=True)
    dias_ht_pct = float(H_win.mean() * 100)
    fase_info = clasificar_fase(lr['p_value'], bt['kupiec_pvalue'], dias_ht_pct, lr['identificado'],
                                n_informativos=lr['n_informativos'])
    if len(dates) > 3 * 252:
        st.warning(f"⚠️ La ventana tiene {len(dates)} días (más de 3 años): probablemente contiene varios "
                   "regímenes, y una única fase los promedia. Para clasificar fases usá ventanas acotadas "
                   "a un episodio.")
    mostrar_fase_detectada(fase_info)

    # ---------------------------------------------------------------- 7. VaR
    st.markdown("---")
    st.markdown('<p class="sub-header">⚠️ 7. VaR Condicional y Backtesting (ventana, in-sample)</p>',
                unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Violaciones observadas", bt['violations'])
    c[1].metric("Violaciones esperadas", f"{bt['expected']:.1f}")
    c[2].metric("Tasa observada", f"{bt['violation_rate'] * 100:.2f}%")
    c[3].metric("Tasa esperada", f"{bt['expected_rate'] * 100:.2f}%")
    msg = (f"Kupiec p = {bt['kupiec_pvalue']:.4f} · Christoffersen independencia p = "
           f"{_fmt(bt['christ_ind_pvalue'])} · cobertura condicional p = {_fmt(bt['christ_cc_pvalue'])}")
    (st.success if bt['passed'] else st.error)(
        ("✅ Test de Kupiec aprobado. " if bt['passed'] else "❌ Test de Kupiec rechazado. ") + msg)
    st.plotly_chart(plot_var_backtesting(port_win, res['var_h'][t0:], dates), use_container_width=True)

    # ---------------------------------------------------------------- 8. OoS
    oos = res.get('oos')
    if cfg['enable_oos']:
        st.markdown("---")
        st.markdown('<p class="sub-header">🔬 8. Validación Out-of-Sample</p>', unsafe_allow_html=True)
        if oos is None:
            st.error(res.get('oos_err') or "No se pudo ejecutar la validación Out-of-Sample.")
        else:
            c1, c2 = st.columns(2)
            c1.info(f"**Entrenamiento (historia previa):** {oos['train_period']} ({oos['n_train']} obs.)")
            c2.info(f"**Prueba (ventana):** {oos['test_period']} ({oos['n_test']} obs.)")
            if oos['n_inf_train'] == 0:
                st.warning("γ no está identificado en el entrenamiento: el DCC-H coincide con el DCC "
                           "estándar en la prueba (γ=0).")
            bh_, bs_ = oos['bt_h'], oos['bt_s']
            tab = pd.DataFrame({
                'Métrica': ['γ estimado (train)', 'Violaciones', 'Tasa observada', 'Tasa esperada',
                            'Kupiec p', 'Christoffersen CC p', 'Pérdida cuantílica media'],
                'DCC Homeostático': [f"{oos['params_h'][2]:.4f}", bh_['violations'],
                                     f"{bh_['violation_rate'] * 100:.2f}%",
                                     f"{bh_['expected_rate'] * 100:.2f}%",
                                     f"{bh_['kupiec_pvalue']:.4f}", _fmt(bh_['christ_cc_pvalue']),
                                     f"{oos['loss_h']:.3e}"],
                'DCC Estándar': ["0", bs_['violations'], f"{bs_['violation_rate'] * 100:.2f}%",
                                 f"{bs_['expected_rate'] * 100:.2f}%", f"{bs_['kupiec_pvalue']:.4f}",
                                 _fmt(bs_['christ_cc_pvalue']), f"{oos['loss_s']:.3e}"]})
            st.dataframe(tab.astype(str), use_container_width=True)
            st.plotly_chart(plot_var_backtesting(oos['port'], oos['var_h'], oos['dates'], oos['var_s'],
                                                 "📊 Out-of-Sample: VaR vs Retornos"),
                            use_container_width=True)
            cal_h = "bien calibrado" if bh_['kupiec_pvalue'] > ALPHA_TEST else "mal calibrado"
            cal_s = "bien calibrado" if bs_['kupiec_pvalue'] > ALPHA_TEST else "mal calibrado"
            if not np.isfinite(oos['dm_p']):
                dm_txt = "El test de Diebold-Mariano no pudo calcularse (pérdidas idénticas o muestra corta)."
            elif oos['dm_p'] < ALPHA_TEST:
                mejor = "DCC-H" if oos['dm_dbar'] < 0 else "DCC estándar"
                dm_txt = (f"Diebold-Mariano: el **{mejor}** tiene una pérdida cuantílica "
                          f"significativamente menor (DM = {oos['dm_stat']:.2f}, p = {oos['dm_p']:.4f}).")
            else:
                dm_txt = (f"Diebold-Mariano: no hay diferencia significativa de precisión entre los modelos "
                          f"(DM = {oos['dm_stat']:.2f}, p = {oos['dm_p']:.4f}).")
            st.info(f"**Cobertura (Kupiec):** DCC-H {cal_h}; DCC estándar {cal_s}.  \n{dm_txt}  \n"
                    f"Días informativos para γ en la prueba: {oos['n_inf_test']}.")

    # ---------------------------------------------------------------- 9. Historial
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 9. Comparativa de las corridas de esta sesión</p>',
                unsafe_allow_html=True)
    st.caption("Construida con corridas reales de esta versión del código. Los valores obtenidos con "
               "versiones anteriores no son comparables (cambiaron la verosimilitud, el umbral EVT y "
               "el tratamiento de la historia previa).")
    hist = st.session_state.setdefault('historial', {})
    if run_key not in hist:
        hist[run_key] = {
            'Período': regime, 'Ventana': f"{dates[0]:%Y-%m-%d} a {dates[-1]:%Y-%m-%d}",
            'Activos': len(tickers),
            'Tensión': 'sorpresa (z)' if cfg['stress_base'] == 'residuos' else 'estado (r)',
            'Umbral': cfg['threshold_window'], 'Método EVT': cfg['gumbel_method'],
            'α Gumbel': cfg['confidence_gumbel'], 'κ': cfg['kappa'],
            'Días H_t': int(H_win.sum()), '% H_t': round(dias_ht_pct, 2),
            'Días informativos γ': lr['n_informativos'], 'γ': round(float(g_h), 4),
            'LR p corregido': lr['p_value'], 'Kupiec p': round(bt['kupiec_pvalue'], 4),
            'Fase': fase_info['fase']}
    boot_p = st.session_state.get(f"boot_p::{run_key}")
    if boot_p is not None:
        hist[run_key]['LR p bootstrap'] = boot_p[0]
        hist[run_key]['B bootstrap'] = boot_p[1]
    hist_df = pd.DataFrame(list(hist.values()))
    if 'LR p bootstrap' in hist_df.columns:
        hist_df['LR p bootstrap'] = hist_df['LR p bootstrap'].map(
            lambda v: '—' if pd.isna(v) else f"{v:.4f}")
        hist_df['B bootstrap'] = hist_df['B bootstrap'].map(lambda v: '—' if pd.isna(v) else int(v))
    st.dataframe(_fmt_df(hist_df, {'LR p corregido': '{:.6f}', 'γ': '{:.4f}', '% H_t': '{:.2f}',
                                   'α Gumbel': '{:.3f}', 'κ': '{:.2f}', 'Kupiec p': '{:.4f}'},
                         na_rep='no identificado'),
                 use_container_width=True)
    st.download_button("📥 Descargar comparativa (CSV)", hist_df.to_csv(index=False),
                       file_name=f"dcc_h_comparativa_{datetime.now():%Y%m%d}.csv", mime="text/csv")
    st.info("**Leyenda:** FASE 1 Estabilidad · FASE 2 Shock exógeno · FASE 3 Saturación · "
            "FASE 4 Transición · SIN EVIDENCIA CONCLUYENTE (γ no significativo con poca información) · "
            "INDETERMINADA (γ no identificado o VaR mal calibrado sin tensión alta)")

    # ---------------------------------------------------------------- 10. Exportar
    st.markdown("---")
    st.markdown('<p class="sub-header">💾 10. Exportar Resultados</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    series_df = pd.DataFrame({'Date': dates, 'Retorno_Portafolio': port_win,
                              'VaR_DCC_H': res['var_h'][t0:], 'Violacion': bt['violation_series'],
                              'H_Indicator': H_win.values, 'Prop_Stressed': prop_win.values})
    c1.download_button("📥 Series temporales de la ventana (CSV)", series_df.to_csv(index=False),
                       file_name=f"dcc_h_series_{datetime.now():%Y%m%d}.csv", mime="text/csv")
    summary = {'Modelo': 'DCC-GARCH Homeostático', 'Activos': len(tickers),
               'Tensión': cfg['stress_base'], 'Ventana umbral': cfg['threshold_window'],
               'Ventana': f"{dates[0]:%Y-%m-%d} a {dates[-1]:%Y-%m-%d}",
               'Historia previa (obs.)': t0, 'Método EVT': cfg['gumbel_method'],
               'Bloque': cfg['block_size'], 'Ventana Gumbel': cfg['gumbel_window'],
               'Confianza Gumbel': cfg['confidence_gumbel'], 'Umbral κ': cfg['kappa'],
               'Días H_t': int(H_win.sum()), '% H_t': f"{dias_ht_pct:.2f}%",
               'Días informativos γ': lr['n_informativos'],
               'a': a_h, 'b': b_h, 'γ': g_h, 'LR': lr['lr_statistic'],
               'LR p corregido': lr['p_value'], 'Decisión LR': lr['decision'],
               'Violaciones VaR': bt['violations'], 'Kupiec p': bt['kupiec_pvalue'],
               'Fase': fase_info['fase']}
    c2.download_button("📥 Resumen del modelo (CSV)", pd.DataFrame(summary, index=['Valor']).to_csv(),
                       file_name=f"dcc_h_resumen_{datetime.now():%Y%m%d}.csv", mime="text/csv")


def render_welcome():
    st.markdown("""
    ### 🎯 Modelo DCC-GARCH Homeostático
    Herramienta de la tesis doctoral **"Dinámica de Corrección Homeostática en Mercados Financieros"**.

    1. **Filtrado GARCH(1,1)** por activo (MLE) con diagnósticos
    2. **Teoría de Valores Extremos**: Gumbel sobre máximos por bloque, umbral causal
    3. **Indicador sistémico H_t** con umbral κ
    4. **DCC-H**: correlación dinámica con término homeostático γ y Q^(S) recursivo
    5. **Test LR** con corrección de frontera y panel de robustez (Benjamini-Hochberg)
    6. **VaR condicional** con backtesting de Kupiec y Christoffersen
    7. **Out-of-Sample**: entrenamiento en la historia previa, prueba en la ventana
    8. **Clasificador de fases** del mercado

    **Hipótesis:** H1 (umbrales de Gumbel vs. normal) · H2 (cambio de correlaciones cuando
    H_t = 1, test LR) · H3 (validez del VaR condicional sistémico).

    <div class="warning-box"><strong>⚠️ Nota académica:</strong> aplicación de investigación.
    Configurá los parámetros en la barra lateral y presioná <em>Ejecutar Modelo</em>.</div>
    """, unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header">🎓 Modelo DCC-GARCH Homeostático con EVT</p>',
                unsafe_allow_html=True)
    st.markdown("**Tesis Doctoral en Economía Financiera** | Detección de Regímenes de "
                "Corrección Homeostática")
    st.markdown("---")

    cfg, regime = sidebar_config()
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True):
        if len(cfg['tickers']) < 2:
            st.sidebar.error("Ingresá al menos 2 tickers.")
        elif cfg['start'] >= cfg['end']:
            st.sidebar.error("La fecha de inicio debe ser anterior a la de fin.")
        else:
            # CORRECCIÓN [5]: la configuración ejecutada se guarda en session_state;
            # las interacciones posteriores no borran los resultados.
            st.session_state['run_cfg'] = cfg
            st.session_state['run_regime'] = regime

    if 'run_cfg' not in st.session_state:
        render_welcome()
        return

    run_cfg = st.session_state['run_cfg']
    run_key = repr(sorted(run_cfg.items()))
    if run_cfg != cfg:
        st.info("ℹ️ Cambiaste parámetros en la barra lateral: los resultados corresponden a la última "
                "ejecución. Presioná *Ejecutar Modelo* para actualizarlos.")
    with st.spinner("Descargando datos y ejecutando el modelo (la primera vez puede tardar)..."):
        res = cached_pipeline(**{k: run_cfg[k] for k in PARAM_KEYS})
    if 'error' in res:
        st.error(f"❌ {res['error']}")
        return
    render_results(res, run_cfg, st.session_state.get('run_regime', ''), run_key)


if __name__ == "__main__":
    main()
