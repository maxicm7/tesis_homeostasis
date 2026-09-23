# ============================================================================
# 🎲 ESTUDIO MONTE CARLO DEL ESTIMADOR DCC-H (Sección 3.9 de la tesis)
# ============================================================================
# Archivo: montecarlo_dcch.py
# Requiere: app_tesis.py (v2.1 o posterior) en la MISMA carpeta. Se importa el
#           estimador directamente de la app, de modo que lo que se valida es
#           exactamente el código que produce los resultados empíricos.
#
# Uso (desde una terminal, en la carpeta de los dos archivos):
#   python montecarlo_dcch.py                      # piloto: γ∈{0,0.03,0.07}, T∈{126,252,1000}, 200 réplicas
#   python montecarlo_dcch.py --reps 20            # prueba rápida (minutos)
#   python montecarlo_dcch.py --T 126 252 504 1000 2000 --reps 1000
#                                                  # estudio completo de la Sección 3.9
#   NOTA: con a=0.02 y b=0.90, la restricción a+b+γ<1 exige γ<0.08; los valores
#   γ∈{0.15, 0.30} de la Sección 3.9.1 son inviables con esos a y b.
#
# Qué responde:
#   - Tamaño del test LR corregido (tasa de rechazo con γ=0; debería ser ≈5%)
#   - Potencia del test (tasa de rechazo con γ>0) según el largo de la ventana T
#   - Sesgo y RMSE de γ̂
#   - Con qué frecuencia γ queda sin identificar (0 días informativos)
#
# DISEÑO DEL PROCESO GENERADOR DE DATOS (DGP)
#   El DGP es el propio DCC-H ("el modelo es verdadero"), como es estándar para
#   validar un estimador:
#   1. Q̄ equicorrelacionada (ρ̄ = 0.3) y Q^(S) equicorrelacionada (ρ_S = 0.7):
#      "flight to correlation" (Sección 3.9.1).
#   2. Q_t sigue la ecuación del DCC-H con (a, b, γ) verdaderos y H_{t−1}.
#   3. Innovaciones t de Student multivariadas ESTANDARIZADAS (varianza 1,
#      ν = 5 por defecto): z_t = R_t^{1/2} · e_t, con E[z_t z_t'] = R_t, tal como
#      exige el modelo. Las colas pesadas y la dependencia en las colas de la t
#      multivariada generan extremos conjuntos de forma endógena, como en los
#      residuos GARCH reales. (Una versión anterior amplificaba los residuos en
#      "crisis"; eso rompe el supuesto de varianza unitaria y el DGP deja de ser
#      un DCC-H: con esos datos γ no se recupera ni con las matrices verdaderas.)
#   4. H_t se calcula con LA MISMA regla que la app (Gumbel sobre máximos por
#      bloque, ventana causal, α y κ): el estrés que "activa" γ es el
#      detectado por el modelo, tal como postula la tesis.
#   5. Cada réplica = historia previa (burn-in, por defecto 756 días = 3 años)
#      + ventana de análisis de largo T. La verosimilitud se evalúa solo sobre
#      la ventana, igual que la opción "Solo la ventana de análisis" de la app.
#
# LIMITACIÓN DEL PILOTO: se simulan directamente los residuos estandarizados
# z_t (Etapa 2). No se simula ni reestima la Etapa 1 (GARCH), por lo que el
# piloto no mide el efecto de la incertidumbre del GARCH. El estudio completo
# puede agregar esa capa.
# ============================================================================

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Silenciar avisos de Streamlit al importar la app fuera de "streamlit run"
logging.getLogger("streamlit").setLevel(logging.ERROR)
os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app_tesis as A  # noqa: E402


# ============================================================================
# ⚙️ CONFIGURACIÓN
# ============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Monte Carlo del estimador DCC-H (Sección 3.9)")
    p.add_argument('--gammas', type=float, nargs='+', default=[0.0, 0.03, 0.07],
                   help="γ verdaderos (con a=0.02, b=0.90 debe ser γ < 0.079)")
    p.add_argument('--T', type=int, nargs='+', default=[126, 252, 1000], help="Largos de la ventana")
    p.add_argument('--reps', type=int, default=200, help="Réplicas por combinación (γ, T)")
    p.add_argument('--hist', type=int, default=756, help="Días de historia previa (burn-in)")
    p.add_argument('--N', type=int, default=6, help="Número de activos")
    p.add_argument('--a', type=float, default=0.02)
    p.add_argument('--b', type=float, default=0.90)
    p.add_argument('--rho-bar', type=float, default=0.30, help="Correlación de Q̄")
    p.add_argument('--rho-s', type=float, default=0.70, help="Correlación de Q^(S)")
    p.add_argument('--nu', type=float, default=5.0,
                   help="Grados de libertad de la t multivariada (0 = normal)")
    p.add_argument('--alpha', type=float, default=0.95, help="Confianza Gumbel (α)")
    p.add_argument('--kappa', type=float, default=0.30, help="Umbral sistémico (κ)")
    p.add_argument('--gumbel-window', type=int, default=252)
    p.add_argument('--block', type=int, default=5)
    p.add_argument('--min-obs', type=int, default=10, help="Mínimo de días de estrés para Q^(S)")
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Procesos en paralelo")
    p.add_argument('--out', type=str, default='resultados_montecarlo', help="Carpeta de salida")
    return p.parse_args(argv)


def equicorr(N, rho):
    return rho * np.ones((N, N)) + (1 - rho) * np.eye(N)


# ============================================================================
# 🎲 SIMULACIÓN DEL DGP
# ============================================================================

def _gumbel_threshold_from_window(window_abs, block, alpha):
    """Umbral de Gumbel sobre máximos por bloque (misma regla que la app)."""
    m = len(window_abs) // block
    if m < 10:
        return np.nan
    maxima = window_abs[len(window_abs) - m * block:].reshape(m, block).max(axis=1)
    loc, scale = A._gumbel_lmoments(maxima)
    if not np.isfinite(loc):
        return np.nan
    return loc - scale * np.log(-np.log(alpha))


def simulate_dcch(rng, T_total, gamma, cfg):
    """
    Simula z_t (T_total x N) y el H_t verdadero del DCC-H con innovaciones t
    multivariadas estandarizadas. H_t se determina con la regla de la app
    sobre los z ya simulados (causal).
    """
    N, a, b = cfg['N'], cfg['a'], cfg['b']
    Q_bar, Q_S = equicorr(N, cfg['rho_bar']), equicorr(N, cfg['rho_s'])
    W, block, alpha, kappa = cfg['gumbel_window'], cfg['block'], cfg['alpha'], cfg['kappa']

    Z = np.zeros((T_total, N))
    H = np.zeros(T_total, dtype=int)
    nu = cfg['nu']
    t_scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
    Q_prev = Q_bar
    for t in range(T_total):
        if t == 0:
            Q_t = Q_bar
        else:
            zz = np.outer(Z[t - 1], Z[t - 1])
            if H[t - 1] == 1:
                Q_t = (1 - a - b - gamma) * Q_bar + a * zz + b * Q_prev + gamma * Q_S
            else:
                Q_t = (1 - a - b) * Q_bar + a * zz + b * Q_prev
        d = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(d, d)

        e = rng.standard_normal(N)
        if nu > 2:   # t multivariada con varianza unitaria (mezcla de escala común)
            e = e * t_scale / np.sqrt(rng.chisquare(nu) / nu)
        Z[t] = np.linalg.cholesky(R_t) @ e

        if t >= W:
            absw = np.abs(Z[t - W:t])
            h = 0
            for j in range(N):
                thr = _gumbel_threshold_from_window(absw[:, j], block, alpha)
                if np.isfinite(thr) and abs(Z[t, j]) > thr:
                    h += 1
            H[t] = int(h / N >= kappa)
        Q_prev = Q_t
    return Z, H


# ============================================================================
# 📐 ESTIMACIÓN (exactamente el pipeline de la app, Etapa 2)
# ============================================================================

def estimate_replica(Z, t0, cfg):
    z_df = pd.DataFrame(Z)
    loc, scale = A.gumbel_rolling_params(z_df, cfg['gumbel_window'], cfg['block'], 'bloques')
    ind = A.stress_indicators(z_df, A.gumbel_thresholds(loc, scale, cfg['alpha']))
    H_hat, _ = A.calculate_systemic_indicator(ind, cfg['kappa'])
    Hv = H_hat.values
    Q_bar_hat = A.ensure_positive_definite(np.corrcoef(Z.T), min_eig=1e-6)
    Qs, active = A.compute_recursive_Qstress(Z, Hv, Q_bar_hat, cfg['min_obs'])
    lr = A.likelihood_ratio_test(Z, Hv, Q_bar_hat, Qs, active, t_start=t0, compute_se=False)
    return Hv, lr


def run_replica(task):
    """Una réplica (se ejecuta en un proceso aparte)."""
    gamma, T, rep, seed, cfg = task
    rng = np.random.default_rng(seed)
    t0 = cfg['hist']
    t_start = time.time()
    row = {'gamma_true': gamma, 'T': T, 'rep': rep, 'seed': seed}
    try:
        Z, H_true = simulate_dcch(rng, t0 + T, gamma, cfg)
        H_hat, lr = estimate_replica(Z, t0, cfg)
        pu = np.asarray(lr['params_unrestricted'], dtype=float)
        row.update({
            'ok': True,
            'H_coincide_DGP': bool(np.array_equal(H_hat, H_true)),
            'dias_Ht_historia': int(H_hat[:t0].sum()),
            'dias_Ht_ventana': int(H_hat[t0:].sum()),
            'dias_informativos': int(lr['n_informativos']),
            'identificado': bool(lr['identificado']),
            'a_hat': pu[0], 'b_hat': pu[1], 'gamma_hat': pu[2],
            'LR': lr['lr_statistic'], 'p_corregido': lr['p_value'],
            'rechaza_H0': bool(lr['identificado'] and lr['p_value'] < 0.05),
            'error': '',
        })
    except Exception as e:  # una réplica fallida no detiene el estudio
        row.update({'ok': False, 'error': repr(e)})
    row['segundos'] = round(time.time() - t_start, 2)
    return row


# ============================================================================
# 📊 RESUMEN
# ============================================================================

def summarize(df):
    out = []
    for (g, T), d in df.groupby(['gamma_true', 'T']):
        ok = d[d['ok']]
        ident = ok[ok['identificado']]
        n, n_id = len(ok), len(ident)
        rej = ok['rechaza_H0'].mean() if n else np.nan
        rej_id = ident['rechaza_H0'].mean() if n_id else np.nan
        err = ident['gamma_hat'] - g
        out.append({
            'gamma_verdadero': g, 'T_ventana': T, 'replicas_ok': n,
            'replicas_fallidas': int((~d['ok']).sum()),
            'pct_identificado': 100 * n_id / n if n else np.nan,
            'dias_Ht_ventana_media': ok['dias_Ht_ventana'].mean(),
            'dias_informativos_media': ok['dias_informativos'].mean(),
            'tasa_rechazo_total': rej,
            'tasa_rechazo_ES_MC': np.sqrt(rej * (1 - rej) / n) if n else np.nan,
            'tasa_rechazo_identificadas': rej_id,
            'gamma_hat_media': ident['gamma_hat'].mean() if n_id else np.nan,
            'gamma_hat_mediana': ident['gamma_hat'].median() if n_id else np.nan,
            'sesgo_gamma': err.mean() if n_id else np.nan,
            'RMSE_gamma': np.sqrt((err ** 2).mean()) if n_id else np.nan,
            'pct_gamma_hat_en_0': 100 * (ident['gamma_hat'] < 1e-4).mean() if n_id else np.nan,
            'sesgo_a': (ok['a_hat'] - float(df.attrs.get('a', np.nan))).mean(),
            'sesgo_b': (ok['b_hat'] - float(df.attrs.get('b', np.nan))).mean(),
            'H_coincide_DGP_pct': 100 * ok['H_coincide_DGP'].mean() if n else np.nan,
        })
    return pd.DataFrame(out)


def print_summary(summ):
    print("\n" + "=" * 100)
    print("RESUMEN DEL MONTE CARLO")
    print("=" * 100)
    for _, r in summ.iterrows():
        tipo = "TAMAÑO (γ=0: debería ser ≈ 5%)" if r['gamma_verdadero'] == 0 else "POTENCIA"
        print(f"\nγ = {r['gamma_verdadero']:.2f} | T = {int(r['T_ventana'])} | réplicas = "
              f"{int(r['replicas_ok'])} (fallidas: {int(r['replicas_fallidas'])})")
        print(f"  γ identificado en {r['pct_identificado']:.1f}% de las réplicas "
              f"(días informativos promedio: {r['dias_informativos_media']:.1f}; "
              f"días H_t=1 en la ventana: {r['dias_Ht_ventana_media']:.1f})")
        print(f"  {tipo}: tasa de rechazo = {100 * r['tasa_rechazo_total']:.1f}% "
              f"(± {196 * r['tasa_rechazo_ES_MC']:.1f} pp al 95%); entre identificadas: "
              f"{100 * r['tasa_rechazo_identificadas']:.1f}%")
        if np.isfinite(r['sesgo_gamma']):
            print(f"  γ̂: media {r['gamma_hat_media']:.4f} | mediana {r['gamma_hat_mediana']:.4f} | "
                  f"sesgo {r['sesgo_gamma']:+.4f} | RMSE {r['RMSE_gamma']:.4f} | "
                  f"γ̂≈0 en {r['pct_gamma_hat_en_0']:.0f}%")
    print("\nNota: 'tasa de rechazo total' cuenta las réplicas sin identificar como no rechazo,")
    print("que es lo que ocurre en la práctica cuando la app informa 'γ no identificado'.")


# ============================================================================
# 🚀 EJECUCIÓN
# ============================================================================

def main(argv=None):
    args = parse_args(argv)
    cfg = {'N': args.N, 'a': args.a, 'b': args.b, 'rho_bar': args.rho_bar, 'rho_s': args.rho_s,
           'nu': args.nu,
           'alpha': args.alpha, 'kappa': args.kappa, 'gumbel_window': args.gumbel_window,
           'block': args.block, 'min_obs': args.min_obs, 'hist': args.hist}
    for g in args.gammas:
        if args.a + args.b + g >= 0.999:
            sys.exit(f"a + b + γ debe ser < 0.999 (γ = {g}).")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'configuracion.json'), 'w', encoding='utf-8') as f:
        json.dump({**vars(args), 'cfg': cfg}, f, indent=2, ensure_ascii=False)

    # Semillas independientes y reproducibles por réplica
    combos = [(g, T) for g in args.gammas for T in args.T]
    seeds = np.random.SeedSequence(args.seed).spawn(len(combos) * args.reps)
    tasks, k = [], 0
    for g, T in combos:
        for rep in range(args.reps):
            tasks.append((g, T, rep, int(seeds[k].generate_state(1)[0]), cfg))
            k += 1

    print(f"Monte Carlo DCC-H: {len(combos)} combinaciones × {args.reps} réplicas = {len(tasks)} "
          f"réplicas, con {args.workers} proceso(s). Resultados en '{args.out}/'.")
    t_ini = time.time()
    rows = []
    step = max(1, len(tasks) // 20)
    if args.workers <= 1:
        iterator = (run_replica(t) for t in tasks)
        for i, row in enumerate(iterator, 1):
            rows.append(row)
            if i % step == 0 or i == len(tasks):
                _progress(i, len(tasks), t_ini)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_replica, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % step == 0 or i == len(tasks):
                    _progress(i, len(tasks), t_ini)

    df = pd.DataFrame(rows).sort_values(['gamma_true', 'T', 'rep']).reset_index(drop=True)
    df.attrs['a'], df.attrs['b'] = args.a, args.b
    df.to_csv(os.path.join(args.out, 'replicas.csv'), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(args.out, 'resumen.csv'), index=False)
    print_summary(summ)
    n_fail = int((~df['ok']).sum())
    if n_fail:
        print(f"\n⚠️ {n_fail} réplica(s) fallaron; ver la columna 'error' en replicas.csv.")
    print(f"\nTiempo total: {(time.time() - t_ini) / 60:.1f} min. Archivos: replicas.csv, resumen.csv, "
          f"configuracion.json")
    return df, summ


def _progress(i, n, t_ini):
    el = time.time() - t_ini
    eta = el / i * (n - i)
    print(f"  {i}/{n} réplicas ({100 * i / n:.0f}%) — transcurrido {el / 60:.1f} min, "
          f"restante ≈ {eta / 60:.1f} min", flush=True)


if __name__ == "__main__":   # necesario para el paralelismo en Windows
    main()
