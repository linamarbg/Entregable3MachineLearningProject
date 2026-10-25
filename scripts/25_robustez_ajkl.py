"""Robustez y sensibilidad del AJ-KL (guía, cuadro 1: sensibilidad a los hiperparámetros propios, estabilidad
ante múltiples semillas, comportamiento bajo ruido, desbalance y menor tamaño de muestra).

En cada pliegue externo k1-k4 se parte de los hiperparámetros que eligió TPE en scripts/22_ajkl.py y del mismo
historial de predicciones fuera de muestra; se cambia una sola cosa a la vez y se evalúa en el mismo bloque
externo V_k. Los nueve modelos base no se reentrenan: las perturbaciones afectan solo al combinador, que es la
contribución original. Como referencia se reporta, con la misma perturbación, el apilamiento global
(profundidad 1, el caso particular sin jerarquía) y, sin perturbación, el mejor benchmark (XGBoost).

Experimentos:
  sensibilidad  λ ∈ {0,01; 0,1; 1; 10; 100; 1000}, γ ∈ {0,98; 0,99; 0,995; 1} y profundidad ∈ {1, 2, 3, 4},
                uno a la vez, con los demás en el valor elegido
  semillas      la búsqueda TPE completa repetida con las semillas 43-46 (la 42 es la del modelo reportado);
                el combinador es determinista, así que la semilla solo cambia los hiperparámetros elegidos
  ruido         etiquetas del historial invertidas al azar (5, 10 y 20 %) y ruido gaussiano en el logit de las
                probabilidades base del historial (σ = 0,1; 0,25; 0,5); la evaluación es en V_k sin ruido
  desbalance    submuestreo de los positivos del historial hasta tasas de 15 % y 8 % (la del panel es 28 %)
  muestra       submuestras aleatorias del historial de 5, 10, 20 y 40 mil filas
Uso: python scripts/25_robustez_ajkl.py <pliegue> <experimento>
"""
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402
import optimizacion as O  # noqa: E402
from original_model import ApilamientoJerarquicoKL  # noqa: E402

RES = RAIZ / "data" / "resultados"
SAL = RES / "robustez_ajkl"
EXPERIMENTOS = ("sensibilidad", "semillas", "ruido", "desbalance", "muestra")
LAMS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMAS = (0.98, 0.99, 0.995, 1.0)
PROFUNDIDADES = (1, 2, 3, 4)
SEMILLAS = (43, 44, 45, 46)
VOLTEOS = (0.05, 0.10, 0.20)
SIGMAS = (0.1, 0.25, 0.5)
TASAS = (0.15, 0.08)
TAMANOS = (5_000, 10_000, 20_000, 40_000)


def importar_22():
    spec = importlib.util.spec_from_file_location("m22", RAIZ / "scripts" / "22_ajkl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def preparar(pliegue):
    """Historial, bloque externo y configuración elegida del pliegue (los mismos de scripts/22_ajkl.py)."""
    m22 = importar_22()
    sel = m22.seleccion()
    cols = m22.columnas_base(sel)
    G = m22.codigos_jerarquia(D.cargar_panel())
    h, part = m22.historial(pliegue, sel)
    V = m22.prob_bloque(pliegue, sel)
    xgb = sel.loc[sel["modelo"] == "XGBoost", "combinacion"].iloc[0]
    return {"m22": m22, "cols": cols, "G": G, "h": h, "part": part,
            "params": dict(joblib.load(RES / "ajkl" / f"{pliegue}.joblib")["resumen"]["params"]),
            "Xh": m22.matriz(h, cols, G), "yh": h[C.OBJ_CLF].to_numpy(), "th": m22.semanas(h["semana_t1"]),
            "Xv": m22.matriz(V, cols, G), "yv": V[C.OBJ_CLF].to_numpy(), "p_xgb": V[xgb].to_numpy(float)}


def metricas(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"auc": roc_auc_score(y, p), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}


def evaluar(ctx, Xh=None, yh=None, th=None, **cambios):
    """Ajusta el combinador con el historial dado (por defecto, el original) y lo evalúa en V_k."""
    Xh = ctx["Xh"] if Xh is None else Xh
    yh = ctx["yh"] if yh is None else yh
    th = ctx["th"] if th is None else th
    t0 = time.perf_counter()
    m = ApilamientoJerarquicoKL(n_modelos=len(ctx["cols"]), **{**ctx["params"], **cambios}).fit(Xh, yh, tiempos=th)
    seg = time.perf_counter() - t0
    return {**metricas(ctx["yv"], m.predict_proba(ctx["Xv"])[:, 1]), "segundos": seg, "filas": int(len(yh))}


def par(ctx, factor, valor, Xh=None, yh=None, th=None):
    """AJ-KL y apilamiento global con la misma perturbación del historial."""
    return [{"factor": factor, "valor": valor, "modelo": "AJ-KL", **evaluar(ctx, Xh, yh, th)},
            {"factor": factor, "valor": valor, "modelo": "Apilamiento global",
             **evaluar(ctx, Xh, yh, th, profundidad=1)}]


def sensibilidad(ctx):
    filas, p = [], ctx["params"]
    for nombre, valores in (("lam", LAMS), ("gamma", GAMMAS), ("profundidad", PROFUNDIDADES)):
        for v in sorted(set(valores) | {p[nombre]}):
            filas.append({"factor": nombre, "valor": float(v), "modelo": "AJ-KL", "elegido": bool(v == p[nombre]),
                          **evaluar(ctx, **{nombre: v})})
    return filas


def semillas(ctx):
    m22, filas = ctx["m22"], []
    for s in SEMILLAS:
        C.fijar_semillas(s)
        obj = m22.ObjetivoAJKL(ctx["h"], ctx["part"], ctx["cols"], ctx["G"])
        hist = pd.DataFrame(O.bayes_tpe(obj, m22.ESPACIO, C.PRESUPUESTO, s)["historial"])
        i = int(hist["puntaje"].to_numpy().argmax())
        mejor = {**dict(hist.loc[i, "params"]), "calibrar": bool(hist.loc[i, "calibrar"])}
        filas.append({"factor": "semilla", "valor": float(s), "modelo": "AJ-KL", **mejor,
                      **evaluar(ctx, **mejor)})
    return filas


def ruido(ctx):
    rng = np.random.default_rng(C.SEMILLA)
    M = len(ctx["cols"])
    filas = []
    for rho in VOLTEOS:
        voltear = rng.random(len(ctx["yh"])) < rho
        filas += par(ctx, "etiquetas invertidas", rho, yh=np.where(voltear, 1 - ctx["yh"], ctx["yh"]))
    for sigma in SIGMAS:
        Xh = ctx["Xh"].copy()
        Xh[:, :M] = np.clip(expit(logit(Xh[:, :M]) + rng.normal(0, sigma, (len(Xh), M))), 1e-4, 1 - 1e-4)
        filas += par(ctx, "ruido en el logit", sigma, Xh=Xh)
    return filas


def desbalance(ctx):
    rng = np.random.default_rng(C.SEMILLA)
    y = ctx["yh"]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    filas = []
    for tasa in TASAS:
        n_pos = int(round(tasa / (1 - tasa) * len(neg)))
        idx = np.sort(np.r_[neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)])
        filas += par(ctx, "tasa de positivos", tasa, ctx["Xh"][idx], y[idx], ctx["th"][idx])
    return filas


def muestra(ctx):
    rng = np.random.default_rng(C.SEMILLA)
    filas = []
    for n in TAMANOS:
        if n >= len(ctx["yh"]):
            continue
        idx = np.sort(rng.choice(len(ctx["yh"]), n, replace=False))
        filas += par(ctx, "filas del historial", n, ctx["Xh"][idx], ctx["yh"][idx], ctx["th"][idx])
    return filas


def ejecutar(pliegue, experimento):
    archivo = SAL / f"{experimento}_{pliegue}.parquet"
    if archivo.exists():
        return pd.read_parquet(archivo)
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    ctx = preparar(pliegue)
    filas = globals()[experimento](ctx)
    base = {"factor": "referencia", "valor": np.nan}
    filas += [{**base, "modelo": "AJ-KL", "elegido": True, **evaluar(ctx)},
              {**base, "modelo": "XGBoost (sin perturbar)", **metricas(ctx["yv"], ctx["p_xgb"])}]
    out = pd.DataFrame(filas).assign(pliegue=pliegue, experimento=experimento)
    SAL.mkdir(parents=True, exist_ok=True)
    out.to_parquet(archivo, index=False)
    return out


if __name__ == "__main__":
    pl, ex = sys.argv[1], sys.argv[2]
    if ex not in EXPERIMENTOS:
        raise SystemExit(f"experimento debe ser uno de {EXPERIMENTOS}")
    t0 = time.perf_counter()
    r = ejecutar(pl, ex)
    print(pl, ex, f"{time.perf_counter() - t0:.0f}s")
    print(r[["factor", "valor", "modelo", "auc", "log_loss", "filas"]].round(4).to_string(index=False))
