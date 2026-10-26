"""Datos de las pestañas de fundamentos, literatura, diseño, validación, errores y conclusiones (data/dashboard/).

  pliegues.parquet              fechas de entrenamiento, hueco y validación de cada bloque de la validación anidada
  hiperparametros_ajkl.parquet  hiperparámetros que eligió TPE en cada pliegue, historial y nodos del combinador
  encogimiento_nodos.parquet    por nodo del combinador final: filas, tamaño efectivo S_r, μ_r = λ/S_r, KL hacia el
                                padre, brecha de Frank-Wolfe e iteraciones (evidencia de las proposiciones 3 y 5)
  complejidad_ajkl.parquet      tiempo de entrenamiento y de predicción del combinador frente a n (sección 5.7)
  robustez_sensibilidad.parquet, robustez_semillas.parquet, robustez_perturbaciones.parquet   sección 5.13
  errores_subgrupo.parquet      AUC del AJ-KL y de XGBoost en 2025 por distancia, escala y PDET (sección 5.12)
  errores_central.parquet       ΔAUC del AJ-KL frente a XGBoost en 2025 por central mayorista
  externos.parquet              resumen de los tres conjuntos de la UCI (sección 5.11)
  sobreajuste.parquet           AUC de la validación interna frente al del bloque externo, por modelo y pliegue
  literatura_matriz.parquet     matriz de novedad por base y cadena (capítulo 4)
  sustentacion.json             cifras sueltas: búsqueda complementaria en WoS y referencias de la robustez
Lee resultados ya calculados; solo reajusta el combinador del pliegue final para medir tiempos.
Uso: python scripts/28_sustentacion_dashboard.py
"""
import glob
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import datos as D  # noqa: E402
import original_model as OM  # noqa: E402

RES = RAIZ / "data" / "resultados"
SAL = RAIZ / "data" / "dashboard"
PLIEGUES = ["k1", "k2", "k3", "k4", "final"]
NIVELES = ["central", "central × grupo", "corredor"]


def cargar_m22():
    spec = importlib.util.spec_from_file_location("m22", RAIZ / "scripts" / "22_ajkl.py")
    m22 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m22)
    return m22


def pliegues(panel):
    train, test = D.particion(panel)
    filas = []
    for k, (tr, va) in enumerate(D.pliegues_externos(train), start=1):
        s_tr, s_va = train["semana_t1"].iloc[tr], train["semana_t1"].iloc[va]
        filas.append({"pliegue": f"k{k}", "inicio_entrenamiento": s_tr.min(), "fin_entrenamiento": s_tr.max(),
                      "inicio_validacion": s_va.min(), "fin_validacion": s_va.max(), "tipo": "Bloque externo"})
    filas.append({"pliegue": "final", "inicio_entrenamiento": train["semana_t1"].min(), "fin_entrenamiento": train["semana_t1"].max(),
                  "inicio_validacion": test["semana_t1"].min(), "fin_validacion": test["semana_t1"].max(), "tipo": "Prueba 2025"})
    pd.DataFrame(filas).to_parquet(SAL / "pliegues.parquet", index=False)


def hiperparametros(res):
    filas = []
    for pl, r in res.items():
        s = r["resumen"]
        filas.append({"pliegue": pl, "λ": s["params"]["lam"], "γ": s["params"]["gamma"], "profundidad": s["params"]["profundidad"],
                      "calibrar": bool(s["params"].get("calibrar", False)), "semivida (semanas)": np.log(2) / np.log(1 / s["params"]["gamma"])
                      if s["params"]["gamma"] < 1 else np.inf, "filas del historial": s["filas_historial"],
                      "nodos por nivel": str(s["nodos"]), "segundos (búsqueda + reajuste)": round(s["t_total"])})
    pd.DataFrame(filas).to_parquet(SAL / "hiperparametros_ajkl.parquet", index=False)


def encogimiento_y_complejidad(m22, res, panel):
    """Recorre los nodos del combinador final (como la sección 5.7) y mide tiempos frente a n."""
    sel = m22.seleccion()
    cols = m22.columnas_base(sel)
    G = m22.codigos_jerarquia(panel)
    h, _ = m22.historial("final", sel)
    Xh, yh, th = m22.matriz(h, cols, G), h["interrupcion_t1"].to_numpy(float), m22.semanas(h["semana_t1"])
    mod = res["final"]["modelo"]
    P = Xh[:, :9]
    a = mod.gamma ** (th.max() - th)
    kl = lambda w, v: float(np.sum(w * np.log(w / v)))
    u = np.full(9, 1 / 9)
    filas = [{"nivel": "raíz", "nodo": "raíz", "n": len(yh), "S_r": a.sum(), "mu": mod.lam / a.sum(), "kl_padre": kl(mod.raiz_, u),
              "brecha": OM.brecha_fw(mod.raiz_, P, yh, a, u, mod.lam / a.sum()), "iteraciones": len(mod.historial_[("raiz",)]) - 1}]
    for l, tabla in enumerate(mod.pesos_):
        cod = Xh[:, 9 + l].astype(int)
        orden = np.argsort(cod, kind="stable")
        unicos, inicios = np.unique(cod[orden], return_index=True)
        grupos = {int(c): orden[i:f] for c, i, f in zip(unicos, inicios, np.r_[inicios[1:], len(orden)])}
        for c, w in tabla.items():
            idx = grupos[c]
            padre = mod.raiz_ if l == 0 else mod.pesos_[l - 1].get(int(Xh[idx[0], 9 + l - 1]), mod.raiz_)
            S = a[idx].sum()
            filas.append({"nivel": NIVELES[l], "nodo": str(c), "n": len(idx), "S_r": S, "mu": mod.lam / S, "kl_padre": kl(w, padre),
                          "brecha": OM.brecha_fw(w, P[idx], yh[idx], a[idx], padre, mod.lam / S),
                          "iteraciones": len(mod.historial_[(l, c)]) - 1})
    pd.DataFrame(filas).to_parquet(SAL / "encogimiento_nodos.parquet", index=False)

    filas_t, rng = [], np.random.default_rng(0)
    for n in [10_000, 30_000, 100_000, len(yh)]:
        idx = np.sort(rng.choice(len(yh), min(n, len(yh)), replace=False))
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            m = OM.ApilamientoJerarquicoKL(**mod.get_params()).fit(Xh[idx], yh[idx], tiempos=th[idx])
            ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        m.predict_proba(Xh[:1000])
        filas_t.append({"n": len(idx), "segundos_fit": float(np.median(ts)), "ms_por_1000_pred": 1000 * (time.perf_counter() - t0)})
    pd.DataFrame(filas_t).to_parquet(SAL / "complejidad_ajkl.parquet", index=False)


def robustez(res):
    """Resúmenes de la sección 5.13 con los mismos cálculos del libro."""
    rob = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RES / "robustez_ajkl" / "*.parquet")))], ignore_index=True)
    r_aj = rob[(rob.factor == "referencia") & (rob.modelo == "AJ-KL")].drop_duplicates("pliegue").set_index("pliegue")
    ref, ref_ll = r_aj["auc"], r_aj["log_loss"]
    xgb = rob[rob.modelo == "XGBoost (sin perturbar)"].drop_duplicates("pliegue").set_index("pliegue")["auc"]
    malla = {"lam": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "gamma": [0.98, 0.99, 0.995, 1.0], "profundidad": [1, 2, 3, 4]}
    sens = rob[(rob.experimento == "sensibilidad") & (rob.factor != "referencia")].copy()
    sens = sens[[any(np.isclose(v, m) for m in malla[f]) for f, v in zip(sens.factor, sens.valor)]]
    sens["dauc"] = sens["auc"] - sens["pliegue"].map(ref)
    sens["dll"] = sens["log_loss"] - sens["pliegue"].map(ref_ll)
    sens["sobre_xgb"] = sens["auc"] > sens["pliegue"].map(xgb)
    (sens.groupby(["factor", "valor"]).agg(auc_media=("auc", "mean"), auc_min=("auc", "min"), auc_max=("auc", "max"),
                                           dauc_medio=("dauc", "mean"), dauc_peor=("dauc", "min"), dll_medio=("dll", "mean"),
                                           pliegues_sobre_xgb=("sobre_xgb", "sum"))
     .reset_index().to_parquet(SAL / "robustez_sensibilidad.parquet", index=False))
    sem = rob[(rob.experimento == "semillas") & (rob.factor == "semilla")][["pliegue", "valor", "auc", "lam", "gamma", "profundidad"]]
    base42 = pd.DataFrame([{"pliegue": pl, "valor": 42.0, "auc": ref[pl], **{k: res[pl]["resumen"]["params"][k] for k in ("lam", "gamma", "profundidad")}}
                           for pl in ref.index])
    sem = pd.concat([base42, sem], ignore_index=True)
    sem["semilla"] = sem["valor"].astype(int)
    sem["xgb"] = sem["pliegue"].map(xgb)
    sem.drop(columns="valor").to_parquet(SAL / "robustez_semillas.parquet", index=False)
    otros = rob[rob.experimento.isin(["ruido", "desbalance", "muestra"]) & (rob.factor != "referencia")].copy()
    otros["d_ref"] = otros["auc"] - otros["pliegue"].map(ref)
    otros["d_xgb"] = otros["auc"] - otros["pliegue"].map(xgb)
    (otros.groupby(["experimento", "factor", "valor", "modelo"]).agg(auc_media=("auc", "mean"), dauc_sin_perturbar=("d_ref", "mean"),
                                                                    dauc_xgb=("d_xgb", "mean"), log_loss=("log_loss", "mean"),
                                                                    pliegues=("auc", "size"))
     .reset_index().to_parquet(SAL / "robustez_perturbaciones.parquet", index=False))
    return {"auc_ajkl_ref": float(ref.mean()), "auc_xgb_ref": float(xgb.mean())}


def errores(res, panel):
    """Sección 5.12: subgrupos y centrales de la prueba 2025 (pliegue final de la validación anidada)."""
    pru = pd.read_parquet(RES / "pred_prueba_clf.parquet")
    p = res["final"]["pred"]
    pru["AJ-KL"] = pd.Series(p["s"], index=p["fila"]).reindex(pru["fila"]).to_numpy()
    pru["XGBoost"] = pru["XGBoost | Sin balanceo | Random Search"]
    info = panel.set_index("fila").loc[pru["fila"], ["dist_km", "pdet_origen", "mercado", "log_base8"]].reset_index(drop=True)
    pru = pd.concat([pru.reset_index(drop=True), info], axis=1)
    pru["distancia"] = pd.qcut(pru["dist_km"], 4, labels=["Q1 corta", "Q2", "Q3", "Q4 larga"])
    pru["escala"] = pd.qcut(pru["log_base8"], 4, labels=["Q1 pequeño", "Q2", "Q3", "Q4 grande"])
    pru["PDET"] = pru["pdet_origen"].map({0: "No PDET", 1: "PDET"})
    auc = lambda g, c: roc_auc_score(g["interrupcion_t1"], g[c])
    filas = [{"grupo": grupo, "nivel": str(nivel), "n": len(g), "tasa": g["interrupcion_t1"].mean(),
              "auc_ajkl": auc(g, "AJ-KL"), "auc_xgb": auc(g, "XGBoost")}
             for grupo in ["distancia", "escala", "PDET"] for nivel, g in pru.groupby(grupo, observed=True)]
    sub = pd.DataFrame(filas)
    sub["dauc"] = sub["auc_ajkl"] - sub["auc_xgb"]
    sub.to_parquet(SAL / "errores_subgrupo.parquet", index=False)
    cent = [{"central": c, "n": len(g), "tasa": g["interrupcion_t1"].mean(), "dauc": auc(g, "AJ-KL") - auc(g, "XGBoost")}
            for c, g in pru.groupby("mercado", observed=True) if g["interrupcion_t1"].nunique() > 1]
    pd.DataFrame(cent).sort_values("dauc").to_parquet(SAL / "errores_central.parquet", index=False)


def externos():
    ext = pd.read_parquet(RES / "externos" / "comparacion_v21.parquet")
    bases = ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Regresión logística", "SVM", "KNN", "Decision Tree", "Naive Bayes"]
    filas = []
    for nombre, g in ext.groupby("nombre"):
        mejor = g[bases].mean().idxmax()
        filas.append({"conjunto": nombre, "auc_ajkl": g["AJ-KL"].mean(), "auc_global": g["AJ-KL global (profundidad 1)"].mean(),
                      "auc_xgb": g["XGBoost"].mean(), "mejor_base": mejor, "auc_mejor_base": g[mejor].mean(),
                      "gana_xgb": int((g["AJ-KL"] > g["XGBoost"]).sum()), "gana_mejor_base": int((g["AJ-KL"] > g[mejor]).sum()),
                      "pliegues": len(g), "dll_xgb": (g["AJ-KL | log_loss"] - g["XGBoost | log_loss"]).mean()})
    pd.DataFrame(filas).to_parquet(SAL / "externos.parquet", index=False)


def sobreajuste():
    """AUC interno (validación interna con que se eligieron los hiperparámetros) frente al del bloque externo."""
    t = pd.read_csv(RES / "tabla_maestra.csv")
    t = t[(t.tarea == "clf") & t.pliegue.isin(["k1", "k2", "k3", "k4"])]
    sel = pd.read_csv(RES / "seleccion_mejores.csv")
    elegidas = set(sel["combinacion"]) | set(t.loc[t.grupo != "Entregable 2 (140 combinaciones)", "combinacion"])
    t = t[t.combinacion.isin(elegidas)][["modelo", "combinacion", "pliegue", "puntaje_interno", "auc", "filas_entrenamiento"]]
    t.rename(columns={"puntaje_interno": "auc_interno", "auc": "auc_externo"}).to_parquet(SAL / "sobreajuste.parquet", index=False)


def literatura():
    m = pd.read_csv(RAIZ / "data" / "revision" / "matriz_novedad_equipo.csv")
    m[["base", "cadena", "consulta", "resultados", "revisados", "relevantes", "observacion", "fecha"]].to_parquet(
        SAL / "literatura_matriz.parquet", index=False)
    w = pd.read_csv(RAIZ / "data" / "revision" / "matriz_novedad_wos.csv")
    return {"wos_cadenas": len(w), "wos_resultados": int(w["resultados"].sum()), "wos_revisados": int(w["revisados"].sum()),
            "wos_relevantes": int(w["relevantes"].sum())}


def main():
    panel = D.cargar_panel()
    res = {pl: joblib.load(RES / "ajkl" / f"{pl}.joblib") for pl in PLIEGUES}
    pliegues(panel)
    hiperparametros(res)
    encogimiento_y_complejidad(cargar_m22(), res, panel)
    cifras = robustez(res)
    errores(res, panel)
    externos()
    sobreajuste()
    cifras.update(literatura())
    (SAL / "sustentacion.json").write_text(json.dumps(cifras, ensure_ascii=False, indent=2))
    print(json.dumps(cifras, ensure_ascii=False))


if __name__ == "__main__":
    main()
