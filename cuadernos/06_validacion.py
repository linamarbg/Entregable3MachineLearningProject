# %% [markdown]
# # 6. Validación estadística
#
# El modelo original se considera superior solo si (guía, sección 6.1): (1) su AUC medio del bucle
# externo supera al de la mejor combinación de cada uno de los siete modelos, (2) al menos tres pruebas
# distintas rechazan la igualdad de desempeño con p ajustado por comparaciones múltiples menor que 0,05
# (0,10 en el MCS) y (3) el tamaño del efecto no es despreciable. Este capítulo aplica ese protocolo y
# reporta también las pruebas que no rechazan.
#
# ## 6.1 Protocolo
#
# * **Universo:** el AJ-KL y las siete mejores combinaciones del capítulo 3 (los benchmarks del curso), con las mismas particiones
#   externas, la misma semilla y el mismo preprocesamiento. Las predicciones de los benchmarks son las del
#   Entregable 2, emparejadas fila por fila con las del AJ-KL. Los dos comparadores recientes del estado del arte
#   (LightGBM y CatBoost, ajustados con el mismo protocolo) se contrastan aparte en la sección 6.8.
# * **Bloques:** el bucle externo cubre cuatro años (k1-k4) y se divide en 48 bloques mensuales; la prueba
#   2025 tiene 52 semanas. Usar meses y semanas en lugar de solo cinco pliegues da potencia a las pruebas
#   no paramétricas sin repetir la búsqueda de hiperparámetros, que tardó 29 horas-CPU para los 140 modelos.
#   Los meses de un mismo pliegue comparten modelo y año, así que no son independientes: la prueba de
#   Friedman se complementa con una permutación restringida al pliegue y las pruebas sobre 2025 usan errores
#   estándar por bloques de semana o bootstrap estacionario.
# * **Ajuste por comparaciones múltiples:** Holm dentro de cada familia de siete comparaciones (AJ-KL frente
#   a cada benchmark); Nemenyi ya controla el error de familia de sus 28 pares; el MCS controla el error de
#   familia por construcción.
#
# ### Fundamento estadístico de las pruebas
#
# Sean $k$ modelos comparados en $N$ bloques (los 48 meses del bucle externo) y $d_b$ la diferencia de una
# métrica entre el modelo original y un benchmark en el bloque $b$. Cada prueba responde una pregunta distinta y
# trata la dependencia entre bloques de una forma distinta; por eso la guía pide al menos tres.
#
# **Friedman** {cite}`friedman1937`. Con $R_j$ el rango medio del modelo $j$ en los $N$ bloques,
#
# $$
# \chi^2_F = \frac{12N}{k(k+1)}\Big[\sum_{j=1}^k R_j^2 - \frac{k(k+1)^2}{4}\Big],
# \qquad
# F_F = \frac{(N-1)\,\chi^2_F}{N(k-1) - \chi^2_F},
# $$ (eq-friedman)
#
# con $F_F \sim F_{k-1,\,(k-1)(N-1)}$ bajo $H_0$: todos los modelos tienen la misma distribución de rangos
# {cite}`iman1980`. La distribución de referencia supone bloques independientes; como los meses de un mismo
# pliegue comparten modelo entrenado, el p-valor que se reporta es el de una **permutación restringida**: en
# cada réplica se permutan las etiquetas de los modelos con la misma permutación en todos los meses de un
# pliegue, y $p$ es la fracción de réplicas con estadístico mayor o igual que el observado.
#
# **Nemenyi** {cite}`nemenyi1963,demsar2006`. Dos modelos difieren si $|R_i - R_j| > \mathrm{CD} =
# q_\alpha\sqrt{k(k+1)/(6N)}$, con $q_\alpha$ el cuantil del rango studentizado dividido por $\sqrt 2$. Controla el
# error de familia de todos los pares y por eso es conservadora.
#
# **Wilcoxon de rangos con signo** {cite}`wilcoxon1945`. $W^+ = \sum_{b:\,d_b > 0}\mathrm{rango}(|d_b|)$; bajo
# $H_0$ (las $d_b$ son simétricas alrededor de cero),
# $z = \{W^+ - N(N+1)/4\} / \sqrt{N(N+1)(2N+1)/24}$ es aproximadamente normal. No supone normalidad, pero sí
# simetría e independencia entre meses (la sección 6.3 revisa ambas). Los p-valores se ajustan con Holm
# {cite}`holm1979`: ordenados $p_{(1)} \le \dots \le p_{(m)}$, $\tilde p_{(i)} = \max_{j \le i}\min\{1, (m-j+1)\,p_{(j)}\}$.
#
# **DeLong** {cite}`delong1988,sun2014fast`. El AUC es un estadístico U,
# $\hat\theta = \frac{1}{n_1 n_0}\sum_{i \in \mathcal P}\sum_{j \in \mathcal N}\psi(s_i, s_j)$ con
# $\psi(a, b) = \mathbf 1[a > b] + \tfrac12\mathbf 1[a = b]$; sus componentes estructurales dan la covarianza de
# $(\hat\theta_1, \hat\theta_2)$ en la misma muestra y $z = (\hat\theta_1 - \hat\theta_2)/\sqrt{\widehat{\mathrm{Var}}(\hat\theta_1 - \hat\theta_2)}$.
# Esa varianza supone filas independientes; en 2025 las filas de una misma semana comparten choques, así que el
# error estándar se reemplaza por el del **bootstrap por bloques de semana** y el intervalo es BCa
# {cite}`efron1987bca`.
#
# **Diebold-Mariano** {cite}`diebold1995` con la corrección de Harvey, Leybourne y Newbold {cite}`harvey1997`.
# Con $d_t$ la diferencia de log-loss semanal entre el benchmark y el original en las $T = 52$ semanas de 2025,
#
# $$
# \mathrm{DM} = \frac{\bar d}{\sqrt{\hat\sigma^2_{\mathrm{HAC}}/T}},
# \qquad
# \mathrm{DM}^\ast = \mathrm{DM}\,\sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}} \sim t_{T-1},
# $$ (eq-dm)
#
# donde $\hat\sigma^2_{\mathrm{HAC}}$ es la varianza de largo plazo de Newey-West {cite}`newey1987hac`, que admite
# autocorrelación de las diferencias, y $h = 1$ es el horizonte. Supone que $d_t$ es estacionaria en covarianza.
#
# **Model Confidence Set** {cite}`hansen2011mcs`. Elimina modelos de uno en uno mientras la prueba de igual
# capacidad predictiva (estadístico de rango sobre las diferencias de pérdida por pares) rechace; el conjunto que
# queda contiene al mejor modelo con probabilidad al menos $1 - \alpha$. Las distribuciones se obtienen con el
# bootstrap estacionario {cite}`politis1994`, que respeta la dependencia temporal.
#
# **t corregida de Nadeau y Bengio** {cite}`nadeau2003`. Con $J$ pliegues,
# $t = \bar d / \sqrt{(1/J + n_{\mathrm{eval}}/n_{\mathrm{entr}})\,s^2} \sim t_{J-1}$: el término
# $n_{\mathrm{eval}}/n_{\mathrm{entr}}$ corrige la subestimación de la varianza por el solapamiento de los
# entrenamientos. Con solo cinco pliegues tiene poca potencia, y por eso es complementaria.
#
# **Tamaños del efecto.** Delta de Cliff $\delta = P(X > Y) - P(X < Y)$ {cite}`cliff1993` con los umbrales de
# {cite:t}`romano2006cliff` (despreciable si $|\delta| < 0{,}147$) y $d_z = \bar d / s_d$ pareada. La regla de
# decisión de la guía exige tres pruebas con $\tilde p < 0{,}05$ (0,10 en el MCS) **y** un efecto no despreciable.
#
# | Prueba | $H_0$ | Distribución de referencia | Cómo trata la dependencia |
# |---|---|---|---|
# | Friedman | Mismos rangos para todos los modelos | $F_{k-1,(k-1)(N-1)}$ y permutación | Permutación restringida al pliegue |
# | Nemenyi | Mismo rango medio por par | Rango studentizado | Ninguna (conservadora) |
# | Wilcoxon-Holm | Mediana de $d_b$ igual a cero | Normal (aprox.) | Se revisa la autocorrelación (6.3) |
# | DeLong | Mismo AUC en 2025 | Normal | Error estándar por bloques de semana |
# | Diebold-Mariano HLN | Misma log-loss esperada | $t_{T-1}$ | Varianza HAC de Newey-West |
# | MCS | Igual capacidad predictiva | Bootstrap | Bootstrap estacionario |
# | Nadeau-Bengio | Misma métrica esperada | $t_{J-1}$ | Corrección de varianza por solapamiento |

# %%
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
from IPython.display import Markdown, display
from scipy import stats

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
pio.renderers.default = "notebook_connected"
import plots as PL

V = "../data/resultados/validacion/"
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
tabla = pd.read_parquet(V + "tabla_pruebas.parquet")
W = pd.read_parquet(V + "wilcoxon_mensual.parquet")
Dl = pd.read_parquet(V + "delong_2025.parquet")
DM = pd.read_parquet(V + "diebold_mariano_2025.parquet")
NB = pd.read_parquet(V + "nadeau_bengio.parquet")
res = json.loads(open(V + "resumen.json").read())
mens = pd.read_parquet(V + "metricas_mensuales_externo.parquet")
corto = lambda c: c.split(" | ")[0]

# %% [markdown]
# ## 6.2 Criterio 1: AUC medio del bucle externo

# %%
sel = pd.read_csv("../data/resultados/seleccion_mejores.csv")
ejul = [joblib.load(f"../data/resultados/ajkl/{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]]
sota = {m: [joblib.load(f"../data/resultados/sota/{m}_{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]] for m in ["LightGBM", "CatBoost"]}
cols = ["auc", "auc_pr", "brier", "log_loss", "ece", "f1", "precision", "recall", "accuracy"]
comp = pd.DataFrame([{"modelo": "AJ-KL (original)", **{c: np.mean([e[c] for e in ejul]) for c in cols},
                      "auc_de": np.std([e["auc"] for e in ejul], ddof=1)}] +
                    [{"modelo": f"{m} (estado del arte)", **{c: np.mean([e[c] for e in v]) for c in cols},
                      "auc_de": np.std([e["auc"] for e in v], ddof=1)} for m, v in sota.items()] +
                    [{"modelo": r.modelo, **{c: getattr(r, f"{c}_media") for c in cols}, "auc_de": r.auc_de}
                     for r in sel.itertuples()]).set_index("modelo")
display(comp.round(4))
mejor = {c: (comp[c].idxmin() if c in ("brier", "log_loss", "ece") else comp[c].idxmax()) for c in cols}
display(Markdown("Modelo con el mejor valor medio externo en cada métrica: " +
                 "; ".join(f"**{c}**: {m}" for c, m in mejor.items()) + "."))

# %% [markdown]
# ### Tabla resumen
#
# La guía (sección 6.3) pide, para el modelo original y los benchmarks, la media ± desviación estándar del bucle
# externo, un intervalo de confianza bootstrap y los tiempos de entrenamiento e inferencia. El intervalo es del AUC
# medio de los 48 meses del bucle externo con el bootstrap estacionario {cite}`politis1994`, que remuestrea bloques
# de meses consecutivos y así respeta su dependencia. Los tiempos son los de los modelos de producción (entrenados
# con las 622 mil filas; k-NN con su muestra de 40 mil): el del AJ-KL suma el ajuste de sus nueve modelos base y el
# del combinador, y su inferencia incluye las predicciones de los nueve modelos base.

# %%
import estadistica as E
import experimento as X
from sklearn.metrics import roc_auc_score

mensual = mens.pivot(index="bloque", columns="modelo", values="auc")
for m_ in ["LightGBM", "CatBoost"]:
    serie = {}
    for pl in ["k1", "k2", "k3", "k4"]:
        _, Vk = X.cargar_pliegue(pl)
        p_ = joblib.load(f"../data/resultados/sota/{m_}_{pl}.joblib")["pred"]
        d_ = pd.DataFrame({"y": Vk["interrupcion_t1"].to_numpy(),
                           "mes": pd.to_datetime(Vk["semana_t1"]).dt.to_period("M").astype(str).to_numpy(),
                           "s": pd.Series(p_["s"], index=p_["fila"]).reindex(Vk["fila"]).to_numpy()})
        serie.update({mes: roc_auc_score(g["y"], g["s"]) for mes, g in d_.groupby("mes")})
    mensual[m_] = pd.Series(serie)
art = pd.read_csv("../data/resultados/artefactos_resumen.csv").set_index("modelo")
base9 = list(sel["modelo"]) + ["LightGBM", "CatBoost"]
t_ajkl = art.loc[base9, "segundos_fit"].sum() + joblib.load("../data/resultados/ajkl/final.joblib")["resumen"]["t_refit"]
equiv = {"AJ-KL (original)": ([c for c in mensual.columns if c.startswith("AJ-KL")][0], "AJ-KL (original)"),
         "LightGBM (estado del arte)": ("LightGBM", "LightGBM"), "CatBoost (estado del arte)": ("CatBoost", "CatBoost")}
equiv.update({r.modelo: (r.combinacion, r.modelo) for r in sel.itertuples()})
filas_r = []
for nombre in comp.index:
    col_m, col_a = equiv[nombre]
    ic = E.ic_bootstrap_estacionario(mensual[col_m].dropna().sort_index().to_numpy(), B=2000)
    filas_r.append({"modelo": nombre, "AUC externo (media ± DE, 4 años)": f"{comp.loc[nombre, 'auc']:.4f} ± {comp.loc[nombre, 'auc_de']:.4f}",
                    "AUC mensual medio": ic["estimado"], "IC 95 % inf.": ic["ic_inf"], "IC 95 % sup.": ic["ic_sup"],
                    "entrenamiento (s)": t_ajkl if nombre.startswith("AJ-KL") else art.loc[col_a, "segundos_fit"],
                    "inferencia (ms / 1000 filas)": art.loc[col_a, "ms_por_1000"]})
resumen_tabla = pd.DataFrame(filas_r).set_index("modelo")
display(resumen_tabla.round(4))
display(Markdown(
    f"Los intervalos se solapan: con 48 meses, el bootstrap del AUC medio de cada modelo tiene un ancho de unas "
    f"{(resumen_tabla['IC 95 % sup.'] - resumen_tabla['IC 95 % inf.']).median():.4f} unidades, mayor que la distancia "
    "entre los cuatro primeros modelos. Por eso la comparación no se hace con intervalos marginales sino con pruebas "
    "pareadas mes a mes (secciones 6.4 a 6.7), que eliminan la variación común entre meses. El AJ-KL es el más lento "
    f"en entrenar ({t_ajkl:.0f} s, porque ajusta los nueve modelos base) y el "
    f"{int((resumen_tabla['inferencia (ms / 1000 filas)'] >= resumen_tabla.loc['AJ-KL (original)', 'inferencia (ms / 1000 filas)']).sum())}.º "
    f"más lento en predecir ({resumen_tabla.loc['AJ-KL (original)', 'inferencia (ms / 1000 filas)']:.0f} ms por cada mil "
    "filas), un costo despreciable para una alerta semanal."))

# %% [markdown]
# ## 6.3 Supuestos de cada prueba

# %%
t = mens.pivot(index="bloque", columns="modelo", values="auc")
ORIG = [c for c in t.columns if c.startswith("AJ-KL")][0]
filas = []
for b in sel["combinacion"]:
    d = (t[ORIG] - t[b]).to_numpy()
    filas.append({"benchmark": corto(b), "media ΔAUC": d.mean(), "asimetría": stats.skew(d),
                  "Shapiro-Wilk p": stats.shapiro(d).pvalue, "autocorrelación rezago 1": pd.Series(d).autocorr(1)})
sup = pd.DataFrame(filas).set_index("benchmark")
display(sup.round(4))
display(Markdown(
    "Wilcoxon supone diferencias simétricas alrededor de la mediana; la asimetría de las diferencias mensuales es "
    f"pequeña (entre {sup['asimetría'].min():.2f} y {sup['asimetría'].max():.2f}). La normalidad no es necesaria para "
    "Wilcoxon, pero sí para la t de Nadeau-Bengio, por eso esta última se usa solo como prueba complementaria. La "
    "autocorrelación de rezago 1 de las diferencias mensuales indica cuánto se parecen meses consecutivos: cuando no "
    "es despreciable, los p de Wilcoxon son algo optimistas y la evidencia debe leerse junto con las pruebas que "
    "respetan la dependencia (permutación por pliegue, DeLong con bloques de semana, Diebold-Mariano con varianza HAC)."))

# %% [markdown]
# ## 6.4 Tabla de pruebas de hipótesis

# %%
mostrar = tabla.copy()
mostrar["conclusión"] = np.where(mostrar["rechaza_h0"], "Rechaza H0", "No rechaza H0")
display(mostrar[["prueba", "comparacion", "h0", "supuestos", "estadistico", "p_crudo", "p_ajustado", "efecto", "conclusión"]]
        .round({"estadistico": 3}))
n_comp = len(tabla)
display(Markdown(f"Se reportan {n_comp} contrastes en total: 1 de Friedman, 7 de Nemenyi, 7 de Wilcoxon, 7 de DeLong, "
                 "2 MCS y 7 de Diebold-Mariano, además de las 7 t de Nadeau-Bengio y las 28 pruebas bayesianas de las "
                 "tablas siguientes."))

# %% [markdown]
# ## 6.5 Diagrama de diferencia crítica

# %%
PL.diagrama_cd(pd.Series({corto(k): v for k, v in res["friedman"]["rangos"].items()}), res["friedman"]["CD"],
               "Friedman-Nemenyi: rango medio del AUC en 48 meses del bucle externo (1 = mejor)").show()

# %% [markdown]
# ## 6.6 Tamaños del efecto e intervalos
#
# Se reportan dos versiones del delta de Cliff porque responden preguntas distintas. El **delta de Cliff
# de dos muestras** {cite}`cliff1993` compara la distribución de los 48 AUC mensuales del AJ-KL con la del
# benchmark como si fueran muestras independientes; la variación entre meses (choques, estacionalidad) es
# mucho mayor que la diferencia entre modelos y lo vuelve pequeño. La **dominancia pareada** es la media de
# signo(ΔAUC) mes a mes: mide qué tan consistente es la ventaja cuando se compara en el mismo mes. La guía
# acepta también la d de Cohen pareada, $d_z$ = media / desviación estándar de las diferencias.

# %%
w = W[W.metrica == "auc"].copy()
w["benchmark"] = w["benchmark"].map(corto)
display(w[["benchmark", "media_dif", "ic_inf", "ic_sup", "meses_gana", "p_holm", "cliff", "mag_cliff",
           "cliff_pareado", "mag_cliff_pareado", "d_cohen", "mag_d", "bayes_p_izq", "bayes_p_rope", "bayes_p_der"]]
        .round(4).set_index("benchmark"))
otros = W[W.metrica != "auc"].assign(benchmark=lambda d: d["benchmark"].map(corto))
display(otros.pivot(index="benchmark", columns="metrica", values="meses_gana").rename(columns=lambda c: f"meses que gana ({c})"))

# %% [markdown]
# La prueba bayesiana de rangos con signo {cite}`benavoli2014signedrank,benavoli2017bayesian` usa una región
# de equivalencia práctica (ROPE) de ±0,01 en AUC, la frontera entre diferencias «triviales» y «pequeñas» de
# la escala del curso. Las columnas `bayes_p_*` son las probabilidades posteriores de que el benchmark sea
# mejor, de que la diferencia caiga en la ROPE y de que el AJ-KL sea mejor.
#
# ## 6.7 Diferencias en la prueba 2025 (intervalos BCa por bloques de semana)

# %%
d = Dl.copy()
d["benchmark"] = d["benchmark"].map(corto)
for m_ in ["auc", "auc_pr", "brier", "log_loss", "ece"]:
    d[m_] = [f"{a:+.4f} [{b:+.4f}; {c:+.4f}]" for a, b, c in zip(d[f"{m_}_dif"], d[f"{m_}_ic_inf"], d[f"{m_}_ic_sup"])]
display(d.set_index("benchmark")[["auc", "auc_pr", "brier", "log_loss", "ece", "p_delong_iid", "p_delong_bloques", "p_holm"]])
display(Markdown("Las diferencias son AJ-KL − benchmark: positivas son mejores en AUC y AUC-PR; negativas son mejores en "
                 "Brier, log-loss y ECE."))

# %%
nb = NB.assign(benchmark=lambda x: x["benchmark"].map(corto)).set_index("benchmark")
display(nb.round(4))
dm = DM.assign(benchmark=lambda x: x["benchmark"].map(corto))
display(dm.pivot(index="benchmark", columns="perdida", values="p_holm").round(4).rename(columns=lambda c: f"p Holm DM ({c})"))

# %% [markdown]
# ## 6.8 Comparación con el estado del arte reciente
#
# LightGBM {cite}`ke2017lightgbm` y CatBoost {cite}`prokhorenkova2018catboost` son los GBDT que la literatura
# reciente sobre datos tabulares sigue señalando como referencia {cite}`shwartzziv2022tabular,borisov2024tabular`.
# No se vieron en clase; se ajustaron con exactamente el mismo protocolo que los benchmarks (mismos pliegues,
# misma muestra $S_k$, TPE con 24 evaluaciones y semilla 42). La tabla resume, para cada uno, el AUC medio del
# bucle externo, la diferencia en 2025 con DeLong por bloques de semana y los meses del bucle externo en que el
# AJ-KL gana.

# %%
import os
if os.path.exists(V + "comparadores_sota.parquet"):
    display(pd.read_parquet(V + "comparadores_sota.parquet").round(5))

# %% [markdown]
# ## 6.9 Comparación con información igual: modelos de producción en 2025
#
# En la validación anidada, el combinador del AJ-KL ve las predicciones fuera de muestra de los bloques
# externos anteriores, más filas etiquetadas que la muestra $S_k$ de 40 mil filas de los benchmarks (aunque
# ninguna posterior a su corte de entrenamiento). La sección 5.8 muestra que reducir ese historial a 40 mil
# filas no cambia el resultado. Una segunda forma de igualar la información es la de producción (Bloque B):
# todos los modelos se reentrenan con las 622.244 filas de entrenamiento, y el AJ-KL combina esos nueve
# artefactos con el combinador del pliegue final, entrenado con predicciones de semanas que los benchmarks
# también usaron para entrenar. Cada artefacto se evalúa una sola vez en las 103.466 filas de 2025.

# %%
if os.path.exists(V + "produccion_2025.parquet"):
    pr = pd.read_parquet(V + "produccion_2025.parquet").assign(benchmark=lambda x: x["benchmark"].map(corto))
    for m_ in ["auc", "auc_pr", "brier", "log_loss", "ece"]:
        pr[m_] = [f"{a:+.4f} [{b:+.4f}; {c:+.4f}]" for a, b, c in zip(pr[f"{m_}_dif"], pr[f"{m_}_ic_inf"], pr[f"{m_}_ic_sup"])]
    display(pr.set_index("benchmark")[["auc_orig", "auc_bench", "auc", "p_delong_holm", "mcnemar_orig_acierta",
                                        "mcnemar_bench_acierta", "p_mcnemar_holm", "brier", "log_loss", "ece"]].round(5))
    art = pd.read_csv("../data/resultados/artefactos_resumen.csv")
    display(art[["modelo", "auc_2025", "auc_pr_2025", "brier_2025", "log_loss_2025", "ece_2025", "f1_2025", "mb", "ms_por_1000"]]
            .sort_values("auc_2025", ascending=False).round(4).set_index("modelo"))

# %% [markdown]
# ## 6.10 Veredicto

# %%
rm = json.loads(open("../data/dashboard/resumen_modelos.json").read())
display(Markdown(rm["veredicto"]))
