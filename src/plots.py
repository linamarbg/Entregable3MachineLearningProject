"""Figuras Plotly reutilizables del dashboard (y del libro). Paleta de cinco colores, plantilla
única, títulos y ejes con unidades."""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

# Paleta de cinco colores tomada del tema (campo, carretera y cosecha): verde montaña para la estructura y
# los títulos, azul río para los datos y los benchmarks, ámbar cosecha para el modelo original, rojo tomate
# para las interrupciones y gris piedra para las referencias.
VERDE, AZUL_RIO, AMBAR, TOMATE, PIEDRA = "#1f5135", "#2b6f8e", "#e39b2d", "#c8423b", "#7c858d"
PALETA = [VERDE, AZUL_RIO, AMBAR, TOMATE, PIEDRA]
AZUL_OSC, AZUL, NARANJA, ROJO, GRIS = PALETA     # nombres por función que usan los cuadernos del libro
ESCALA_VERDE = [[0, "#f7f4ec"], [0.5, "#8fbf9a"], [1, VERDE]]
ESCALA_AMBAR = [[0, "#fbf8f1"], [0.5, "#f1c77f"], [1, "#b56d0e"]]
ESCALA_RIESGO = ["#4e9a6a", "#c9b458", AMBAR, TOMATE]      # tasa de interrupción baja → alta
EVENTOS = [("2020-03-23", "2020-05-31", "Confinamiento"), ("2021-04-28", "2021-06-15", "Paro nacional")]

pio.templates["proyecto"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color="#24302a"),
    paper_bgcolor="white", plot_bgcolor="white", colorway=PALETA, separators=",.",   # coma decimal y punto de miles
    xaxis=dict(gridcolor="#efece4", zerolinecolor="#dcd6c8", linecolor="#cfc8b6", automargin=True),
    yaxis=dict(gridcolor="#efece4", zerolinecolor="#dcd6c8", linecolor="#cfc8b6", automargin=True),
    margin=dict(l=55, r=20, t=70, b=50), hoverlabel=dict(font_size=12),
    title=dict(x=0.01, xanchor="left", font=dict(size=15)),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11)),
))
pio.templates.default = "proyecto"


def color_modelo(nombre: str) -> str:
    return NARANJA if nombre.startswith("AJ-KL") else AZUL if nombre == "XGBoost" else GRIS


def vacia(mensaje: str = "Sin datos para los filtros elegidos") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=GRIS))
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    return fig


def partir(texto: str, ancho: int = 26) -> str:
    """Parte una etiqueta larga en líneas para que quepa en una tarjeta angosta."""
    return "<br>".join(textwrap.wrap(texto, ancho))


# ------------------------------------------------------------------------------- EDA
def serie_tasa(sem: pd.DataFrame) -> go.Figure:
    if sem.empty:
        return vacia()
    s = sem.groupby("semana_t1")[["corredores", "interrupciones"]].sum()
    s["tasa"] = 100 * s["interrupciones"] / s["corredores"]
    s["media_movil"] = s["tasa"].rolling(8, min_periods=1).mean()
    fig = go.Figure()
    for ini, fin, nombre in EVENTOS:
        fig.add_vrect(x0=ini, x1=fin, fillcolor=GRIS, opacity=0.15, line_width=0, annotation_text=nombre,
                      annotation_position="top left", annotation_font_size=10)
    fig.add_scatter(x=s.index, y=s["tasa"], name="Tasa semanal", line=dict(color=AZUL, width=1), opacity=0.6,
                    hovertemplate="%{x|%d-%m-%Y}<br>%{y:.1f} %<extra></extra>")
    fig.add_scatter(x=s.index, y=s["media_movil"], name="Media móvil de 8 semanas", line=dict(color=AZUL_OSC, width=2.5),
                    hovertemplate="%{x|%d-%m-%Y}<br>%{y:.1f} %<extra></extra>")
    fig.update_layout(title="Porcentaje de corredores con interrupción en la semana siguiente", height=360)
    fig.update_yaxes(title="Corredores con interrupción (%)").update_xaxes(title="Semana objetivo (t+1)")
    return fig


def balance_clases(m: pd.DataFrame) -> go.Figure:
    if m.empty:
        return vacia()
    c = m["interrupcion_t1"].map({0: "Sin interrupción", 1: "Interrupción"}).value_counts()
    fig = go.Figure(go.Bar(x=c.index, y=100 * c.values / c.sum(), marker_color=[GRIS if i.startswith("Sin") else ROJO for i in c.index],
                           text=[f"{100 * v / c.sum():.1f}".replace(".", ",") + f" %<br>({v:,} filas)".replace(",", ".") for v in c.values],
                           textposition="outside", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_layout(title="Balance de clases del objetivo", height=330, showlegend=False)
    fig.update_yaxes(title="Corredor-semanas (%)", range=[0, 100]).update_xaxes(title="")
    return fig


def histograma(m: pd.DataFrame, var: str, etiqueta: str, recorte: bool = True) -> go.Figure:
    if m.empty or var not in m:
        return vacia()
    d = m[[var, "interrupcion_t1"]].dropna()
    if recorte and d[var].dtype.kind == "f":
        lo, hi = d[var].quantile([0.01, 0.99])
        d = d[d[var].between(lo, hi)]
    d["Clase"] = d["interrupcion_t1"].map({0: "Sin interrupción", 1: "Interrupción"})
    fig = px.histogram(d, x=var, color="Clase", barmode="overlay", histnorm="percent", nbins=50, opacity=0.65,
                       color_discrete_map={"Sin interrupción": GRIS, "Interrupción": ROJO})
    fig.update_layout(title=f"Distribución de «{etiqueta}» por clase (1 % extremo recortado)", height=340)
    fig.update_xaxes(title=etiqueta).update_yaxes(title="Corredor-semanas de la clase (%)")
    return fig


def bivariado(m: pd.DataFrame, x: str, y: str, color: str, etiquetas: dict) -> go.Figure:
    if m.empty:
        return vacia()
    d = m.sample(min(len(m), 6000), random_state=42).copy()
    if color == "interrupcion_t1":
        d[color] = d[color].map({0: "Sin interrupción", 1: "Interrupción"})
    es_cat_x = d[x].dtype == object or str(d[x].dtype) == "category"
    if es_cat_x:
        top = d[x].value_counts().index[:12]
        d = d[d[x].isin(top)]
        fig = px.box(d, x=x, y=y, color=color if d[color].nunique() <= 6 else None, points=False,
                     color_discrete_sequence=[GRIS, ROJO, AZUL, NARANJA, AZUL_OSC])
    else:
        fig = px.scatter(d, x=x, y=y, color=color, opacity=0.45,
                         color_discrete_sequence=[GRIS, ROJO, AZUL, NARANJA, AZUL_OSC],
                         color_continuous_scale=ESCALA_VERDE)
    fig.update_layout(title=f"«{etiquetas.get(y, y)}» frente a «{etiquetas.get(x, x)}» (6 000 puntos al azar)", height=420)
    fig.update_xaxes(title=etiquetas.get(x, x)).update_yaxes(title=etiquetas.get(y, y))
    return fig


def correlacion(c: pd.DataFrame, etiquetas: dict) -> go.Figure:
    """Mapa de calor de una matriz de correlación ya calculada (filas y columnas con los nombres de las variables)."""
    if c.empty:
        return vacia()
    nombres = [etiquetas.get(v, v) for v in c.columns]
    fig = go.Figure(go.Heatmap(z=c.values, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=[[0, AZUL_RIO], [0.5, "#f7f4ec"], [1, TOMATE]],
                               colorbar=dict(title="ρ"), hovertemplate="%{y}<br>%{x}<br>ρ = %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlación de Spearman entre variables numéricas", height=620)
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9)).update_yaxes(tickfont=dict(size=9), autorange="reversed")
    return fig


def mapa_faltantes(falt: pd.DataFrame, etiquetas: dict) -> go.Figure:
    if falt.empty:
        return vacia()
    f = falt.set_index("variable")
    f = f[(f > 0).any(axis=1)]
    if f.empty:
        return vacia("Ninguna variable del modelo tiene valores faltantes")
    fig = go.Figure(go.Heatmap(z=f.values, x=f.columns, y=[partir(etiquetas.get(v, v)) for v in f.index], colorscale=ESCALA_AMBAR,
                               colorbar=dict(title="%"), hovertemplate="%{y}<br>%{x}: %{z:.2f} % faltante<extra></extra>"))
    fig.update_layout(title="Valores faltantes por variable y año (% de filas)", height=360)
    fig.update_xaxes(title="Año de la semana objetivo")
    return fig


def atipicos(m: pd.DataFrame, var: str, etiqueta: str) -> go.Figure:
    if m.empty:
        return vacia()
    d = m[[var, "anio_t1"]].dropna()
    fig = px.box(d, x="anio_t1", y=var, points="outliers", color_discrete_sequence=[AZUL])
    fig.update_layout(title=f"Atípicos de «{etiqueta}» por año (regla de Tukey: 1,5 × RIC)", height=360)
    fig.update_xaxes(title="Año de la semana objetivo").update_yaxes(title=etiqueta)
    return fig


def descomposicion(serie: pd.Series) -> go.Figure:
    from statsmodels.tsa.seasonal import STL
    if len(serie) < 110:
        return vacia("La serie filtrada es demasiado corta para la descomposición (se requieren 2 años)")
    r = STL(serie, period=52, robust=True).fit()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Serie observada (%)", "Tendencia", "Estacionalidad anual", "Residuo"))
    for i, (comp, col) in enumerate([(serie, AZUL_OSC), (r.trend, AZUL), (r.seasonal, NARANJA), (r.resid, GRIS)], 1):
        fig.add_scatter(x=serie.index, y=comp, line=dict(color=col, width=1.5), showlegend=False, row=i, col=1)
    fig.update_layout(title="Descomposición STL de la tasa semanal de interrupción (periodo de 52 semanas)", height=620)
    return fig


def acf_pacf(serie: pd.Series, rezagos: int = 60) -> go.Figure:
    from statsmodels.tsa.stattools import acf, pacf
    if len(serie) < rezagos + 10:
        return vacia("Serie demasiado corta para la autocorrelación")
    a, p = acf(serie, nlags=rezagos), pacf(serie, nlags=rezagos)
    banda = 1.96 / np.sqrt(len(serie))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Autocorrelación (ACF)", "Autocorrelación parcial (PACF)"))
    for j, v in enumerate([a, p], 1):
        fig.add_bar(x=list(range(len(v))), y=v, marker_color=AZUL, showlegend=False, row=1, col=j)
        fig.add_hline(y=banda, line_dash="dash", line_color=ROJO, row=1, col=j)
        fig.add_hline(y=-banda, line_dash="dash", line_color=ROJO, row=1, col=j)
    fig.update_layout(title="ACF y PACF de la tasa semanal (bandas al 95 %)", height=340)
    fig.update_xaxes(title="Rezago (semanas)")
    return fig


# ------------------------------------------------------------------------------- modelos
def roc_pr(y, scores: dict[str, np.ndarray]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Curva ROC", "Curva precisión-recall"), horizontal_spacing=0.12)
    for nombre, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        paso = max(1, len(fpr) // 2000)
        col = color_modelo(nombre)
        fig.add_scatter(x=fpr[::paso], y=tpr[::paso], name=f"{nombre} (AUC {roc_auc_score(y, s):.4f})",
                        line=dict(color=col, width=2.5 if col != GRIS else 1.5), row=1, col=1)
        paso = max(1, len(rec) // 2000)
        fig.add_scatter(x=rec[::paso], y=prec[::paso], name=nombre, showlegend=False,
                        line=dict(color=col, width=2.5 if col != GRIS else 1.5), row=1, col=2)
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GRIS, dash="dot"), showlegend=False, row=1, col=1)
    fig.add_hline(y=float(np.mean(y)), line_dash="dot", line_color=GRIS, row=1, col=2)
    fig.update_xaxes(title="Tasa de falsos positivos", row=1, col=1).update_yaxes(title="Tasa de verdaderos positivos", row=1, col=1)
    fig.update_xaxes(title="Recall", row=1, col=2).update_yaxes(title="Precisión", row=1, col=2)
    fig.update_layout(title="Discriminación en la prueba 2025 (103 466 corredor-semanas)", height=460,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(t=80, b=110))
    return fig


def calibracion(y, probs: dict[str, np.ndarray], bins: int = 15) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GRIS, dash="dot"), name="Calibración perfecta")
    for nombre, p in probs.items():
        d = pd.DataFrame({"y": y, "p": p})
        d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
        g = d.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).query("n >= 30")
        fig.add_scatter(x=g["p"], y=g["y"], mode="lines+markers", name=nombre, line=dict(color=color_modelo(nombre)),
                        customdata=g["n"], hovertemplate="predicha %{x:.2f}<br>observada %{y:.2f}<br>n = %{customdata}<extra></extra>")
    fig.update_layout(title="Diagrama de calibración en 2025 (15 intervalos de probabilidad)", height=420)
    fig.update_xaxes(title="Probabilidad predicha", range=[0, 1]).update_yaxes(title="Frecuencia observada de interrupción", range=[0, 1])
    return fig


def matriz_confusion(y, p, umbral: float, nombre: str) -> go.Figure:
    cm = confusion_matrix(y, (p >= umbral).astype(int), labels=[0, 1])
    pct = 100 * cm / cm.sum()
    texto = [[f"{cm[i, j]:,}<br>({pct[i, j]:.1f} %)".replace(",", ".") for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=cm, x=["Predice: no", "Predice: sí"], y=["Real: no", "Real: sí"], text=texto,
                               texttemplate="%{text}", colorscale=ESCALA_VERDE, showscale=False,
                               hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
    fig.update_layout(title=f"{nombre}: interrupción en t+1<br><sup>umbral {umbral:.3f} (máximo F1 interno)</sup>", height=380,
                      margin=dict(l=70, r=10, t=80, b=40))
    fig.update_yaxes(autorange="reversed")
    return fig


def barras_shap(phi: pd.Series, etiquetas: dict, titulo: str, n: int = 12,
                eje: str = "Contribución a la probabilidad de interrupción (puntos de probabilidad)", signo: bool = True) -> go.Figure:
    """Barras horizontales de valores de Shapley. Con `signo`, rojo sube la probabilidad y azul la baja."""
    s = phi.reindex(phi.abs().sort_values(ascending=False).index[:n])[::-1]
    colores = [ROJO if v > 0 else AZUL for v in s.values] if signo else [AZUL_OSC] * len(s)
    fig = go.Figure(go.Bar(x=s.values, y=[etiquetas.get(i, i) for i in s.index], orientation="h", marker_color=colores,
                           hovertemplate="%{y}: %{x:+.4f}<extra></extra>" if signo else "%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title=titulo, height=440, showlegend=False, margin=dict(l=10, r=20, t=60, b=50))
    fig.update_xaxes(title=eje)
    return fig


def diagrama_cd(rangos: pd.Series, cd: float, titulo: str) -> go.Figure:
    """Diagrama de diferencia crítica (Demšar, 2006): rango medio de cada modelo y la CD de Nemenyi."""
    r = rangos.sort_values()
    fig = go.Figure()
    y = np.arange(len(r))[::-1]
    fig.add_scatter(x=r.values, y=y, mode="markers+text", text=[f"{n} ({v:.2f})" for n, v in r.items()],
                    textposition="middle right", marker=dict(size=12, color=[color_modelo(n) for n in r.index]),
                    showlegend=False, hovertemplate="%{text}<extra></extra>")
    mejor = r.iloc[0]
    fig.add_shape(type="rect", x0=mejor, x1=mejor + cd, y0=-0.6, y1=len(r) - 0.4, fillcolor=NARANJA, opacity=0.08, line_width=0)
    fig.add_annotation(x=mejor + cd, y=len(r) - 0.4, text=f"CD = {cd:.2f}", showarrow=False, yanchor="bottom", font=dict(color=NARANJA))
    fig.update_layout(title=titulo, height=380)
    fig.update_xaxes(title="Rango medio (1 = mejor)", range=[0.5, len(r) + 2.5]).update_yaxes(visible=False)
    return fig


def convergencia_tpe(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for pl, g in hist.groupby("pliegue"):
        g = g.sort_values("evaluacion")
        fig.add_scatter(x=g["evaluacion"], y=g["auc_interno"].cummax(), mode="lines+markers", name=pl,
                        line=dict(color=NARANJA if pl == "final" else GRIS, width=2.5 if pl == "final" else 1.2))
    fig.update_layout(title="Convergencia de TPE: mejor AUC interno acumulado del modelo original por pliegue externo", height=360)
    fig.update_xaxes(title="Evaluación (presupuesto de 24)").update_yaxes(title="Mejor AUC interno")
    return fig


def aportes_base(pesos, probs, nombres) -> go.Figure:
    """Descomposición exacta del pool del AJ-KL: q = Σ_m w_m p_m (antes de la calibración)."""
    orden = np.argsort(pesos * probs)
    fig = go.Figure(go.Bar(x=(pesos * probs)[orden], y=[nombres[i] for i in orden], orientation="h", marker_color=NARANJA,
                           customdata=np.c_[pesos[orden], probs[orden]],
                           hovertemplate="%{y}<br>peso %{customdata[0]:.3f} × probabilidad %{customdata[1]:.3f} = %{x:.4f}<extra></extra>"))
    fig.update_layout(title=f"Aporte de cada modelo base al pool del AJ-KL (suma = {np.sum(pesos * probs):.3f})", height=380,
                      showlegend=False, margin=dict(l=10, r=20, t=60, b=50))
    fig.update_xaxes(title="Peso del nodo × probabilidad del modelo base")
    return fig


def mapa_pesos(t: pd.DataFrame) -> go.Figure:
    """Mapa de calor de los pesos del combinador por central (filas) y modelo base (columnas)."""
    m = t.set_index("central")
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale=ESCALA_AMBAR, zmin=0,
                               colorbar=dict(title="peso"), hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title="Pesos del pool por central mayorista", height=max(420, 18 * len(m)))
    fig.update_xaxes(tickangle=30).update_yaxes(tickfont=dict(size=10))
    return fig


# ------------------------------------------------------------------------------- mapa de corredores
CLASES_RIESGO = [(0.0, 0.25, "menos de 25 %"), (0.25, 0.50, "25 % a 50 %"), (0.50, 0.75, "50 % a 75 %"), (0.75, 1.01, "75 % o más")]


def mapa_corredores(corr: pd.DataFrame, cent: pd.DataFrame, central: str | None = None, min_semanas: int = 26) -> go.Figure:
    """Mapa de los corredores municipio de origen → central mayorista.

    Cada línea es un corredor con al menos `min_semanas` semanas observadas, coloreado por su tasa histórica de
    interrupción (cuatro clases). Los círculos verdes son las centrales, con área proporcional a las toneladas
    del último año; los puntos son los municipios de origen.
    """
    d = corr[corr["semanas"] >= min_semanas]
    c = cent
    if central:
        d, c = d[d["mercado"] == central], cent[cent["mercado"] == central]
    fig = go.Figure()
    for (lo, hi, etiqueta), color in zip(CLASES_RIESGO, ESCALA_RIESGO):
        g = d[(d["tasa"] >= lo) & (d["tasa"] < hi)]
        if g.empty:
            continue
        n = len(g)
        lon = np.c_[g["lon_o"], g["lon_m"], np.full(n, np.nan)].ravel()
        lat = np.c_[g["lat_o"], g["lat_m"], np.full(n, np.nan)].ravel()
        fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", line=dict(width=1.4 if central else 0.8, color=color),
                                    opacity=0.75 if central else 0.45, name=f"Interrupción {etiqueta} ({n})",
                                    hoverinfo="skip", legendgroup=etiqueta))
        texto = (g["mpio_origen"].str.title() + " (" + g["depto_origen"].str.title() + ") → " + g["mercado"]
                 + "<br>tasa de interrupción " + (100 * g["tasa"]).round(1).astype(str) + " %"
                 + "<br>abastecimiento medio " + (g["t_dia"] / 1000).round(2).astype(str) + " t/día · "
                 + g["dist_km"].round(0).astype(int).astype(str) + " km · " + g["semanas"].astype(str) + " semanas"
                 + np.where(g["pdet"] == 1, "<br>municipio PDET", ""))
        fig.add_trace(go.Scattermap(lon=g["lon_o"], lat=g["lat_o"], mode="markers", marker=dict(size=5, color=color),
                                    text=texto, hovertemplate="%{text}<extra></extra>", showlegend=False, legendgroup=etiqueta))
    ton = c["toneladas_ultimo_anio"].fillna(0)
    fig.add_trace(go.Scattermap(lon=c["lon"], lat=c["lat"], mode="markers", name="Central mayorista (área ∝ toneladas)",
                                marker=dict(size=8 + 34 * np.sqrt(ton / max(ton.max(), 1)), color=VERDE, opacity=0.9),
                                text=c["mercado"] + "<br>" + c["corredores"].astype(str) + " corredores · tasa de interrupción "
                                + (100 * c["tasa"]).round(1).astype(str) + " %<br>" + (ton / 1e3).round(0).astype(int).astype(str)
                                + " mil toneladas en el último año",
                                hovertemplate="%{text}<extra></extra>"))
    centro, zoom = dict(lat=4.3, lon=-73.6), 4.45                   # Colombia completa
    if central and len(d):
        # encuadre de la central y sus orígenes: con teselas de 512 px, el zoom 5 muestra unos 12° de latitud en 560 px
        lats, lons = np.r_[d["lat_o"], c["lat"]], np.r_[d["lon_o"], c["lon"]]
        centro = dict(lat=float((lats.min() + lats.max()) / 2), lon=float((lons.min() + lons.max()) / 2))
        zoom = float(np.clip(np.log2(11 / max(np.ptp(lats), 0.75 * np.ptp(lons), 0.5)) + 5, 4.4, 9))
    fig.update_layout(map=dict(style="carto-positron", center=centro, zoom=zoom),
                      height=560, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="v", x=0.01, y=0.99, xanchor="left", yanchor="top",
                                  bgcolor="rgba(255,255,255,0.85)", bordercolor="#dcd6c8", borderwidth=1, font=dict(size=11)))
    return fig


# ------------------------------------------------------------------------------- sustentación (rúbrica)
def linea_tiempo_pliegues(pl: pd.DataFrame) -> go.Figure:
    """Bloques de la validación anidada: entrenamiento, hueco y validación (o prueba 2025) de cada pliegue."""
    fig = go.Figure()
    filas = list(pl["pliegue"])[::-1]
    for (nombre, color), (ini, fin) in zip([("Entrenamiento", PIEDRA), ("Hueco (8 semanas; 10 antes de 2025)", "#e8e2d4"), ("Validación externa / prueba 2025", AMBAR)],
                                          [("inicio_entrenamiento", "fin_entrenamiento"), ("fin_entrenamiento", "inicio_validacion"),
                                           ("inicio_validacion", "fin_validacion")]):
        dur = (pd.to_datetime(pl[fin]) - pd.to_datetime(pl[ini])).dt.days * 86_400_000
        fig.add_bar(y=pl["pliegue"], x=dur, base=pd.to_datetime(pl[ini]), orientation="h", name=nombre, marker_color=color,
                    hovertemplate="%{y}: %{base|%d-%m-%Y} a " + pd.to_datetime(pl[fin]).dt.strftime("%d-%m-%Y") + "<extra>" + nombre + "</extra>")
    fig.update_layout(barmode="overlay", title="Validación anidada temporal: cada bloque externo valida 52 semanas que el modelo no vio",
                      height=330, bargap=0.35)
    fig.update_yaxes(categoryorder="array", categoryarray=filas, title=None)
    fig.update_xaxes(type="date", title="Semana objetivo")
    return fig


def encogimiento_nodos(nod: pd.DataFrame) -> go.Figure:
    """KL de los pesos de cada nodo hacia su padre frente a su tamaño de muestra efectivo (proposición 5)."""
    fig = go.Figure()
    for nivel, color in [("central", VERDE), ("central × grupo", AZUL_RIO), ("corredor", AMBAR)]:
        g = nod[nod["nivel"] == nivel]
        if len(g):
            fig.add_scatter(x=g["S_r"], y=g["kl_padre"], mode="markers", name=f"{nivel} ({len(g)} nodos)",
                            marker=dict(color=color, size=6 if nivel == "corredor" else 8, opacity=0.55 if nivel == "corredor" else 0.85),
                            customdata=np.c_[g["n"], g["mu"]],
                            hovertemplate="S_r = %{x:,.0f} · n = %{customdata[0]:,}<br>μ_r = %{customdata[1]:.2e}<br>"
                                          "KL(w_r ‖ padre) = %{y:.2e}<extra></extra>")
    fig.update_xaxes(type="log", title="Tamaño de muestra efectivo del nodo, S_r (escala log)")
    fig.update_yaxes(type="log", title="KL(w_r ‖ w_padre) (escala log)")
    fig.update_layout(title="Encogimiento en los datos: los nodos con poca muestra se quedan cerca de su padre", height=420)
    return fig


def complejidad_tiempos(t: pd.DataFrame) -> go.Figure:
    pend = float(np.polyfit(np.log(t["n"]), np.log(t["segundos_fit"]), 1)[0])
    fig = go.Figure(go.Scatter(x=t["n"], y=t["segundos_fit"], mode="lines+markers", line=dict(color=AMBAR, width=2.5),
                               hovertemplate="n = %{x:,}<br>%{y:.2f} s<extra></extra>", name="AJ-KL"))
    fig.update_xaxes(type="log", title="Filas del historial del combinador (escala log)")
    fig.update_yaxes(type="log", title="Segundos de entrenamiento (escala log)")
    fig.update_layout(title=f"Entrenamiento frente a n (pendiente log-log {pend:.2f})".replace(".", ","), height=330, showlegend=False)
    return fig


def sensibilidad(sens: pd.DataFrame, auc_xgb: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Encogimiento λ (escala log)", "Olvido γ", "Profundidad de la jerarquía"],
                        horizontal_spacing=0.08)
    for j, (f, eje) in enumerate([("lam", "log"), ("gamma", "linear"), ("profundidad", "linear")], start=1):
        g = sens[sens["factor"] == f].sort_values("valor")
        fig.add_scatter(x=g["valor"], y=g["auc_media"], mode="lines+markers", line=dict(color=AMBAR, width=2.5),
                        error_y=dict(type="data", symmetric=False, array=g["auc_max"] - g["auc_media"],
                                     arrayminus=g["auc_media"] - g["auc_min"], color=AMBAR, thickness=1),
                        name="AJ-KL (media y rango de k1-k4)", showlegend=j == 1, row=1, col=j)
        fig.add_hline(y=auc_xgb, line_dash="dot", line_color=AZUL_RIO, row=1, col=j)
        fig.update_xaxes(type=eje, row=1, col=j)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=AZUL_RIO, dash="dot"), name="XGBoost (media k1-k4)")
    fig.update_yaxes(title="AUC en el bloque externo", row=1, col=1)
    fig.update_layout(title="Sensibilidad a los hiperparámetros propios (se mueve uno a la vez)", height=420,
                      legend=dict(orientation="h", yanchor="top", y=-0.16, x=0), margin=dict(t=80, b=90))
    return fig


def barras_delta(df: pd.DataFrame, etiqueta: str, valor: str, titulo: str, alto: int = 360) -> go.Figure:
    """Barras horizontales de una diferencia (AJ-KL − referencia): ámbar si gana el AJ-KL, gris si pierde."""
    d = df.sort_values(valor)
    fig = go.Figure(go.Bar(x=d[valor], y=d[etiqueta], orientation="h", marker_color=np.where(d[valor] > 0, AMBAR, PIEDRA),
                           hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#9b9483")
    fig.update_layout(title=titulo, height=alto, showlegend=False)
    fig.update_xaxes(title="ΔAUC (AJ-KL − XGBoost)")
    return fig


def sobreajuste(s: pd.DataFrame) -> go.Figure:
    """AUC de la validación interna (con que se eligieron los hiperparámetros) frente al del bloque externo."""
    g = s.groupby("modelo")[["auc_interno", "auc_externo"]].mean().sort_values("auc_externo")
    fig = go.Figure()
    for m, r in g.iterrows():
        fig.add_scatter(x=[r["auc_interno"], r["auc_externo"]], y=[m, m], mode="lines", line=dict(color="#cfc8b6", width=3),
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=g["auc_interno"], y=g.index, mode="markers", name="Validación interna (selección)",
                    marker=dict(color=PIEDRA, size=10, symbol="circle-open", line=dict(width=2)),
                    hovertemplate="%{y}: interno %{x:.4f}<extra></extra>")
    fig.add_scatter(x=g["auc_externo"], y=g.index, mode="markers", name="Bloque externo (no visto)",
                    marker=dict(color=[NARANJA if m.startswith("AJ-KL") else AZUL_RIO for m in g.index], size=10),
                    hovertemplate="%{y}: externo %{x:.4f}<extra></extra>")
    fig.update_layout(title="Diagnóstico de sobreajuste: AUC interno frente a externo (media de k1-k4)", height=420)
    fig.update_xaxes(title="AUC")
    return fig


def matriz_literatura(m: pd.DataFrame) -> go.Figure:
    g = m.groupby("base", sort=False)[["resultados", "revisados", "relevantes"]].sum()
    fig = go.Figure()
    for col, color, nombre in [("resultados", PIEDRA, "Resultados"), ("revisados", AZUL_RIO, "Revisados"), ("relevantes", AMBAR, "Relevantes")]:
        fig.add_bar(x=g.index, y=g[col], name=nombre, marker_color=color, text=g[col], textposition="outside",
                    hovertemplate="%{x}: %{y} " + nombre.lower() + "<extra></extra>")
    fig.update_layout(title="Búsqueda sistemática (23 de octubre de 2026): registros por base", barmode="group", height=360)
    fig.update_yaxes(title="Registros")
    return fig
