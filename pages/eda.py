"""Sección 2: Análisis exploratorio con filtros globales que actualizan todos los gráficos."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme

import data as DT
import plots as PL
import sustentacion as S

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Corredores de abastecimiento", order=1)

ETQ = dict(zip(DT.diccionario()["variable"], DT.diccionario()["descripcion"])) if len(DT.diccionario()) else {}
NUMERICAS = ["ratio_t", "ratio_t_1", "log_base8", "cv8", "pend8_rel", "activas8", "activas26", "n_envios8",
             "n_alimentos8", "antig_sem", "dist_km", "frac_caida_mercado_t", "frac_caida_nacional_t",
             "frac_caida_mpio_t", "frac_caida_vec_t", "tasa_int_vec26", "tasa_int_mpio26", "festivos_t1",
             "sh_verduras", "sh_frutas", "sh_tuberculos", "sh_granos", "ratio_t1"]
CATEGORICAS = ["mercado", "depto_origen", "grupo_dom", "mismo_depto", "local", "pdet_origen", "interrupcion_t1"]


def opciones(cols):
    return [{"label": ETQ.get(c, c), "value": c} for c in cols]


def lectura(id_):
    return html.Div(id=id_, className="lectura")


def tarjeta(titulo, cuerpo):
    return dbc.Card([dbc.CardHeader(titulo), dbc.CardBody(cuerpo)])


def layout(**_):
    m = DT.muestra()
    if m.empty:
        return dbc.Alert("Faltan los datos del EDA (ejecute scripts/17_datos_dashboard.py).", color="warning")
    centrales = sorted(m["mercado"].dropna().unique())
    deptos = sorted(m["depto_origen"].dropna().unique())
    d_ini, d_fin = m["semana_t1"].min().date(), m["semana_t1"].max().date()
    km_max = KM_MAX
    return html.Div([
        html.Div(dbc.Row([
            dbc.Col([
                html.Div("Sección 2 · Análisis exploratorio", className="sobretitulo"),
                html.H2("¿Cómo se comportan los corredores?"),
                html.P("Los filtros de la barra actualizan todos los gráficos de la página. Las distribuciones usan una muestra "
                       "estratificada de 80 mil corredor-semanas (10 mil por año); la serie semanal usa el panel completo.",
                       className="mb-0"),
            ], lg=7, className="d-flex flex-column justify-content-center"),
            dbc.Col(html.Img(src=dash.get_asset_url("img/banner_eda.svg"),
                             alt="Ilustración: una lupa revisa barras y una serie semanal sobre surcos de cultivo"),
                    lg=5, className="d-none d-lg-block"),
        ]), className="cabecera"),
        html.Div(dbc.Row([
            dbc.Col([dbc.Label("Central mayorista", html_for="f-central"),
                     dcc.Dropdown(id="f-central", options=centrales, multi=True, placeholder="Todas")], xs=12, md=6, xl=3),
            dbc.Col([dbc.Label("Departamento de origen", html_for="f-depto"),
                     dcc.Dropdown(id="f-depto", options=deptos, multi=True, placeholder="Todos")], xs=12, md=6, xl=3),
            dbc.Col([dbc.Label("Semanas objetivo", html_for="f-fechas"),
                     dcc.DatePickerRange(id="f-fechas", min_date_allowed=d_ini, max_date_allowed=d_fin,
                                         start_date=d_ini, end_date=d_fin, display_format="DD/MM/YYYY")], xs=12, md=6, xl=3),
            dbc.Col([dbc.Label("Distancia origen-central (km)", html_for="f-km"),
                     dcc.RangeSlider(id="f-km", min=0, max=km_max, step=50, value=[0, km_max],
                                     marks={int(v): (f"{int(v)}+" if v == km_max else f"{int(v)}") for v in np.linspace(0, km_max, 5)},
                                     tooltip={"placement": "bottom"})], xs=12, md=6, xl=3),
        ], className="g-2"), className="filtros"),
        dbc.Alert(id="eda-aviso", color="warning", is_open=False),

        dbc.Row([
            dbc.Col(tarjeta("Serie de interrupciones", [dcc.Loading(dcc.Graph(id="g-serie")), lectura("l-serie")]), lg=8),
            dbc.Col(tarjeta("Objetivo y balance de clases", [dcc.Graph(id="g-balance"), lectura("l-balance")]), lg=4),
        ]),
        tarjeta("Estadística descriptiva de las variables numéricas (muestra filtrada)", [
            dash_table.DataTable(id="t-desc", sort_action="native", page_size=8, style_table={"overflowX": "auto"},
                                 style_cell=S.CELDA, style_header=S.ENCABEZADO_TABLA),
            lectura("l-desc")]),
        dbc.Row([
            dbc.Col(tarjeta("Distribuciones univariadas", [
                dcc.Dropdown(id="s-hist", options=opciones(NUMERICAS), value="ratio_t", clearable=False,
                             className="mb-2"),
                dcc.Graph(id="g-hist"), lectura("l-hist")]), lg=6),
            dbc.Col(tarjeta("Explorador bivariado", [
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id="s-x", options=opciones(NUMERICAS + CATEGORICAS[:3]), value="ratio_t", clearable=False), xs=4),
                    dbc.Col(dcc.Dropdown(id="s-y", options=opciones(NUMERICAS), value="ratio_t1", clearable=False), xs=4),
                    dbc.Col(dcc.Dropdown(id="s-color", options=opciones(CATEGORICAS + ["dist_km"]), value="interrupcion_t1",
                                         clearable=False), xs=4),
                ], className="g-1 mb-2"),
                dcc.Graph(id="g-bi"), lectura("l-bi")]), lg=6),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Matriz de correlación", [dcc.Graph(id="g-corr"), lectura("l-corr")]), lg=7),
            dbc.Col([
                tarjeta("Valores faltantes", [dcc.Graph(id="g-falt"), lectura("l-falt")]),
                tarjeta("Valores atípicos", [
                    dcc.Dropdown(id="s-atip", options=opciones(NUMERICAS), value="cv8", clearable=False, className="mb-2"),
                    dcc.Graph(id="g-atip"), lectura("l-atip")]),
            ], lg=5),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Serie de tiempo: descomposición", [dcc.Loading(dcc.Graph(id="g-stl")), lectura("l-stl")]), lg=7),
            dbc.Col(tarjeta("Autocorrelación y estacionariedad", [dcc.Graph(id="g-acf"), html.Div(id="t-estac"), lectura("l-acf")]), lg=5),
        ]),
        dbc.Card([dbc.CardHeader("Preprocesamiento: transformaciones y prevención de fuga de información"), dbc.CardBody(dbc.Row([
            dbc.Col([html.H6("Transformaciones y por qué", className="fw-bold"), S.md(S.TRANSFORMACIONES)], lg=6),
            dbc.Col([html.H6("Cinco reglas contra la fuga de información", className="fw-bold"), S.md(S.FUGA)], lg=6),
        ]))], id="fuga"),
    ])


KM_MAX = float(np.ceil(DT.muestra()["dist_km"].quantile(0.995) / 100) * 100) if len(DT.muestra()) else 1e9


def filtrar(central, depto, ini, fin, km):
    m, s = DT.muestra(), DT.semanal()
    km = km or [0, KM_MAX]
    hasta = np.inf if km[1] >= KM_MAX else km[1]          # el extremo superior del deslizador es abierto
    cond_m = m["semana_t1"].between(pd.Timestamp(ini), pd.Timestamp(fin)) & m["dist_km"].between(km[0], hasta)
    cond_s = s["semana_t1"].between(pd.Timestamp(ini), pd.Timestamp(fin))
    if central:
        cond_m &= m["mercado"].isin(central)
        cond_s &= s["mercado"].isin(central)
    if depto:
        cond_m &= m["depto_origen"].isin(depto)
        cond_s &= s["depto_origen"].isin(depto)
    return m[cond_m], s[cond_s]


FILTROS = [Input("f-central", "value"), Input("f-depto", "value"), Input("f-fechas", "start_date"),
           Input("f-fechas", "end_date"), Input("f-km", "value")]


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


@callback(Output("g-serie", "figure"), Output("l-serie", "children"), Output("g-balance", "figure"),
          Output("l-balance", "children"), Output("eda-aviso", "children"), Output("eda-aviso", "is_open"), *FILTROS)
def resumen(central, depto, ini, fin, km):
    m, s = filtrar(central, depto, ini, fin, km)
    if len(m) < 50 or s.empty:
        f = PL.vacia()
        return f, "", f, "", "La selección tiene muy pocas observaciones; amplíe los filtros.", True
    tasa = s.groupby("semana_t1")[["corredores", "interrupciones"]].sum()
    t = tasa["interrupciones"] / tasa["corredores"]
    pico = t.idxmax()
    l1 = (f"En la selección, en promedio {pct(t.mean())} de los corredores se interrumpe cada semana; el máximo fue "
          f"{pct(t.max())} la semana del {pico:%d-%m-%Y}. La tasa sube en los choques comunes (confinamiento de 2020, "
          f"paro de 2021), por eso la validación respeta el orden temporal.")
    r = m["interrupcion_t1"].mean()
    filas = f"{len(m):,}".replace(",", ".")
    razon = f"{(1 - r) / r:.1f}".replace(".", ",")
    l2 = (f"{pct(r)} de las {filas} corredor-semanas de la muestra filtrada son interrupciones: el problema es "
          f"desbalanceado (1 a {razon}), por eso se evalúa con AUC, AUC-PR y F1 y no con la exactitud.")
    return PL.serie_tasa(s), l1, PL.balance_clases(m), l2, "", False


@callback(Output("g-hist", "figure"), Output("l-hist", "children"), Input("s-hist", "value"), *FILTROS)
def univariado(var, central, depto, ini, fin, km):
    m, _ = filtrar(central, depto, ini, fin, km)
    if len(m) < 50:
        return PL.vacia(), ""
    g = m.groupby("interrupcion_t1")[var].median()
    texto = (f"Mediana de «{ETQ.get(var, var)}»: {g.get(1, np.nan):.3g} en las interrupciones frente a "
             f"{g.get(0, np.nan):.3g} sin interrupción. Cuanto más se separan las dos distribuciones, más información "
             f"aporta la variable para anticipar el evento.")
    return PL.histograma(m, var, ETQ.get(var, var)), texto


@callback(Output("g-bi", "figure"), Output("l-bi", "children"), Input("s-x", "value"), Input("s-y", "value"),
          Input("s-color", "value"), *FILTROS)
def bivariado(x, y, color, central, depto, ini, fin, km):
    m, _ = filtrar(central, depto, ini, fin, km)
    if len(m) < 50:
        return PL.vacia(), ""
    texto = ""
    if m[x].dtype.kind in "fi" and m[y].dtype.kind in "fi":
        rho = m[[x, y]].corr(method="spearman").iloc[0, 1]
        texto = (f"Correlación de Spearman entre las dos variables: ρ = {rho:.2f}. "
                 + ("La relación monótona es fuerte." if abs(rho) > 0.5 else "La relación monótona es débil o moderada; "
                    "puede haber relaciones no lineales que los árboles capturan mejor."))
    return PL.bivariado(m, x, y, color, ETQ), texto


@callback(Output("g-corr", "figure"), Output("l-corr", "children"), Output("g-falt", "figure"), Output("l-falt", "children"),
          *FILTROS)
def correlaciones(central, depto, ini, fin, km):
    m, _ = filtrar(central, depto, ini, fin, km)
    if len(m) < 50:
        return PL.vacia(), "", PL.vacia(), ""
    vars_ = [v for v in NUMERICAS if m[v].nunique() > 1]
    # Spearman como Pearson sobre los rangos de cada variable: con faltantes difiere de la versión por pares de
    # pandas en menos de 0,002, pero es unas 20 veces más rápida (en el plan gratuito de Render tardaba 30 s)
    c = m[vars_].rank().corr()
    v = c.to_numpy(copy=True)
    np.fill_diagonal(v, 0)
    i, j = np.unravel_index(np.nanargmax(np.abs(v)), v.shape)
    obj = c["ratio_t1"].drop("ratio_t1").abs().sort_values(ascending=False)
    l1 = (f"El par más correlacionado es «{ETQ.get(vars_[i])}» y «{ETQ.get(vars_[j])}» (ρ = {v[i, j]:.2f}). "
          f"Con la razón de t+1 se asocian sobre todo «{ETQ.get(obj.index[0])}» y «{ETQ.get(obj.index[1])}», aunque "
          f"ninguna supera |ρ| = {obj.iloc[0]:.2f}: el abastecimiento de la semana siguiente es difícil de anticipar.")
    cols = [v for v in DT.diccionario()["variable"] if v in m.columns and v not in ("interrupcion_t1", "ratio_t1")]
    falt = m[cols + ["anio_t1"]]
    tabla = falt[cols].isna().groupby(falt["anio_t1"]).mean().T * 100
    tabla.columns = tabla.columns.astype(str)
    tabla = tabla.reset_index(names="variable")
    total = falt[cols].isna().mean()
    l2 = ("Ninguna variable tiene faltantes en la selección." if total.max() == 0 else
          f"La variable con más faltantes es «{ETQ.get(total.idxmax(), total.idxmax())}» ({total.max() * 100:.2f} %); se "
          "imputa con la mediana del entrenamiento dentro de la tubería, sin usar información de validación.")
    return PL.correlacion(c, ETQ), l1, PL.mapa_faltantes(tabla, ETQ), l2


FMT3 = Format(precision=3, scheme=Scheme.decimal_or_exponent, decimal_delimiter=",", group_delimiter=".", group=Group.yes)


@callback(Output("t-desc", "data"), Output("t-desc", "columns"), Output("l-desc", "children"), *FILTROS)
def descriptiva(central, depto, ini, fin, km):
    m, _ = filtrar(central, depto, ini, fin, km)
    if len(m) < 50:
        return [], [], ""
    vars_ = [v for v in NUMERICAS if v in m.columns]
    d = m[vars_].describe().T
    y = m["interrupcion_t1"]
    med1, med0 = m.loc[y == 1, vars_].median(), m.loc[y == 0, vars_].median()
    t = pd.DataFrame({"Variable": [ETQ.get(v, v) for v in vars_], "n": d["count"].astype(int).to_numpy(),
                      "Media": d["mean"].to_numpy(), "Desv. est.": d["std"].to_numpy(), "Mín.": d["min"].to_numpy(),
                      "P25": d["25%"].to_numpy(), "Mediana": d["50%"].to_numpy(), "P75": d["75%"].to_numpy(), "Máx.": d["max"].to_numpy(),
                      "% faltantes": 100 * m[vars_].isna().mean().to_numpy(),
                      "Mediana con interrupción": med1.to_numpy(), "Mediana sin interrupción": med0.to_numpy()})
    entero = Format(precision=0, scheme=Scheme.fixed, group=Group.yes, group_delimiter=".", decimal_delimiter=",")
    columnas = [{"name": "Variable", "id": "Variable"}] + [{"name": c, "id": c, "type": "numeric", "format": entero if c == "n" else FMT3}
                                                           for c in t.columns[1:]]
    asim = m[[v for v in vars_ if v != "ratio_t1"]].skew().sort_values(ascending=False)
    iqr = (d["75%"] - d["25%"]).replace(0, np.nan)
    sep = ((med1 - med0).abs() / iqr).drop("ratio_t1", errors="ignore").dropna().sort_values(ascending=False)
    v = sep.index[0]
    c = lambda x, f: format(x, f).replace(".", ",")
    texto = (f"Las variables más asimétricas son «{ETQ.get(asim.index[0])}», «{ETQ.get(asim.index[1])}» y «{ETQ.get(asim.index[2])}» "
             f"(asimetría {c(asim.iloc[0], '.1f')}; {c(asim.iloc[1], '.1f')} y {c(asim.iloc[2], '.1f')}): colas largas a la derecha que "
             f"justifican el log(1 + x) y la winsorización del preprocesamiento. La que más separa las clases es «{ETQ.get(v)}»: "
             f"mediana {c(med1[v], '.3g')} con interrupción frente a {c(med0[v], '.3g')} sin ella.")
    return t.to_dict("records"), columnas, texto


@callback(Output("g-atip", "figure"), Output("l-atip", "children"), Input("s-atip", "value"), *FILTROS)
def atipicos(var, central, depto, ini, fin, km):
    m, _ = filtrar(central, depto, ini, fin, km)
    if len(m) < 50:
        return PL.vacia(), ""
    v = m[var].dropna()
    q1, q3 = v.quantile([0.25, 0.75])
    frac = ((v < q1 - 1.5 * (q3 - q1)) | (v > q3 + 1.5 * (q3 - q1))).mean()
    texto = (f"{pct(frac)} de los valores de «{ETQ.get(var, var)}» están fuera de 1,5 × RIC. Son valores reales "
             "(corredores muy grandes o muy volátiles), no errores: los árboles los toleran y los modelos lineales los "
             "reciben winsorizados al 1 % y 99 %.")
    return PL.atipicos(m, var, ETQ.get(var, var)), texto


@callback(Output("g-stl", "figure"), Output("l-stl", "children"), Output("g-acf", "figure"), Output("t-estac", "children"),
          Output("l-acf", "children"), *FILTROS)
def serie_temporal(central, depto, ini, fin, km):
    from statsmodels.tsa.stattools import adfuller, kpss
    _, s = filtrar(central, depto, ini, fin, km)
    if s.empty:
        return PL.vacia(), "", PL.vacia(), "", ""
    t = s.groupby("semana_t1")[["corredores", "interrupciones"]].sum()
    serie = (100 * t["interrupciones"] / t["corredores"]).asfreq("7D").interpolate()
    if len(serie) < 110:
        f = PL.vacia("Se requieren al menos dos años de semanas para la descomposición y las pruebas")
        return f, "", f, "", ""
    from statsmodels.tsa.seasonal import STL
    r = STL(serie, period=52, robust=True).fit()
    fuerza = max(0.0, 1 - r.resid.var() / (r.resid + r.seasonal).var())
    adf = adfuller(serie, autolag="AIC")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(serie, regression="c", nlags="auto")
    tabla = dbc.Table([html.Thead(html.Tr([html.Th("Prueba"), html.Th("H0"), html.Th("Estadístico"), html.Th("p")])),
                       html.Tbody([html.Tr([html.Td("ADF"), html.Td("raíz unitaria"), html.Td(f"{adf[0]:.2f}"), html.Td(f"{adf[1]:.4f}")]),
                                   html.Tr([html.Td("KPSS"), html.Td("estacionaria"), html.Td(f"{kp[0]:.2f}"), html.Td(f"{kp[1]:.3f}")])])],
                      size="sm", className="tabla mt-2", bordered=False)
    concl = ("estacionaria en nivel (ADF rechaza y KPSS no rechaza)" if adf[1] < 0.05 and kp[1] >= 0.05 else
             "con indicios de no estacionariedad o cambios de nivel (las dos pruebas no coinciden)")
    l1 = (f"La fuerza de la estacionalidad anual es {fuerza:.2f} (0 = nula, 1 = total): la tasa tiene un ciclo anual "
          "moderado (festivos de fin de año, cosechas) y una tendencia que cambia con los choques.")
    l2 = (f"La serie filtrada es {concl}. La autocorrelación decae lentamente: semanas consecutivas se parecen, lo que "
          "justifica el bootstrap por bloques de semana en la comparación estadística.")
    return PL.descomposicion(serie), l1, PL.acf_pacf(serie), tabla, l2
