"""Sección 1: Contexto (narrativa, formulación del problema, mapa de los corredores y recorrido de la sustentación)."""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html

import data as DT
import plots as PL
import sustentacion as S

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Corredores de abastecimiento", order=0)


def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kpi(valor, etiqueta, icono, tono=""):
    return dbc.Card(dbc.CardBody([
        html.Div(html.I(className=f"bi bi-{icono}", **{"aria-hidden": "true"}), className="icono"),
        html.Div([html.Div(etiqueta, className="kpi-etiqueta"), html.Div(valor, className="kpi-valor")]),
    ]), className=f"kpi h-100 {tono}")


def encabezado(icono, texto):
    return dbc.CardHeader([html.I(className=f"bi bi-{icono}", **{"aria-hidden": "true"}), texto])


def paso(numero, titulo, texto):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(numero, className="badge rounded-pill mb-2"),
        html.H6(titulo, className="fw-bold mb-1"),
        html.Small(texto),
    ]), className="h-100"), xs=12, sm=6, lg=3, className="mb-2 paso")


def layout(**_):
    R = DT.resumen_eda()
    if not R:
        return dbc.Alert("Faltan los datos del dashboard (ejecute scripts/17_datos_dashboard.py).", color="warning")
    ton = R.get("toneladas_millones_por_anio", {})
    t25 = ton.get("2025", ton.get(2025, 0))
    t18 = ton.get("2018", ton.get(2018, 0))
    dic = DT.diccionario()
    corr, _ = DT.mapa()
    centrales = sorted(corr["mercado"].unique()) if len(corr) else []
    return html.Div([
        html.Div(dbc.Row([
            dbc.Col([
                html.Div("Proyecto final de Machine Learning · Entregable 3", className="sobretitulo"),
                html.H1("¿Qué corredores de abastecimiento de alimentos se interrumpirán la próxima semana?"),
                html.P("Cada semana, miles de cargas de alimentos viajan desde 925 municipios hasta las 32 principales "
                       "centrales mayoristas de Colombia. Cuando un corredor (municipio de origen → central) cae a menos de "
                       "la mitad de su abastecimiento habitual, la central pierde una fuente y los precios y la "
                       "disponibilidad local se resienten. Este tablero anticipa esas interrupciones con la información "
                       "disponible al cierre de cada semana.", className="mb-0"),
                html.Div([
                    html.Div([html.Div(f"{num(t25, 2)} M t", className="cifra"),
                              html.Div(f"de alimentos llegaron a las 32 centrales en 2025 ({num(100 * (t25 / t18 - 1), 0)} % más que en 2018)."),
                              html.Div("Fuente: DANE, SIPSA-Abastecimiento, microdatos 2018-2025 (cálculo propio).", className="fuente mt-1")],
                             className="cifra-caja"),
                    html.Div([html.Div(f"{num(100 * R['tasa_interrupcion'], 1)} %", className="cifra ambar"),
                              html.Div("de las corredor-semanas terminan en interrupción la semana siguiente.")],
                             className="cifra-caja"),
                ], className="cifras"),
            ], lg=6),
            dbc.Col(html.Img(src=dash.get_asset_url("img/hero_corredor.svg"), className="ilustracion",
                             alt="Ilustración: un camión con frutas y verduras recorre una carretera entre montañas hacia una "
                                 "central mayorista"),
                    lg=6, className="mt-3 mt-lg-0 d-flex align-items-center"),
        ]), className="hero"),
        html.Img(src=dash.get_asset_url("img/productos.svg"), className="banda-productos",
                 alt="Tomate, plátano, papa, aguacate, zanahoria, mazorca, naranja y cebolla"),

        dbc.Row([dbc.Col(kpi(v, e, i, t), xs=6, md=4, xl=2, className="mb-2") for v, e, i, t in [
            (num(R["registros_crudos"]), "Registros crudos del SIPSA", "database", ""),
            (num(R["filas"]), "Corredor-semanas del panel", "table", "kpi-ambar"),
            (num(R["corredores"]), "Corredores", "signpost-split", ""),
            (num(R["municipios"]), "Municipios de origen", "geo-alt", "kpi-ambar"),
            (num(R["centrales"]), "Centrales mayoristas", "shop", ""),
            (num(R["semanas"]), "Semanas (2018-2025)", "calendar-week", "kpi-ambar"),
        ]]),

        S.recorrido(),

        dbc.Card([encabezado("geo-alt-fill", "El mapa de los corredores"), dbc.CardBody([
            dbc.Row([
                dbc.Col([dbc.Label("Central mayorista", html_for="mapa-central"),
                         dcc.Dropdown(id="mapa-central", options=centrales, placeholder="Todas las centrales")], md=5),
                dbc.Col(html.Div(id="mapa-resumen", className="leyenda-mapa mt-md-4 pt-md-2"), md=7),
            ], className="g-2 mb-2"),
            dcc.Loading(dcc.Graph(id="mapa-corredores", config={"scrollZoom": True, "displaylogo": False})),
            html.Div("Cada línea une un municipio de origen con la central que abastece; el color es la fracción de sus semanas "
                     "que terminó en interrupción. Entre los 3.151 corredores regulares (26 semanas o más), los de menos de 100 km "
                     "se interrumpen el 31 % de las semanas y los más largos cerca del 43 %; el cuartil de menor volumen, el 54 %, "
                     "y el de mayor volumen, el 16 % (ρ de Spearman −0,68). Los corredores que salen de municipios PDET se "
                     "interrumpen más (42,6 % frente a 37,5 %).", className="lectura"),
        ])]),

        dbc.Row([
            dbc.Col(dbc.Card([encabezado("lightbulb", "Por qué importa"), dbc.CardBody([
                html.P("Las centrales mayoristas concentran la distribución de alimentos frescos hacia tiendas, plazas y "
                       "restaurantes. Una interrupción en un corredor obliga a buscar otros orígenes, encarece el "
                       "transporte y puede trasladarse a los precios al consumidor."),
                html.P("Una alerta con una semana de anticipación permite a la central, a los comerciantes y a las "
                       "autoridades de abastecimiento priorizar el seguimiento de los corredores en riesgo y activar "
                       "orígenes alternativos.", className="mb-0"),
            ])]), lg=6),
            dbc.Col(dbc.Card([encabezado("question-circle", "Pregunta de investigación y objetivos"), dbc.CardBody([
                html.P(html.Strong("¿Se puede anticipar, con la información disponible al cierre de la semana t, qué "
                                   "corredores tendrán en t+1 un abastecimiento menor que la mitad de su promedio de 8 "
                                   "semanas?")),
                html.Ul([
                    html.Li("Llevar a producción la mejor combinación (balanceo × optimizador) de cada uno de los 7 modelos del Entregable 2."),
                    html.Li("Proponer un modelo original (AJ-KL) que combine los modelos con pesos distintos por central, grupo de alimentos y corredor."),
                    html.Li("Verificar estadísticamente si el modelo original supera a los benchmarks."),
                ], className="mb-0"),
            ])]), lg=6),
        ]),

        dbc.Card([encabezado("calculator", "Formulación matemática del problema"), dbc.CardBody(S.md(S.FORMULACION_PROBLEMA))],
                 id="formulacion"),

        dbc.Card([encabezado("database", "Datos"), dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P([html.Strong("Fuente: "), "microdatos del componente de abastecimiento del SIPSA (DANE), que "
                            "registran cada carga que ingresa a las centrales mayoristas."]),
                    html.P([html.Strong("Unidad de análisis: "), "corredor-semana (municipio de origen → central)."]),
                    html.P([html.Strong("Periodo: "), f"semanas objetivo del {R['inicio']} al {R['fin']}; entrenamiento "
                            f"hasta el {R['fin_entrenamiento']} ({num(R['filas_entrenamiento'])} filas) y prueba con todo "
                            f"2025 ({num(R['filas_prueba'])} filas)."]),
                    html.P([html.Strong("Tarea: "), "clasificación binaria. Objetivo ", html.Code("interrupcion_t1"),
                            " = 1 si la razón entre el abastecimiento de t+1 y su promedio de 8 semanas es menor que 0,5."]),
                    html.P([html.Strong("Variables: "), f"{R['variables']} predictores (historia del corredor, composición de "
                            "la carga, calendario, contexto de la central y del país, y vecindad espacial)."], className="mb-0"),
                ], lg=5),
                dbc.Col([
                    html.Div("Diccionario de variables (se puede filtrar y ordenar)", className="fw-semibold mb-1"),
                    dash_table.DataTable(
                        data=dic.to_dict("records"), columns=[{"name": n, "id": c} for c, n in
                                                              [("variable", "Variable"), ("descripcion", "Descripción"),
                                                               ("tipo", "Tipo"), ("rango", "Rango")]],
                        page_size=8, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "textAlign": "left", "padding": "4px"},
                        style_header={"backgroundColor": "#e9f2ea", "fontWeight": 700, "color": "#1f5135"}),
                ], lg=7),
            ]),
        ])]),

        dbc.Card([encabezado("diagram-3", "Metodología"), dbc.CardBody(dbc.Row([
            paso("1", "ETL y panel", "15,5 millones de registros agregados a 744.921 corredor-semanas; variables con información hasta t."),
            paso("2", "EDA", "Desbalance (28 % de interrupciones), estacionalidad, choques regionales y dependencia espacial."),
            paso("3", "140 combinaciones", "7 modelos × 4 balanceos × 4 optimizadores (clasificación) con validación anidada temporal."),
            paso("4", "Selección", "Mejor combinación por modelo según el AUC externo; reentrenamiento y serialización."),
            paso("5", "Modelo original", "AJ-KL: apilamiento jerárquico de 9 modelos con encogimiento KL hacia el nodo padre y olvido temporal."),
            paso("6", "Validación estadística", "Friedman-Nemenyi, Wilcoxon-Holm, DeLong, MCS y Diebold-Mariano con tamaños del efecto."),
            paso("7", "Interpretabilidad", "Pesos por nodo, aporte de cada modelo base y SHAP global y local frente a XGBoost."),
            paso("8", "Producción", "Artefactos con metadata.json, prueba de ida y vuelta y despliegue en Render."),
        ]))]),

        dbc.Card([encabezado("shield-check", "Limitaciones, ética y sesgos"), dbc.CardBody([
            html.Ul([
                html.Li("El modelo usa solo información de la semana t: choques de t+1 (bloqueos viales, lluvias extremas, "
                        "paros) no se pueden anticipar y limitan cualquier modelo."),
                html.Li("El SIPSA cubre las principales centrales: los corredores hacia plazas pequeñas no están representados."),
                html.Li("Las alertas no deben usarse para sancionar a productores o municipios: un riesgo alto refleja "
                        "condiciones de la ruta, no conductas individuales."),
                html.Li("Los municipios PDET y los corredores largos tienen tasas de interrupción mayores; el desempeño por "
                        "subgrupo se reporta en el informe para no ocultar errores concentrados en territorios vulnerables."),
                html.Li("La probabilidad es una estimación con incertidumbre; la decisión final requiere conocimiento local."),
            ], className="mb-1"),
            html.Small(["Autores: Camilo Molina Guerrero y Lina Margarita Buelvas · Versión 3.1 · Actualizado en octubre de 2026 · ",
                        html.A("Informe técnico (Jupyter Book)", href=DT.LIBRO, target="_blank"), " · ",
                        html.A("Código", href=DT.REPO, target="_blank")]),
        ])]),
    ])


@callback(Output("mapa-corredores", "figure"), Output("mapa-resumen", "children"), Input("mapa-central", "value"))
def actualizar_mapa(central):
    corr, cent = DT.mapa()
    if corr.empty:
        return PL.vacia("Faltan los datos del mapa (ejecute scripts/26_mapa_dashboard.py)"), ""
    d = corr[corr["semanas"] >= 26]
    if central:
        d = d[d["mercado"] == central]
    resumen = (f"{num(len(d))} corredores regulares (26 semanas o más) · tasa media de interrupción "
               f"{num(100 * d['tasa'].mean(), 1)} % · distancia mediana {num(d['dist_km'].median())} km · "
               f"{num(int(d['pdet'].sum()))} desde municipios PDET")
    return PL.mapa_corredores(corr, cent, central), resumen
