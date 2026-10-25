"""Dashboard predictivo del proyecto final: interrupciones en corredores de abastecimiento (SIPSA).

Aplicación multipágina (use_pages=True) con tres secciones: Contexto, EDA y Modelos. Solo carga
datos livianos (data/dashboard/) y modelos ya entrenados (artifacts/); nunca entrena ni optimiza.

Local:   python app.py            -> http://127.0.0.1:8050
Render:  gunicorn app:server      (ver render.yaml)
"""
import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import data as DT  # noqa: E402

FUENTES = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Nunito:wght@700;800&display=swap"

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP, FUENTES],
    title="Interrupciones en corredores de abastecimiento",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "description", "content": "Dashboard predictivo de interrupciones semanales en corredores de "
                                                  "abastecimiento de alimentos de Colombia (SIPSA, DANE)."}],
)
server = app.server  # servidor WSGI que usa gunicorn en Render

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([html.Img(src=app.get_asset_url("img/logo.svg"), alt=""),
                         "Corredores de abastecimiento · SIPSA"], href="/"),
        dbc.NavbarToggler(id="nav-toggler"),
        dbc.Collapse(dbc.Nav([
            dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
            dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact")),
            dbc.NavItem(dbc.Button([html.I(className="bi bi-journal-text me-1", **{"aria-hidden": "true"}), "Informe (Jupyter Book)"],
                                   href=DT.LIBRO, target="_blank", color="light", outline=True, size="sm",
                                   className="ms-lg-3 my-1", id="boton-libro")),
            dbc.NavItem(dbc.Button([html.I(className="bi bi-github me-1", **{"aria-hidden": "true"}), "Código"],
                                   href=DT.REPO, target="_blank", color="light", outline=True, size="sm",
                                   className="ms-lg-2 my-1", id="boton-repo")),
        ], navbar=True, className="ms-auto align-items-lg-center"), id="nav-collapse", navbar=True),
    ], fluid=True),
    color="primary", dark=True, fixed="top", expand="lg", className="navbar-principal",
)

app.layout = html.Div([
    navbar,
    dbc.Container(dash.page_container, fluid=True, className="px-3 px-md-4"),
    html.Footer(dbc.Container([
        html.Img(src=app.get_asset_url("img/productos.svg"), alt=""),
        "Camilo Molina Guerrero y Lina Margarita Buelvas · Machine Learning, Doctorado en Ingeniería · ",
        "Datos: SIPSA-Abastecimiento (DANE), 2018-2025 · Versión 3.1 (octubre de 2026) · Ilustraciones propias",
    ], fluid=True), className="pie"),
])


@dash.callback(dash.Output("nav-collapse", "is_open"), dash.Input("nav-toggler", "n_clicks"),
               dash.State("nav-collapse", "is_open"), prevent_initial_call=True)
def alternar_menu(n, abierto):
    return not abierto


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG", "0") == "1", port=int(os.environ.get("PORT", 8050)))
