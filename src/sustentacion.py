"""Secciones del dashboard que siguen la rúbrica de la sustentación.

Reúne los textos (con ecuaciones en LaTeX, que dibuja MathJax) y los bloques de las pestañas de fundamentos,
literatura, diseño experimental, validación, errores y conclusiones. Los textos resumen los capítulos 4 a 8 del
libro; las tablas y figuras se leen de data/dashboard/ (scripts/28_sustentacion_dashboard.py).
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dash_table, dcc, html

import data as DT
import plots as PL

ENCABEZADO_TABLA = {"backgroundColor": "#e9f2ea", "fontWeight": 700, "color": "#1f5135"}
CELDA = {"fontSize": 12, "textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "5px",
         "fontFamily": "Inter, Segoe UI, sans-serif"}


def md(texto: str, clase: str = "") -> dcc.Markdown:
    return dcc.Markdown(texto, mathjax=True, className=f"md-sustentacion {clase}".strip())


def tarjeta(titulo, cuerpo, icono: str | None = None, clase: str = "") -> dbc.Card:
    cabecera = [html.I(className=f"bi bi-{icono}", **{"aria-hidden": "true"}), titulo] if icono else titulo
    return dbc.Card([dbc.CardHeader(cabecera), dbc.CardBody(cuerpo)], className=clase)


def tabla(df: pd.DataFrame, page_size: int = 12, **kw) -> dash_table.DataTable:
    return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
                                page_size=page_size, style_table={"overflowX": "auto"}, style_cell=CELDA,
                                style_header=ENCABEZADO_TABLA, **kw)


def tabla_md(filas: list[list[str]], encabezado: list[str]) -> dbc.Table:
    """Tabla de texto con fórmulas: cada celda pasa por Markdown con MathJax."""
    cab = html.Thead(html.Tr([html.Th(c) for c in encabezado]))
    cuerpo = html.Tbody([html.Tr([html.Td(md(c, "mb-0")) for c in f]) for f in filas])
    return dbc.Table([cab, cuerpo], bordered=False, hover=True, responsive=True, size="sm", className="tabla-md")


def coma(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def signo4(x: float) -> str:
    x = round(float(x), 4) + 0.0              # sin «−0,0000»
    return f"{x:+.4f}".replace(".", ",") if x else "0,0000"


def lectura(texto) -> html.Div:
    return html.Div(texto, className="lectura")


# ------------------------------------------------------------------------------- criterio 1 · formulación del problema
FORMULACION_PROBLEMA = r"""
**Unidad de análisis.** Un corredor $c = (o, m)$ une un municipio de origen $o$ con una central mayorista $m$.
Sea $A_{c,t}$ el abastecimiento del corredor en la semana $t$ (toneladas por día) y $\bar A_{c,t}$ su promedio de
las 8 semanas previas. La razón de abastecimiento y la variable objetivo son

$$
R_{c,t+1} = \frac{A_{c,t+1}}{\bar A_{c,t}}, \qquad Y_{c,t+1} = \mathbb{1}\{R_{c,t+1} < 0{,}5\} \in \{0, 1\},
$$

con $R = 0$ cuando el corredor no envía nada: hay **interrupción** si llega menos de la mitad de lo habitual.

**Predictores.** $x_{c,t} \in \mathbb{R}^d$ reúne solo información disponible al cierre de la semana $t$: historia y
volatilidad del corredor, razones recientes $R_{c,t}$ y $R_{c,t-1}$, composición de la carga, calendario (los
festivos de $t+1$ se conocen de antemano), caída simultánea en la central y en el país y vecindad espacial.

**Tarea.** Clasificación binaria probabilística: estimar

$$
\pi(x) = P\left(Y_{c,t+1} = 1 \mid x_{c,t} = x\right)
$$

con un clasificador $\hat f$ entrenado con las semanas pasadas y evaluado hacia adelante. El criterio de selección
es el AUC, la probabilidad de que una interrupción reciba un puntaje mayor que una no interrupción,
$\mathrm{AUC}(\hat f) = P\big(\hat f(X^+) > \hat f(X^-)\big)$, porque no depende del umbral ni del desbalance
moderado (28 % de positivos). La alerta es $\hat f(x) \ge \tau$, con $\tau$ el umbral de máximo F1 en la validación
interna.
"""

# ------------------------------------------------------------------------------- criterio 2 · preprocesamiento y fuga
TRANSFORMACIONES = r"""
* **Objetivo normalizado.** $R$ divide por el promedio del propio corredor: corredores de 1 y de 500 t/día quedan en
  la misma escala, y la escala entra aparte como $\log \bar A_{c,t}$.
* **Dos preprocesadores dentro del pipeline** (`ColumnTransformer`, ajustado solo con el entrenamiento del pliegue):
  * *Lineales, distancias y núcleos* (regresión logística, KNN, Naive Bayes, SVM): imputación con la mediana,
    $\log(1 + x)$ en conteos, distancia y antigüedad, winsorización al 1 % y 99 %, splines cúbicos (5 nudos en
    cuantiles) en $R_{c,t}$ y $R_{c,t-1}$, estandarización y one-hot de central, departamento y grupo dominante.
  * *Árboles* (Decision Tree, Random Forest, XGBoost): imputación y one-hot; sin transformaciones monótonas,
    que no cambian las particiones de un árbol.
* **Municipio de origen:** codificación por objetivo (`TargetEncoder`) con ajuste cruzado interno de 5 particiones,
  para que ninguna fila use su propia etiqueta.
* **Calendario:** la semana del año entra como seno y coseno (es cíclica).
* **Balanceo** (SMOTE, ADASYN o `class_weight`) solo sobre el entrenamiento de cada pliegue.
"""

FUGA = r"""
1. **Información hasta $t$.** Todas las variables usan semanas $\le t$; la única de $t+1$ son los festivos, que se
   conocen de antemano.
2. **Todo se ajusta dentro del pliegue.** Imputación, escalado, codificaciones y balanceo viven en el pipeline y se
   ajustan solo con el entrenamiento de cada pliegue.
3. **Hueco de 8 semanas** entre entrenamiento y validación (el objetivo usa un promedio de 8 semanas) y de 10 semanas
   entre el entrenamiento final y 2025.
4. **Validación hacia adelante.** Ventana creciente: nunca se entrena con semanas posteriores a las que se evalúan.
5. **2025 no elige nada.** Los hiperparámetros se eligieron con la validación interna; el combinador del AJ-KL solo ve
   predicciones fuera de muestra de semanas anteriores al corte de cada pliegue.
"""


# ------------------------------------------------------------------------------- criterio 3 · fundamentos del AJ-KL
FORMULACION_AJKL = r"""
Sean $p_i = (p_{i1}, \dots, p_{iM}) \in [\varepsilon, 1-\varepsilon]^M$ las probabilidades **fuera de muestra** de los
$M = 9$ modelos base para la corredor-semana $i$ (los 7 del curso, LightGBM y CatBoost), $y_i$ su etiqueta y $t_i$ su
semana. La jerarquía es país → central → central × grupo de alimentos → corredor; cada nodo $r$ tiene un padre
$\pi(r)$ y un vector de pesos en el símplex $\Delta^M = \{w \ge 0,\ \sum_m w_m = 1\}$.

**Predicción: pool lineal del nodo más profundo de la fila**

$$
q_i = w_r^\top p_i .
$$

**Función objetivo de cada nodo** (de la raíz hacia las hojas)

$$
w_r = \arg\min_{w \in \Delta^M} J_r(w), \qquad
J_r(w) = \underbrace{\sum_{i \in r} \gamma^{\,t_{\max} - t_i}\, \ell(y_i, w^\top p_i)}_{\text{log-loss con olvido}}
\; + \; \underbrace{\lambda\, \mathrm{KL}(w \,\|\, w_{\pi(r)})}_{\text{encogimiento hacia el padre}},
$$

con la **pérdida logarítmica** $\ell(y, q) = -y \log q - (1-y)\log(1-q)$, el **olvido** $\gamma \in (0, 1]$ (semivida
de $\log 2 / \log(1/\gamma)$ semanas) y la **fuerza del encogimiento** $\lambda \ge 0$; en la raíz el padre es la
uniforme. Dividiendo por el tamaño de muestra efectivo $S_r = \sum_{i \in r}\gamma^{t_{\max}-t_i}$,

$$
J_r / S_r = f_r(w) + \mu_r\, \mathrm{KL}(w \,\|\, w_{\pi(r)}), \qquad \mu_r = \lambda / S_r :
$$

**el encogimiento efectivo crece cuando el nodo tiene pocos datos.** Después del pool, una calibración beta
$\sigma(c + a \log q - b \log(1-q))$ con $a, b \ge 0$ corrige la subdispersión del promedio, solo si mejora la
log-loss interna.
"""

OPTIMIZACION = r"""
Cada nodo se resuelve con **descenso espejo compuesto** con la entropía como geometría (Beck y Teboulle, 2003): el
paso $k$ resuelve $\min_{w \in \Delta^M} \eta \langle \nabla f_r(w^{(k)}), w\rangle + \eta\mu_r \mathrm{KL}(w\|w_{\pi(r)}) +
\mathrm{KL}(w\|w^{(k)})$, que tiene **solución cerrada** (proposición 2):

$$
\log w^{(k+1)}_m = \frac{\log w^{(k)}_m - \eta\, \nabla_m f_r(w^{(k)}) + \eta \mu_r \log w_{\pi(r),m}}{1 + \eta \mu_r} - c_k,
\qquad
\nabla_m f_r(w) = \frac{1}{S_r}\sum_{i \in r} \gamma^{t_{\max}-t_i}\, \frac{q_i - y_i}{q_i(1-q_i)}\, p_{im}.
$$

* Es una **actualización multiplicativa**: respeta el símplex sin proyecciones y la KL hacia el padre entra exacta.
* El paso $\eta$ se ajusta por **retroceso** (se acepta si $F_r = f_r + \mu_r\mathrm{KL}$ baja). Si el paso se estanca
  (un peso casi en cero solo crece en proporción a sí mismo), se prueba un paso de **Frank-Wolfe** hacia el vértice
  $e_j$, $j = \arg\min_m \nabla_m F_r$, con búsqueda lineal.
* **Parada con certificado:** la brecha de Frank-Wolfe $g(w) = \langle \nabla F_r(w), w\rangle - \min_m \nabla_m F_r(w)$
  acota el error, $F_r(w) - F_r^\star \le g(w)$; se detiene con $g(w) < 10^{-8}$.
* *¿Por qué no L-BFGS?* Llega al mismo óptimo (se usa como referencia en las pruebas), pero exige reparametrizar el
  símplex; el paso espejo es cerrado, interpretable (como los pesos exponenciales de la literatura de expertos) y
  cuesta $O(nM)$ por iteración.
"""

REGULARIZACION = r"""
**La KL hacia el padre es la regularización.** Con $\lambda > 0$ el objetivo es estrictamente convexo y el óptimo es
único e interior (proposición 1). Es el **estimador MAP** de una verosimilitud de Bernoulli ponderada con un prior
entrópico centrado en el padre (proposición 5):

$$
\pi(w) \propto \exp\{-\lambda\, \mathrm{KL}(w \,\|\, w_{\pi(r)})\}
$$

$$
\log \pi(w \mid \text{datos}) = -J_r(w) + \text{cte},
$$

así que es un Bayes empírico jerárquico sobre los pesos: con $S_r \gg \lambda$ dominan los datos del nodo; con
$S_r \ll \lambda$ el nodo copia a su padre. **Casos límite** (proposición 4): $\lambda \to \infty$ da el promedio
simple; $\lambda \to 0^+$, el apilamiento local; profundidad 1, el apilamiento global; $\gamma = 1$, sin olvido. Las
reglas estándar son casos particulares del AJ-KL, lo que hace natural la ablación.
"""

ESTADISTICO = r"""
* **¿Por qué la log-loss?** Es una regla de puntuación **estrictamente propia** (proposición 7): si
  $Y \sim \mathrm{Ber}(\pi)$, $\ \mathbb{E}[\ell(Y, q)] - \mathbb{E}[\ell(Y, \pi)] = \mathrm{KL}(\mathrm{Ber}(\pi)\,\|\,\mathrm{Ber}(q)) \ge 0$,
  con igualdad solo en $q = \pi$. Los pesos buscan probabilidades calibradas, no solo un buen orden.
* **Capacidad y generalización** (proposiciones 6 y 8). Fijados los modelos base, la clase del nodo es la envolvente
  convexa de $M$ funciones, con complejidad de Rademacher $\le \sqrt{2\log M / n_r}$. Con la contracción de
  Ledoux-Talagrand (la log-loss es $1/\varepsilon$-lipschitziana), con probabilidad $\ge 1 - \delta$:
  $$
  R(w) \le \hat R(w) + \frac{2}{\varepsilon}\sqrt{\frac{2\log M}{n_r}} + \log\frac{1}{\varepsilon}\sqrt{\frac{\log(1/\delta)}{2 n_r}} .
  $$
  La cota depende de $\log M$ y no del número de variables: el combinador tiene muy poca capacidad por nodo, y el
  encogimiento protege a los nodos pequeños, donde el término $1/\sqrt{n_r}$ es grande.
* **La calibración no cambia el AUC** (proposición 9): con $a, b \ge 0$,
  $g'(q) = g(q)\{1-g(q)\}\{a/q + b/(1-q)\} > 0$, así que conserva el orden.
* **Consistencia** (proposición 10): con $\gamma = 1$ y filas independientes, $\mu_r = \lambda/n_r \to 0$ y
  $\hat w_r \to w_r^\star$, el mejor pool lineal del nodo (teorema de los estimadores extremos).
"""

PROPOSICIONES = [
    ("1 · Convexidad y unicidad",
     r"Para $\lambda > 0$, $J_r$ es estrictamente convexa en $\Delta^M$ y tiene un único minimizador, interior si el "
     r"padre tiene soporte completo.",
     r"$\ell(y, w^\top p)$ es convexa en $w$ (composición afín de $-\log$); la KL tiene hessiano $\mathrm{diag}(1/w_m) \succ 0$. "
     r"Convexa + estrictamente convexa en un compacto convexo: existe y es único. Como $\partial \mathrm{KL}/\partial w_m \to -\infty$ "
     r"en $w_m \to 0$, el óptimo no tiene componentes nulas."),
    ("2 · Actualización exacta y KKT",
     r"La actualización multiplicativa es la solución única del subproblema del descenso espejo, y sus puntos fijos son "
     r"exactamente las condiciones KKT del nodo.",
     r"Lagrangiano con multiplicador para $\sum_m w_m = 1$; la entropía mantiene $w_m > 0$. Igualando la derivada a cero "
     r"se obtiene la fórmula cerrada. En un punto fijo, $\nabla_m f + \mu(\log w_m - \log v_m + 1)$ es constante en $m$: "
     r"estacionariedad sobre el símplex."),
    ("3 · Certificado y convergencia",
     r"(a) $F_r(w) - F_r^\star \le g(w)$ para todo $w$. (b) Con paso $\eta \le 1/L$, $F_r(w^{(K)}) - F_r^\star \le "
     r"\mathrm{KL}(w^\star \| w_{\pi(r)})/(\eta K)$.",
     r"(a) Por convexidad, $F(w) - F(w^\star) \le \langle \nabla F(w), w - w^\star\rangle \le g(w)$. (b) $f_r$ es $L$-suave "
     r"respecto de $\|\cdot\|_1$ con $L \le 1/\varepsilon^2$; la entropía es 1-fuertemente convexa (Pinsker) y se aplica la "
     r"tasa del gradiente de Bregman para funciones relativamente suaves (Lu, Freund y Nesterov, 2018)."),
    ("4 · Casos límite",
     r"$\lambda \to 0^+$: apilamiento local (el minimizador más cercano al padre); $\lambda \to \infty$: promedio simple; "
     r"profundidad 1: apilamiento global; $\gamma = 1$: sin olvido.",
     r"Regularización de Tikhonov para $\lambda \to 0^+$; dividiendo por $\lambda$, el problema tiende a "
     r"$\min \mathrm{KL}(w\|w_{\pi(r)})$, cuyo óptimo es el padre, y por inducción desde la raíz todo tiende a la uniforme."),
    ("5 · Interpretación bayesiana",
     r"El nodo es el MAP de una verosimilitud de Bernoulli ponderada por $\gamma^{t_{\max}-t_i}$ con prior "
     r"$\pi(w) \propto \exp\{-\lambda \mathrm{KL}(w\|w_{\pi(r)})\}$.",
     r"El logaritmo del posterior es $-\sum_i \gamma^{t_{\max}-t_i}\ell(y_i, w^\top p_i) - \lambda\mathrm{KL}(w\|w_{\pi(r)})$ "
     r"$+ \text{cte} = -J_r(w) + \text{cte}$."),
    ("6 · Capacidad",
     r"La clase $\{x \mapsto w^\top p(x) : w \in \Delta^M\}$ tiene complejidad de Rademacher empírica $\le \sqrt{2\log M/n_r}$.",
     r"El supremo de una función lineal sobre el símplex está en un vértice: la complejidad de la envolvente convexa es la "
     r"de los $M$ modelos base, y el lema de Massart da la cota."),
    ("7 · Regla de puntuación propia",
     r"$\mathbb{E}[\ell(Y,q)] - \mathbb{E}[\ell(Y,\pi)] = \mathrm{KL}(\mathrm{Ber}(\pi)\|\mathrm{Ber}(q))$: el único "
     r"minimizador es $q = \pi$.",
     r"Se desarrolla la diferencia y se reconoce la definición de la KL; la desigualdad de Gibbs da $\ge 0$ con igualdad "
     r"solo en $q = \pi$."),
    ("8 · Cota de generalización",
     r"Con filas i.i.d., con probabilidad $\ge 1-\delta$ y para todo $w$: $R(w) \le \hat R(w) + \frac{2}{\varepsilon}"
     r"\sqrt{2\log M/n_r} + \log(1/\varepsilon)\sqrt{\log(1/\delta)/(2n_r)}$.",
     r"La log-loss es $(1/\varepsilon)$-lipschitziana y acotada por $\log(1/\varepsilon)$; cota uniforme de Rademacher "
     r"(Bartlett y Mendelson) + contracción de Ledoux-Talagrand + proposición 6."),
    ("9 · Calibración monótona",
     r"$g(q) = \sigma(c + a\log q - b\log(1-q))$ con $a, b \ge 0$ es estrictamente creciente y no cambia el AUC.",
     r"$g'(q) = g(1-g)(a/q + b/(1-q)) > 0$; el AUC solo depende del orden. En los pliegues calibrados salió "
     r"$a \in [1{,}09; 1{,}19]$ y $b \in [0{,}77; 0{,}99]$: la restricción no se activó."),
    ("10 · Consistencia",
     r"Con $\gamma = 1$, $\lambda$ fijo, filas i.i.d. y minimizador poblacional único, $\hat w_r \to w_r^\star$ en "
     r"probabilidad cuando $n_r \to \infty$.",
     r"$(\lambda/n_r)\mathrm{KL}$ tiende a cero uniformemente; la convergencia uniforme del riesgo empírico (proposición 8) "
     r"y el teorema de consistencia de los estimadores extremos (Newey y McFadden, teorema 2.1) dan el resultado."),
]

SUPUESTOS = [
    ["Probabilidades base fuera de muestra", "Toda la formulación",
     "El historial del pliegue $k$ son predicciones de bloques externos anteriores; ninguna semana es posterior al corte"],
    ["Jerarquía anidada", "Proposiciones 1 a 5", "El código del corredor incluye la central y el grupo: se cumple por construcción"],
    [r"$p_i \in [\varepsilon, 1-\varepsilon]$", "Proposiciones 3 y 8", "Recorte explícito en el código"],
    ["Filas independientes en el nodo", "Proposiciones 8 y 10",
     "No se cumple estrictamente; por eso la evidencia es la validación hacia adelante y pruebas con bloques de semana y varianza HAC"],
    ["La relación cambia lentamente", r"Olvido con semivida $\log 2/\log(1/\gamma)$", r"Sensibilidad a $\gamma$ en la robustez"],
    ["Calibración monótona", "Proposición 9", r"Restricción $a, b \ge 0$ en el código y prueba unitaria"],
]

TEORIA_EVIDENCIA = [
    ["1 · Convexidad", "Un solo óptimo por nodo", "`test_descenso_espejo_alcanza_el_optimo_de_referencia`", "Mismo óptimo que L-BFGS"],
    ["2 y 3 · KKT y certificado", "Error en el objetivo menor que la tolerancia", "`test_converge_con_modelos_correlacionados_y_valores_por_defecto`",
     "Brechas por nodo (abajo)"],
    ["4 · Casos límite", "Las reglas estándar son casos particulares", "`test_casos_limite_de_lambda`", "Ablación"],
    ["5 · Bayes", r"El encogimiento depende de $\lambda/S_r$", "`test_recupera_los_pesos_de_cada_region`", "Figura de encogimiento"],
    ["6 y 8 · Capacidad", r"La brecha decrece como $\sqrt{\log M/n_r}$", "—", "Robustez: historial reducido"],
    ["7 · Regla propia", "Probabilidades calibradas", "`test_calibracion_beta_mejora_un_pool_descalibrado`", "Brier, log-loss y ECE"],
    ["9 · Calibración", "El AUC no cambia al calibrar", "`test_calibracion_beta_es_monotona_y_conserva_el_auc`", r"$a, b > 0$ en los pliegues"],
    ["10 · Consistencia", "Más datos, mejor pool local", "`test_recupera_los_pesos_de_cada_region`", "Curva de tamaño de muestra"],
]

COMPLEJIDAD = r"""
Con $L$ niveles, $K$ iteraciones por nodo y $n$ filas: cada fila pertenece a un solo nodo por nivel, así que una
iteración sobre todo el nivel cuesta $O(nM)$ y agrupar por nodo $O(n \log n)$.

* **Entrenamiento:** $O\big(L\,(n \log n + K\, n\, M)\big)$, más los modelos base, que ya están entrenados.
* **Inferencia:** $O(L \log N + M)$ por fila ($N$ nodos por nivel: búsqueda binaria y un producto interno).
* **Memoria:** $O(|\mathcal T|\, M)$, un vector de $M$ pesos por nodo.
"""


def fundamentos() -> list:
    nod = DT.tabla("encogimiento_nodos")
    comp = DT.tabla("complejidad_ajkl")
    bloques = [
        dbc.Alert([html.I(className="bi bi-mortarboard me-2"), "Criterio 3 de la rúbrica (25 %): formulación del algoritmo, función "
                   "objetivo, fundamento estadístico y matemático, optimización, supuestos, regularización, complejidad y "
                   "generalización. Las demostraciones completas están en el capítulo 5 del informe."], color="light", className="guia-rubrica"),
        tarjeta("Formulación matemática y función objetivo", md(FORMULACION_AJKL), "function"),
        tarjeta("Principio de optimización: descenso espejo con certificado", md(OPTIMIZACION), "graph-down-arrow"),
        dbc.Row([
            dbc.Col(tarjeta("Regularización: encogimiento KL e interpretación bayesiana", md(REGULARIZACION), "bullseye", "h-100"), lg=5,
                    className="mb-3"),
            dbc.Col(tarjeta("Fundamento estadístico", md(ESTADISTICO), "clipboard-data", "h-100"), lg=7, className="mb-3"),
        ]),
        tarjeta("Las diez proposiciones (enunciado e idea de la demostración)", dbc.Accordion([
            dbc.AccordionItem([md(f"**Enunciado.** {e}"), md(f"*Idea de la demostración.* {d}", "text-secondary")],
                              title=f"Proposición {t}") for t, e, d in PROPOSICIONES], start_collapsed=True, always_open=True), "list-ol"),
    ]
    if len(nod):
        res = nod.groupby("nivel", sort=False).agg(nodos=("nodo", "size"), filas_mediana=("n", "median"), mu_mediana=("mu", "median"),
                                                    iter_max=("iteraciones", "max"), brecha_max=("brecha", "max"))
        res = res.reset_index().rename(columns={"nivel": "Nivel", "nodos": "Nodos", "filas_mediana": "Filas (mediana)",
                                                "mu_mediana": "μ_r (mediana)", "iter_max": "Iteraciones (máx.)",
                                                "brecha_max": "Brecha FW (máx.)"})
        res["μ_r (mediana)"] = res["μ_r (mediana)"].map(lambda v: f"{v:.2e}")
        res["Brecha FW (máx.)"] = res["Brecha FW (máx.)"].map(lambda v: f"{v:.1e}")
        rho = nod[nod["nivel"] != "raíz"][["S_r", "kl_padre"]].corr(method="spearman").iloc[0, 1]
        bloques.append(dbc.Row([
            dbc.Col(tarjeta("Evidencia del encogimiento (combinador del pliegue final)", [
                dcc.Graph(figure=PL.encogimiento_nodos(nod)),
                lectura(f"Cada punto es un nodo. Los nodos con poca muestra efectiva quedan cerca de los pesos de su padre (KL "
                        f"pequeña) y solo los que tienen mucha evidencia se apartan: correlación de Spearman entre S_r y la KL "
                        f"hacia el padre de {coma(rho, 2)}. Es la proposición 5 en los datos."),
            ], "diagram-3"), lg=7),
            dbc.Col(tarjeta("Convergencia por nivel (proposición 3)", [
                tabla(res, page_size=5),
                lectura(f"La brecha de Frank-Wolfe acota el error en el objetivo de cada nodo (F(w) − F* ≤ brecha). Su mediana es "
                        f"{nod['brecha'].median():.1e} y la máxima {nod['brecha'].max():.1e}: los pesos reportados son el óptimo "
                        "de J_r salvo esa tolerancia, no un punto donde el algoritmo se detuvo por número de iteraciones."),
            ], "check2-circle"), lg=5),
        ]))
    bloques.append(dbc.Row([
        dbc.Col(tarjeta("Supuestos y cómo se verifican", tabla_md(SUPUESTOS, ["Supuesto", "Dónde se usa", "Cómo se verifica o se mitiga"]),
                        "list-check"), lg=6),
        dbc.Col(tarjeta("Complejidad y capacidad de generalización", [
            md(COMPLEJIDAD),
            dcc.Graph(figure=PL.complejidad_tiempos(comp)) if len(comp) else html.Div(),
            lectura((f"Con todo el historial ({comp['n'].max():,} filas) el combinador se entrena en "
                     f"{coma(comp['segundos_fit'].iloc[-1], 1)} s y predice 1.000 filas en {coma(comp['ms_por_1000_pred'].iloc[-1], 1)} ms, "
                     "sin contar los modelos base. La capacidad por nodo crece con log M, no con el número de variables "
                     "(proposiciones 6 y 8).").replace(",", ".", 1) if len(comp) else ""),
        ], "speedometer2"), lg=6),
    ]))
    bloques.append(tarjeta("De la teoría a la evidencia",
                           tabla_md(TEORIA_EVIDENCIA, ["Proposición", "Qué garantiza", "Prueba unitaria", "Evidencia empírica"]), "link-45deg"))
    return bloques


# ------------------------------------------------------------------------------- literatura y novedad (criterio 3, R1)
PROTOCOLO_LITERATURA = r"""
* **Problema:** predicción de interrupciones semanales de corredores de abastecimiento de alimentos con los
  microdatos del SIPSA-A (clasificación binaria).
* **Familia del modelo:** combinación de clasificadores con pesos locales o jerárquicos (apilamiento jerárquico,
  selección dinámica de ensambles, combinación de pronósticos de probabilidad, pesos exponenciales).
* **Bases:** Web of Science Core Collection, Scopus y ScienceDirect (acceso institucional), consultadas el
  **23 de octubre de 2026**; seis cadenas por base (una en español) más cadenas de método puro (M1).
* **Revisión:** título y resumen de los resultados y texto completo de los trabajos más cercanos.
"""

FAMILIAS = pd.DataFrame([
    ["A. Apilamiento con pesos heterogéneos o jerárquicos", "Los pesos cambian según la región, con agrupamiento parcial",
     "El desempeño de cada modelo cambia entre centrales y grupos", "Yao et al. (2022); Kejzlar et al. (2023); Naimi y Balzer (2018)"],
    ["B. Combinación de pronósticos de probabilidad", "Pool lineal y su versión con beta (BLP) corrigen la calibración",
     "La salida es una probabilidad", "Ranjan y Gneiting (2010); Gneiting y Ranjan (2013)"],
    ["C. Selección dinámica de ensambles", "Pondera clasificadores según su competencia local",
     "Aporta la idea de región de competencia", "Ko et al. (2008); Woloszynski y Kurzynski (2011); Cruz et al. (2018)"],
    ["D. Agregación con pesos exponenciales", "Actualizaciones multiplicativas con cotas de arrepentimiento",
     "Aporta la regla de actualización", "Cesa-Bianchi et al. (2007); Wintenberger (2017); Berrisch y Ziel (2023)"],
    ["E. Árboles y boosting con efectos mixtos", "Modelan la heterogeneidad por grupo dentro del aprendiz",
     "El panel es agrupado y longitudinal", "Sela y Simonoff (2012); Hajjem et al. (2014); Sigrist (2022)"],
    ["F. Aprendizaje bajo deriva temporal", "Reponderan las observaciones según su antigüedad",
     "Se entrena hasta 2024 y se prueba en 2025", "Scholz y Klinkenberg (2007); Kanamori et al. (2009)"],
], columns=["Familia", "Qué mejora", "Por qué encaja", "Referencias clave"])

CERCANOS = r"""
**Por aplicación.** Niño (2026, *Food Policy*) estima con variables instrumentales el efecto causal de los cierres
viales sobre los precios de los mercados mayoristas colombianos: **no predice interrupciones ni combina modelos**.
RIOT (Zhang et al., 2027) combina probabilidades con transporte óptimo para retrasos en cadenas de suministro con una
**regla global**; P-WEV (Singh et al., 2026) pondera modelos GARCH con PSO para la volatilidad de precios agrícolas
(**regresión**).

**Por método.** El apilamiento jerárquico bayesiano (Yao, Pirš, Vehtari y Gelman, 2022) es el más cercano: pesos que
varían con los datos y se agrupan parcialmente, pero con **MCMC y pesos softmax-lineales, sin olvido**. Yazdi (2026)
compara el encogimiento de Bayes empírico con GBDT en distribución minorista: el peso global se va al GBDT, aunque el
encogimiento gana en un estrato, lo que respalda los **pesos locales**.
"""

DECLARACION = r"""
Al mejor de nuestro conocimiento y según la búsqueda en **Web of Science, Scopus y ScienceDirect** (23 de octubre de
2026), complementada con 20 cadenas por aplicación en Web of Science, **no se encontraron estudios que apliquen un
apilamiento jerárquico con encogimiento KL y olvido temporal, ni ningún otro esquema de combinación local de
clasificadores, a la predicción de interrupciones semanales de corredores de abastecimiento de alimentos.** El AJ-KL
se diferencia en que estima los pesos del ensamble sobre una jerarquía anidada propia del problema, con penalización
KL hacia el nodo padre y olvido temporal, mediante un estimador convexo con actualización multiplicativa de costo lineal.
"""


def literatura() -> list:
    m = DT.tabla("literatura_matriz")
    cif = DT.sustentacion()
    afines = DT.tabla("metodos_afines")
    bloques = [dbc.Alert([html.I(className="bi bi-search me-2"), "Evidencia de novedad (condición R1): la novedad no se declara, se "
                          "contrasta con una búsqueda documentada. Detalle completo en el capítulo 4 del informe."],
                         color="light", className="guia-rubrica")]
    tot = m.groupby("base", sort=False)[["resultados", "revisados", "relevantes"]].sum() if len(m) else pd.DataFrame()
    bloques.append(dbc.Row([
        dbc.Col(tarjeta("Protocolo de búsqueda", md(PROTOCOLO_LITERATURA), "journal-text"), lg=5),
        dbc.Col(tarjeta("Matriz de novedad", [
            dcc.Graph(figure=PL.matriz_literatura(m)) if len(m) else html.Div(),
            lectura((f"{int(tot['resultados'].sum())} registros en las tres bases, {int(tot['revisados'].sum())} revisados y "
                     f"{int(tot['relevantes'].sum())} relevantes: trabajos de la misma familia en otras aplicaciones o del mismo "
                     "sistema de abastecimiento con otra tarea. Búsqueda complementaria en WoS por aplicación: "
                     f"{cif.get('wos_cadenas', 0)} cadenas, {cif.get('wos_resultados', 0)} resultados, {cif.get('wos_revisados', 0)} "
                     f"revisados y {cif.get('wos_relevantes', 0)} relevantes. Ninguno aplica una combinación local o jerárquica de "
                     "clasificadores a interrupciones de corredores.") if len(m) else ""),
        ], "bar-chart"), lg=7),
    ]))
    if len(m):
        t = m[["base", "cadena", "resultados", "revisados", "relevantes", "observacion"]].rename(
            columns={"base": "Base", "cadena": "Cadena", "resultados": "Resultados", "revisados": "Revisados",
                     "relevantes": "Relevantes", "observacion": "Observación"})
        bloques.append(tarjeta("Cadenas de búsqueda (pase el cursor sobre la cadena para ver la consulta)", tabla(
            t, page_size=8, sort_action="native", filter_action="native",
            tooltip_data=[{"Cadena": {"value": q, "type": "text"}} for q in m["consulta"]], tooltip_duration=None), "table"))
    bloques.append(dbc.Row([
        dbc.Col(tarjeta("Familias de modelos mejorados encontradas", tabla(FAMILIAS, page_size=6), "diagram-2"), lg=7),
        dbc.Col(tarjeta("Trabajos más cercanos", md(CERCANOS), "bookmark-star"), lg=5),
    ]))
    if len(afines):
        bloques.append(tarjeta("Diferencias con la literatura (métodos afines por familia algorítmica)", tabla(afines, page_size=8), "intersect"))
    bloques.append(tarjeta("Declaración de novedad (R1)", md(DECLARACION), "patch-check", clase="tarjeta-veredicto"))
    return bloques


# ------------------------------------------------------------------------------- criterio 4 · diseño experimental
VALIDACION_ANIDADA = r"""
* **Bucle externo:** los últimos cuatro años del periodo de entrenamiento en bloques de **52 semanas** (k1 a k4), con
  ventana creciente y un **hueco de 8 semanas** antes de cada bloque. Un quinto pliegue (*final*) entrena hasta el
  27-10-2024 y se prueba con **todo 2025**.
* **Bucle interno:** tres ventanas crecientes dentro del entrenamiento de cada bloque externo eligen los
  hiperparámetros (AUC interno). Para los 140 modelos del curso, la búsqueda usa una muestra de 40 mil filas por
  pliegue con la misma proporción en cada semana.
* **El combinador del AJ-KL** se entrena con las predicciones fuera de muestra de los bloques externos anteriores (en
  k1, de las ventanas internas): nunca ve una semana posterior al corte.
* **Mismas reglas para todos:** mismos pliegues, misma semilla (42), mismo presupuesto de TPE (24 evaluaciones) y el
  mismo criterio (AUC interno).
"""

JUSTIFICACION_HP = r"""
| Hiperparámetro | Espacio de búsqueda | Qué controla |
|---|---|---|
| $\lambda$ | $[0{,}01;\ 1000]$ (escala log) | Fuerza del prior hacia el padre; actúa como $\mu_r = \lambda/S_r$ |
| $\gamma$ | $[0{,}98;\ 1]$ | Olvido: semivida $\log 2/\log(1/\gamma)$ semanas (de 34 semanas a infinito) |
| Profundidad | $\{1, 2, 3, 4\}$ | Hasta qué nivel de la jerarquía hay pesos propios |
| Calibrar | sí / no | Calibración beta solo si baja la log-loss interna |

**TPE** (Optuna, semilla 42, 24 evaluaciones, 8 de arranque), igual que para los benchmarks: con tres
hiperparámetros continuos y un objetivo ruidoso, la búsqueda bayesiana usa mejor el presupuesto que una malla.
"""

IMPLEMENTACION = r"""
* `src/original_model.py`: clase `ApilamientoJerarquicoKL(ClassifierMixin, BaseEstimator)`, programada solo con
  **numpy y scipy** (objetivo, gradiente, paso espejo, brecha de Frank-Wolfe, recorrido de la jerarquía y calibración
  beta por Newton ponderado). Ninguna parte del combinador viene de un `import`.
* Compatible con scikit-learn: `fit`, `predict_proba`, `clone`, `cross_val_score` con AUC.
* **22 pruebas unitarias** del modelo (83 en total): recupera los pesos de cada región en datos sintéticos, alcanza el
  óptimo de SLSQP y de L-BFGS, casos límite de $\lambda$, olvido, nodos no vistos, calibración monótona,
  reproducibilidad y entradas inválidas.
* Los 10 modelos están serializados en `artifacts/` con `metadata.json` (librerías, semilla, hiperparámetros, esquema
  de entrada y métricas) y una prueba de ida y vuelta.
"""

REPRODUCIBILIDAD = r"""
* Semilla global **42** (`src/config.py`), propagada a numpy, scikit-learn, XGBoost, Optuna y DEAP. El AJ-KL es
  determinista: la semilla solo fija el TPE (cinco semillas dan el mismo AUC salvo una diezmilésima).
* Versiones exactas en `requirements.txt` (dashboard) y `requirements-dev.txt` (experimentos y libro).
* Pipeline numerado en `scripts/` y libro ejecutable (capítulo 9); este tablero solo lee resultados y artefactos.
"""

DECISIONES = r"""
* **Descenso espejo y no L-BFGS:** paso cerrado sobre el símplex y certificado de parada; L-BFGS se usa como referencia.
* **Muestra de 40 mil filas** para la búsqueda de los 140 modelos: la validación anidada completa tomó 29 horas-CPU.
* **Predicciones fuera de muestra guardadas** para que el combinador no reentrene los modelos base.
* **El tablero no entrena:** carga artefactos y tablas precalculadas, y calcula la Spearman con rangos para responder
  rápido en el plan gratuito de Render.
"""


def diseno() -> list:
    pl = DT.tabla("pliegues")
    hp = DT.tabla("hiperparametros_ajkl")
    hist = DT.tabla("tpe_historial")
    bloques = [dbc.Alert([html.I(className="bi bi-gear-wide-connected me-2"), "Criterio 4 de la rúbrica (15 %): implementación, "
                          "hiperparámetros, separación de los datos, estrategia de validación, reproducibilidad y decisiones "
                          "computacionales."], color="light", className="guia-rubrica"),
               tarjeta("Separación de los datos y estrategia de validación", [
                   dcc.Graph(figure=PL.linea_tiempo_pliegues(pl)) if len(pl) else html.Div(), md(VALIDACION_ANIDADA)], "calendar-range")]
    if len(hp):
        t = hp.copy()
        t["λ"] = t["λ"].map(lambda v: f"{v:.3g}".replace(".", ","))
        t["γ"] = t["γ"].map(lambda v: coma(v, 4))
        t["semivida (semanas)"] = t["semivida (semanas)"].map(lambda v: "∞" if not np.isfinite(v) else f"{v:.0f}")
        t["calibrar"] = t["calibrar"].map({True: "sí", False: "no"})
        t["filas del historial"] = t["filas del historial"].map(lambda v: f"{int(v):,}".replace(",", "."))
        bloques.append(dbc.Row([
            dbc.Col(tarjeta("Hiperparámetros: espacio y justificación", md(JUSTIFICACION_HP), "sliders"), lg=5),
            dbc.Col(tarjeta("Lo que eligió TPE en cada pliegue", [
                tabla(t, page_size=5),
                dcc.Graph(figure=PL.convergencia_tpe(hist).update_layout(title="Convergencia de TPE: mejor AUC interno acumulado"))
                if len(hist) else html.Div(),
            ], "bullseye"), lg=7),
        ]))
    bloques.append(dbc.Row([
        dbc.Col(tarjeta("Implementación del algoritmo", md(IMPLEMENTACION), "code-slash"), lg=5),
        dbc.Col([tarjeta("Reproducibilidad", md(REPRODUCIBILIDAD), "arrow-repeat"),
                 tarjeta("Decisiones computacionales", md(DECISIONES), "cpu")], lg=7),
    ]))
    return bloques


# ------------------------------------------------------------------------------- criterio 5 · métricas, pruebas y robustez
METRICAS = r"""
| Métrica | Qué mide | Por qué se usa aquí |
|---|---|---|
| **AUC** (selección) | $P(\hat f(X^+) > \hat f(X^-))$ | No depende del umbral ni del desbalance moderado; ordena corredores por riesgo |
| AUC-PR | Precisión frente a exhaustividad | Se concentra en la clase positiva (28 %) |
| Brier | $\frac1n\sum(\hat p_i - y_i)^2$ | Exactitud de la probabilidad |
| Log-loss | $-\frac1n\sum[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]$ | Regla propia; es la pérdida del AJ-KL |
| ECE | Diferencia media entre probabilidad y frecuencia por intervalos | Calibración: leer $\hat p$ como frecuencia |
| F1 | Media armónica de precisión y exhaustividad | Fija el umbral de alerta $\tau$ |
"""

CRITERIO_SUPERIORIDAD = r"""
El modelo original es **superior** solo si (guía, sección 6.1): **(1)** su AUC medio del bucle externo supera al de la
mejor combinación de cada uno de los siete modelos; **(2)** al menos **tres pruebas distintas** rechazan la igualdad con
p ajustado < 0,05 (0,10 en el MCS); y **(3)** el tamaño del efecto no es despreciable. Universo: el AJ-KL y las siete
mejores combinaciones, con las mismas particiones, emparejadas fila por fila. Bloques: 48 meses del bucle externo y
52 semanas de 2025. Ajuste por comparaciones múltiples: Holm en cada familia de siete comparaciones.
"""

PRUEBAS = [
    ["Friedman + permutación por pliegue",
     r"$\chi^2_F = \frac{12N}{k(k+1)}\big[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\big]$, $F_F = \frac{(N-1)\chi^2_F}{N(k-1)-\chi^2_F}$",
     "Todos los modelos tienen los mismos rangos", "Permutación restringida al pliegue"],
    ["Nemenyi", r"$|R_i - R_j| > \mathrm{CD} = q_\alpha\sqrt{k(k+1)/(6N)}$", "Mismo rango medio por par", "Ninguna (conservadora)"],
    ["Wilcoxon-Holm", r"$W^+ = \sum_{d_b > 0}\mathrm{rango}(|d_b|)$; Holm: $\tilde p_{(i)} = \max_{j\le i}\min\{1,(m-j+1)p_{(j)}\}$",
     "Mediana de $d_b$ igual a cero", "Se revisa la autocorrelación"],
    ["DeLong", r"$z = (\hat\theta_1 - \hat\theta_2)/\sqrt{\widehat{\mathrm{Var}}(\hat\theta_1 - \hat\theta_2)}$, AUC como estadístico U",
     "Mismo AUC en 2025", "Error estándar por bloques de semana (BCa)"],
    ["Diebold-Mariano (HLN)", r"$\mathrm{DM} = \bar d/\sqrt{\hat\sigma^2_{\mathrm{HAC}}/T}$, corregido a $t_{T-1}$",
     "Misma log-loss esperada", "Varianza HAC de Newey-West"],
    ["Model Confidence Set", "Elimina modelos mientras se rechace igual capacidad predictiva", "Igual capacidad predictiva",
     "Bootstrap estacionario"],
    ["t de Nadeau-Bengio", r"$t = \bar d/\sqrt{(1/J + n_{ev}/n_{en})s^2}$", "Misma métrica esperada", "Corrige el solapamiento de entrenamientos"],
    ["Tamaño del efecto", r"$d_z = \bar d/s_d$; Cliff $\delta = P(X>Y) - P(X<Y)$ (despreciable si $|\delta| < 0{,}147$)",
     "—", "—"],
]


def fundamento_pruebas() -> list:
    return [tarjeta("Criterio de superioridad y protocolo", md(CRITERIO_SUPERIORIDAD), "flag"),
            tarjeta("Fundamento estadístico de las pruebas", [
                tabla_md(PRUEBAS, ["Prueba", "Estadístico", "Hipótesis nula", "Cómo trata la dependencia"]),
                lectura("Cada prueba responde una pregunta distinta y trata de otra forma la dependencia entre meses y semanas; "
                        "por eso la guía pide al menos tres. Se reportan también las que no rechazan."),
            ], "calculator")]


def robustez() -> list:
    sens = DT.tabla("robustez_sensibilidad")
    sem = DT.tabla("robustez_semillas")
    per = DT.tabla("robustez_perturbaciones")
    cif = DT.sustentacion()
    if sens.empty:
        return []
    lam = sens[sens.factor == "lam"]
    otros = sens[sens.factor != "lam"]
    ds = sem.groupby("pliegue")["auc"].std().mean()
    sobre = int((sem["auc"] > sem["xgb"]).sum())
    nivel = lambda f, v: ("σ = " + coma(v, 2) if f == "ruido en el logit" else f"{100 * v:.0f} %" if v < 1
                          else f"{int(v):,} filas".replace(",", "."))
    aj = per[per.modelo == "AJ-KL"].set_index(["factor", "valor"])
    gl = per[per.modelo == "Apilamiento global"].set_index(["factor", "valor"])["dauc_sin_perturbar"]
    signo = lambda v: signo4(v) if pd.notna(v) else "—"
    t = pd.DataFrame([{"Perturbación": f, "Nivel": nivel(f, v), "ΔAUC AJ-KL": signo(r.dauc_sin_perturbar),
                       "ΔAUC apilamiento global": signo(gl.get((f, v))), "AJ-KL − XGBoost": signo(r.dauc_xgb)}
                      for (f, v), r in aj.iterrows()])
    return [tarjeta("Robustez y sensibilidad (los modelos base quedan fijos; se perturba el combinador)", [
        dcc.Graph(figure=PL.sensibilidad(sens, cif.get("auc_xgb_ref", np.nan))),
        lectura(f"Entre λ = 0,01 y λ = 1000 el AUC medio cambia a lo sumo {coma(abs(lam['dauc_medio'].min()))} y γ o la profundidad, "
                f"a lo sumo {coma(abs(otros['dauc_medio'].min()))}: no hace falta un ajuste fino. Pero el encogimiento importa: en los "
                f"extremos de λ la ventaja sobre XGBoost se pierde (por encima en {int(lam.loc[lam.valor.idxmin(), 'pliegues_sobre_xgb'])} "
                f"y {int(lam.loc[lam.valor.idxmax(), 'pliegues_sobre_xgb'])} de 4 pliegues). Repetir TPE con cinco semillas cambia el "
                f"AUC en {coma(ds)} en promedio por pliegue, y el AJ-KL queda sobre XGBoost en {sobre} de {len(sem)} combinaciones."),
        html.Div("Ruido, desbalance y tamaño de muestra: ΔAUC frente al mismo modelo sin perturbar (media de k1-k4)",
                 className="fw-semibold mt-3 mb-1"),
        tabla(t, page_size=12),
        lectura("Con ruido en las probabilidades base el AJ-KL es más estable que el apilamiento global, porque promediar "
                "entre niveles amortigua el ruido; con 20 % de etiquetas invertidas pierde más, porque los nodos pequeños "
                "aprenden parte del ruido. Con 5 mil filas de historial o con 8 % de positivos queda por debajo de XGBoost: "
                "necesita un historial suficiente, limpio y con la prevalencia natural."),
    ], "shield-shaded")]


# ------------------------------------------------------------------------------- criterio 5 · errores y sobreajuste
def errores() -> list:
    sub = DT.tabla("errores_subgrupo")
    cen = DT.tabla("errores_central")
    sa = DT.tabla("sobreajuste")
    bloques = []
    if len(sa):
        g = sa.groupby("modelo")[["auc_interno", "auc_externo"]].mean()
        g["brecha"] = g["auc_interno"] - g["auc_externo"]
        aj = g.loc[[i for i in g.index if i.startswith("AJ-KL")][0]]
        bloques.append(tarjeta("Diagnóstico de sobreajuste", [
            dcc.Graph(figure=PL.sobreajuste(sa)),
            lectura(f"Si la selección de hiperparámetros sobreajustara, el AUC de la validación interna (con que se eligieron) "
                    f"estaría muy por encima del del bloque externo, que ningún modelo vio. La brecha media interno − externo va de "
                    f"{coma(g['brecha'].min())} a {coma(g['brecha'].max())} entre modelos; la del AJ-KL es {coma(aj['brecha'])}. El "
                    "combinador tiene muy poca capacidad por nodo (proposición 6) y el encogimiento evita que los nodos pequeños "
                    "memoricen su historial."),
        ], "activity"))
    if len(sub):
        s = sub.copy()
        s["subgrupo"] = s["grupo"].map({"distancia": "Distancia", "escala": "Escala"}).fillna(s["grupo"]) + ": " + s["nivel"]
        gana = int((cen["dauc"] > 0).sum()) if len(cen) else 0
        peor, mejor = s.loc[s["dauc"].idxmin()], s.loc[s["dauc"].idxmax()]
        dificil = s.loc[s["auc_ajkl"].idxmin()]
        bloques.append(dbc.Row([
            dbc.Col(tarjeta("Análisis de errores por subgrupo (prueba 2025)", [
                dcc.Graph(figure=PL.barras_delta(s, "subgrupo", "dauc", "¿Dónde gana y dónde pierde frente a XGBoost?", 400)),
                lectura(f"El AJ-KL supera a XGBoost en {int((s['dauc'] > 0).sum())} de {len(s)} subgrupos; la mayor ganancia está en "
                        f"«{mejor['subgrupo']}» ({signo4(mejor['dauc'])}) y la menor en «{peor['subgrupo']}» ({signo4(peor['dauc'])}). "
                        f"El error se concentra en los mismos lugares para los dos modelos: el subgrupo más difícil es "
                        f"«{dificil['subgrupo']}» (AUC {coma(dificil['auc_ajkl'])}, tasa de interrupción "
                        f"{coma(100 * dificil['tasa'], 1)} %)."),
                tabla(s[["subgrupo", "n", "tasa", "auc_ajkl", "auc_xgb", "dauc"]].assign(
                    tasa=lambda d: (100 * d["tasa"]).round(1), auc_ajkl=lambda d: d["auc_ajkl"].round(4),
                    auc_xgb=lambda d: d["auc_xgb"].round(4), dauc=lambda d: d["dauc"].round(4)).rename(
                    columns={"subgrupo": "Subgrupo", "n": "Filas", "tasa": "Tasa (%)", "auc_ajkl": "AUC AJ-KL",
                             "auc_xgb": "AUC XGBoost", "dauc": "ΔAUC"}), page_size=10),
            ], "pie-chart"), lg=6),
            dbc.Col(tarjeta("Por central mayorista", [
                dcc.Graph(figure=PL.barras_delta(cen, "central", "dauc", f"El AJ-KL gana en {gana} de {len(cen)} centrales", 620)),
                lectura(f"Donde más pierde es {cen.iloc[0]['central']} ({signo4(cen.iloc[0]['dauc'])}) y donde más gana, "
                        f"{cen.iloc[-1]['central']} ({signo4(cen.iloc[-1]['dauc'])}). Los pesos por central permiten que el modelo "
                        "se apoye en un modelo base distinto en cada mercado; en las centrales con pocas filas el encogimiento lo "
                        "mantiene cerca de los pesos nacionales."),
            ], "shop"), lg=6),
        ]))
    return bloques


# ------------------------------------------------------------------------------- criterio 6 · conclusiones
HALLAZGOS = r"""
1. **El techo de información es el principal límite.** Los siete modelos del curso quedan dentro de 0,02 de AUC y
   ninguna variable observada en $t$ se asocia con fuerza con $t+1$ ($|\rho| \le 0{,}20$): las mejoras posibles son de
   milésimas.
2. **Combinar con pesos locales encogidos supera de forma consistente al mejor modelo individual.** AUC externo
   0,7874 frente a 0,7867 de XGBoost, con mejor AUC-PR, Brier y log-loss; gana en 39 de 48 meses, en los cuatro años y
   en 2025 (+0,0011, DeLong por bloques de semana, p Holm < 0,001).
3. **El encogimiento es el componente que más aporta.** Sin él ($\lambda = 0$) el AUC baja 0,0017 y la log-loss sube
   0,0097; la ganancia no viene de incluir LightGBM y CatBoost ni de ver más filas.
4. **Es estable frente a sus hiperparámetros y a la semilla,** pero necesita un historial suficiente, limpio y con la
   prevalencia natural.
5. **Un optimizador que se detiene antes del óptimo es una regularización oculta:** resolver cada nodo hasta el
   certificado devolvió a $\lambda$ su papel y volvió interpretables los pesos.
"""

LIMITACIONES = r"""
**Teóricas.** Las cotas de generalización y la consistencia suponen filas independientes; el panel tiene dependencia
por corredor y por semana, así que la evidencia es la validación hacia adelante y no la cota. Las garantías son para
el combinador con los modelos base fijos.

**Empíricas.** La ventaja es consistente pero **trivial en AUC** (+0,0007, dentro de ±0,01 de equivalencia práctica);
el ECE no mejora (0,0117 frente a 0,0115 de XGBoost). En producción (622 mil filas) el AJ-KL supera a XGBoost (0,8007
frente a 0,7999), pero LightGBM y CatBoost reentrenados llegan a 0,8011, sin diferencia significativa. El diseño tuvo
cuatro versiones con 2025 ya conocido: 2025 no es una prueba ciega del diseño.
"""

FUTURO = r"""
* Reentrenar el combinador con predicciones de modelos del mismo tamaño que los de producción.
* Información anticipada de $t+1$: pronósticos de lluvias y alertas de cierres viales.
* Umbral de alerta con un costo explícito de falsos positivos y falsos negativos en lugar del F1.
"""


def conclusiones() -> list:
    R = DT.resultados()
    ext = DT.tabla("externos")
    bloques = [dbc.Alert([html.I(className="bi bi-flag me-2"), "Criterio 6 de la rúbrica (10 %): conclusiones sustentadas, "
                          "limitaciones teóricas y empíricas, capacidad de generalización y trabajo futuro."],
                         color="light", className="guia-rubrica"),
               dbc.Row([dbc.Col(tarjeta(t, md(x), i, clase="h-100 tarjeta-veredicto"), md=4, className="mb-2") for t, x, i in [
                   ("R1 · Novedad: se cumple", "Sin antecedentes de combinación local o jerárquica de clasificadores para "
                    "interrupciones de corredores en **WoS, Scopus ni ScienceDirect** (pestaña *Literatura*).", "patch-check"),
                   ("R2 · Implementación propia: se cumple", "Clase propia con numpy y scipy, **10 proposiciones demostradas**, "
                    "complejidad contrastada con tiempos y **22 pruebas unitarias**; el combinador no viene de un `import`.", "code-square"),
                   ("R3 · Superioridad: se cumple, con matices", "Mayor AUC externo medio que los siete benchmarks; **cinco pruebas** "
                    "rechazan la igualdad frente a XGBoost (Friedman, Wilcoxon-Holm, DeLong, MCS y Diebold-Mariano) y $d_z = 0{,}79$. "
                    "Matices: no rechazan Nemenyi ni Nadeau-Bengio y la ventaja es trivial en AUC (+0,0007).", "trophy"),
               ]]),
               dbc.Row([
                   dbc.Col(tarjeta("Hallazgos", md(HALLAZGOS), "lightbulb"), lg=6),
                   dbc.Col([tarjeta("Limitaciones", md(LIMITACIONES), "exclamation-triangle"),
                            tarjeta("Trabajo futuro", md(FUTURO), "signpost")], lg=6),
               ])]
    if len(ext):
        t = ext.assign(**{c: ext[c].round(4) for c in ["auc_ajkl", "auc_global", "auc_xgb", "auc_mejor_base", "dll_xgb"]})
        t["Gana a XGBoost"] = t["gana_xgb"].astype(str) + " de " + t["pliegues"].astype(str)
        t = t[["conjunto", "auc_ajkl", "auc_global", "auc_xgb", "mejor_base", "auc_mejor_base", "Gana a XGBoost", "dll_xgb"]].rename(columns={
            "conjunto": "Conjunto (UCI)", "auc_ajkl": "AUC AJ-KL", "auc_global": "AUC apilamiento global", "auc_xgb": "AUC XGBoost",
            "mejor_base": "Mejor modelo base", "auc_mejor_base": "AUC mejor base", "dll_xgb": "Δ log-loss vs XGBoost"})
        bloques.append(tarjeta("Capacidad de generalización: tres conjuntos adicionales con orden temporal y grupos anidados", [
            tabla(t, page_size=5),
            lectura(f"Mismo protocolo en otros dominios (calidad del aire, bicicletas compartidas, mercadeo bancario): el AJ-KL tiene "
                    f"mayor AUC medio que XGBoost en {int((ext['auc_ajkl'] > ext['auc_xgb']).sum())} de {len(ext)} conjuntos, pero en "
                    "Bank Marketing, con un cambio de régimen fuerte entre años, el mejor modelo es la regresión logística. En "
                    f"{int((ext['auc_global'] >= ext['auc_ajkl']).sum())} de {len(ext)} conjuntos el apilamiento global (sin jerarquía) "
                    "queda igual o algo por encima del AJ-KL completo: allí la ganancia viene de combinar, no de la jerarquía. En el "
                    "SIPSA la jerarquía aporta poco pero de forma consistente (sin ella el AUC baja 0,0004 en los cinco pliegues, "
                    "pestaña «Modelo AJ-KL»). Cuatro pliegues por conjunto no alcanzan para una prueba formal."),
        ], "globe-americas"))
    return bloques


# ------------------------------------------------------------------------------- recorrido de la sustentación
RECORRIDO = [
    ("1", "Problema y formulación matemática", 2, 10, [("Contexto: pregunta, formulación y datos", "/#formulacion")]),
    ("2", "EDA, datos y preprocesamiento", 3, 15, [("EDA", "/eda"), ("Transformaciones y fuga", "/eda#fuga")]),
    ("3", "Fundamentos teóricos del modelo", 5, 25, [("Modelo AJ-KL", "/modelos?tab=t-orig"), ("Fundamentos", "/modelos?tab=t-fund"),
                                                      ("Literatura y novedad", "/modelos?tab=t-lit")]),
    ("4", "Implementación y diseño experimental", 3, 15, [("Diseño experimental", "/modelos?tab=t-diseno")]),
    ("5", "Resultados y evaluación crítica", 3, 15, [("Líderes y métricas", "/modelos?tab=t-lideres"), ("Predicción en vivo", "/modelos?tab=t-pred"),
                                                      ("Validación estadística", "/modelos?tab=t-comp"), ("Diagnóstico y errores", "/modelos?tab=t-diag")]),
    ("6", "Conclusiones y limitaciones", 2, 10, [("Conclusiones", "/modelos?tab=t-concl")]),
    ("7", "Dominio teórico y preguntas", 2, 10, [("Informe técnico", DT.LIBRO), ("Código", DT.REPO)]),
]


def recorrido() -> dbc.Card:
    filas = []
    for n, criterio, minutos, peso, enlaces in RECORRIDO:
        links = []
        for i, (texto, href) in enumerate(enlaces):
            externo = href.startswith("http")
            links += ([" · "] if i else []) + [html.A(texto, href=href, target="_blank" if externo else None)
                                               if externo or "#" in href else dcc.Link(texto, href=href)]
        filas.append(html.Tr([html.Td(html.Span(n, className="badge rounded-pill")), html.Td(criterio),
                              html.Td(f"{minutos} min"), html.Td(f"{peso} %"), html.Td(links)]))
    return dbc.Card([dbc.CardHeader([html.I(className="bi bi-easel2", **{"aria-hidden": "true"}),
                                     "Recorrido de la sustentación (rúbrica: 7 criterios, 20 minutos)"]),
                     dbc.CardBody(dbc.Table([html.Thead(html.Tr([html.Th(""), html.Th("Criterio"), html.Th("Tiempo"),
                                                                 html.Th("Peso"), html.Th("Dónde está en el tablero")])),
                                             html.Tbody(filas)], hover=True, responsive=True, size="sm", className="mb-0 recorrido"))],
                    className="recorrido-card")
