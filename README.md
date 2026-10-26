# Entregable 3 · Interrupciones en corredores de abastecimiento (SIPSA)

Proyecto final de Machine Learning, Doctorado en Ingeniería. Camilo Molina Guerrero y Lina Margarita Buelvas.

* **Informe (Jupyter Book):** <https://linamarbg.github.io/Entregable3MachineLearningProject/>
* **Dashboard (Render):** <https://sipsa-corredores-e3-2uw9.onrender.com>

¿Se puede anticipar, con la información disponible al cierre de una semana, qué corredores municipio → central
mayorista tendrán la semana siguiente menos de la mitad de su abastecimiento habitual? El repositorio contiene
el panel de 744.921 corredor-semanas (SIPSA-A, DANE, 2018-2025), los artefactos de los siete benchmarks del
curso, LightGBM, CatBoost y el modelo original **AJ-KL** (apilamiento jerárquico con encogimiento KL y olvido
temporal, `src/original_model.py`), el dashboard en Dash y el libro.

## Estructura

| Carpeta | Contenido |
|---|---|
| `app.py`, `pages/`, `assets/` | Dashboard multipágina (Contexto, EDA, Modelos) con mapa de corredores, ilustraciones propias (`assets/img/`) y pestañas que siguen la rúbrica de la sustentación |
| `src/` | Modelo original, carga de datos, figuras, artefactos, explicación SHAP y textos de la sustentación (`sustentacion.py`) |
| `scripts/` | Pipeline numerado (ver capítulo 9 del libro); `25_robustez_ajkl.py` (robustez del modelo original), `26_mapa_dashboard.py` (datos del mapa) y `28_sustentacion_dashboard.py` (tablas de las pestañas de la sustentación) |
| `artifacts/` | 10 modelos serializados con `metadata.json` y prueba de ida y vuelta |
| `data/` | Panel, resultados de la validación anidada, revisión de literatura, datos del dashboard |
| `cuadernos/`, `book/` | Fuentes y Jupyter Book |
| `tests/` | Pruebas unitarias (`pytest tests/`) |

## Ejecutar

```bash
pip install -r requirements.txt
python app.py
```

El entorno completo de los experimentos está en `requirements-dev.txt`; el orden de ejecución, en el capítulo 9.

En macOS sin `brew install libomp`, crear el entorno fuera de iCloud y usar el envoltorio que carga el OpenMP de
scikit-learn:

```bash
python3 -m venv ~/.venvs/entregable3
~/.venvs/entregable3/bin/pip install -r requirements-dev.txt
scripts/ejecutar_mac.sh app.py
```

Para reconstruir el libro con ese entorno: `PY=~/.venvs/entregable3/bin/python scripts/24_libro.sh`.
