# Cumplimiento del Entregable 3 frente a la guía

Revisión del 24 de octubre de 2026, punto por punto contra *Proyecto Final: Tercer Entregable* (guía técnica y
rúbrica). Estado: ✅ cumple · 🔧 cumple después de esta revisión · ⏳ depende de una acción externa.

## Bloque A. Correcciones del Entregable 2 (sección 3)

| Requisito | Estado | Evidencia |
|---|---|---|
| Tabla de trazabilidad con observación, acción, ubicación y evidencia | ✅ | Capítulo 1; `data/resultados/trazabilidad.csv` |
| Todas las observaciones del docente al Entregable 2 resueltas | 🔧 | No se contaba con la retroalimentación escrita del Entregable 2; la fila del capítulo 1 lo declara y remite a la revisión propia del Entregable 2 (12 filas). Si llega antes de la entrega, agregar una fila por observación |
| Resultados recalculados y consistentes; cambios de ranking reportados | ✅ | Sección 1.1 y columna «Resultados recalculados» |

## Bloque B. Mejores combinaciones y artefactos (sección 4)

| Requisito | Estado | Evidencia |
|---|---|---|
| Mejor combinación por modelo según la métrica media del bucle externo; mejor global en los finalistas de Friedman-Nemenyi | ✅ | Sección 3.1; `seleccion_mejores.csv` (columnas `finalista_nemenyi`, `mejor_global`, `motivo`) |
| Tabla maestra con media ± desviación estándar y motivo | ✅ | Sección 3.1 |
| Reentrenamiento con todo el entrenamiento y serialización con joblib en `artifacts/` | ✅ | 10 artefactos (7 del curso, LightGBM, CatBoost y AJ-KL) |
| `metadata.json` con librerías, semilla, hiperparámetros, esquema de entrada (tipos y rangos) y métricas | ✅ | `artifacts/*/metadata.json`; prueba `test_metadata_completa` |
| Prueba de ida y vuelta | 🔧 | Exacta en la plataforma de serialización; ahora también pasa en otras plataformas (como Render) con tolerancia 1e-6 |
| Tamaño de los artefactos controlado | ✅ | Unos 23 MB en total (`13b_aligerar.py`) |

## Bloque C. Modelo original (sección 5)

| Requisito | Estado | Evidencia |
|---|---|---|
| R1. Novedad: revisión en ≥ 3 bases, ≥ 5 cadenas, matriz de novedad, declaración y trabajos cercanos | ✅ | Capítulo 4 (WoS, Scopus y ScienceDirect) |
| Tabla de diferenciación con 5 a 10 métodos afines por familia | ✅ | Capítulo 4 y pestaña «Modelo original» del dashboard |
| R2. Clase propia (`BaseEstimator` + `ClassifierMixin`) con `fit`, `predict`, `predict_proba`, solo numpy y scipy | 🔧 | `src/original_model.py`. Se corrigió el orden de herencia: con el orden anterior, scikit-learn ≥ 1.6 no la reconocía como clasificador |
| Pseudocódigo formal y análisis de complejidad contrastado con tiempos | ✅ | Secciones 5.4, 5.5 y 5.7 |
| Sustento teórico verificable | 🔧 | Proposiciones 1 a 6, más las nuevas 7 a 10 (regla propia, cota de generalización, calibración monótona y consistencia), tabla de supuestos y tabla de teoría a evidencia (sección 5.3) |
| Pruebas unitarias: aprende en datos sintéticos, reproducibilidad, entradas inválidas, `clone` | 🔧 | 22 pruebas del modelo (dos nuevas) y 83 en total, todas pasan |
| Hiperparámetros propios optimizados con TPE dentro de la validación anidada | ✅ | Sección 5.6 y `scripts/22_ajkl.py` |
| Estudio de ablación | ✅ | Sección 5.8 |
| Interpretabilidad (SHAP) frente al mejor benchmark | ✅ | Sección 5.10 y pestaña «Diagnóstico» |
| Al menos tres conjuntos adicionales, mismas particiones y presupuesto | ✅ | Sección 5.11 (Beijing PM2.5, Bike Sharing, Bank Marketing) |
| Comparación con métodos recientes del estado del arte | ✅ | LightGBM y CatBoost (secciones 3 y 6.8) |
| Robustez: sensibilidad, semillas, ruido, desbalance, tamaño de muestra y análisis de errores | 🔧 | **Faltaba para el AJ-KL** (solo existía para el EJUL). Nueva sección 5.13 con `scripts/25_robustez_ajkl.py` (cinco experimentos en los pliegues k1 a k4, 20 archivos en `data/resultados/robustez_ajkl/`); el análisis de errores ya estaba en la sección 5.12 |
| Reproducibilidad completa | ✅ | Capítulo 9, semillas y `requirements*.txt` |

## Bloque D. Verificación estadística (sección 6)

| Requisito | Estado | Evidencia |
|---|---|---|
| Métrica media del bucle externo mejor que la de cada benchmark | ✅ | AJ-KL 0,7874 frente a 0,7867 de XGBoost (sección 6.2) |
| ≥ 3 pruebas con p ajustado < 0,05, al menos una global o con control de data snooping | ✅ | Cinco rechazan frente a XGBoost: Friedman con permutación, Wilcoxon-Holm, DeLong con bloques de semana, MCS y Diebold-Mariano |
| Tamaño del efecto no despreciable | ✅ | d de Cohen pareada 0,79; dominancia pareada 0,62 |
| Supuestos verificados y justificación de cada prueba | 🔧 | Sección 6.3 (supuestos) y nueva sección 6.1 con el estadístico, la distribución de referencia y el tratamiento de la dependencia de cada prueba |
| Tabla resumen con media ± desviación, IC bootstrap y tiempos de entrenamiento e inferencia | 🔧 | **Faltaban el IC y los tiempos.** Nueva tabla en la sección 6.2 (bootstrap estacionario de 48 meses) |
| Tabla de pruebas con H0, supuestos, estadístico, p crudo y ajustado, efecto y conclusión | ✅ | Sección 6.4 |
| Diagrama de diferencia crítica con el modelo original | ✅ | Sección 6.5 y dashboard |
| Discusión honesta de lo que no rechaza | ✅ | Sección 6.10 y capítulo 8 |

## Bloque E. Dashboard (sección 7)

| Requisito | Estado | Evidencia |
|---|---|---|
| Multipágina (`use_pages=True`) con Contexto, EDA y Modelos y la estructura de carpetas pedida | ✅ | `app.py`, `pages/`, `src/`, `artifacts/`, `assets/`, `tests/`, `book/` |
| Contexto: problema con cifra citada, pregunta, datos, diccionario, metodología, limitaciones y ética | ✅ | Página Contexto, ahora con ilustración propia y mapa de corredores |
| EDA: filtros globales con callbacks, objetivo, univariadas, explorador bivariado, correlación, faltantes, atípicos, series de tiempo e interpretación por figura | ✅ | Página EDA |
| Modelos: líderes, predicción en vivo con validación y SHAP local, diagnóstico, SHAP global, comparación estadística y subsección del modelo original | ✅ | Página Modelos (cinco pestañas) |
| Diseño: paleta ≤ 5 colores, tipografía, barra fija, plantilla Plotly, responsividad, rendimiento, robustez y accesibilidad | 🔧 | Nueva identidad visual del tema: verde montaña, azul río, ámbar cosecha, rojo tomate y gris piedra; cinco ilustraciones SVG propias con texto alternativo; menú que se colapsa en tableta |

## Bloque F. Despliegue en Render (sección 8)

| Requisito | Estado | Evidencia |
|---|---|---|
| `requirements.txt` con versiones exactas, `render.yaml`, `gunicorn app:server` | ✅ | Raíz del repositorio |
| URL pública funcional con los cambios de esta revisión | 🔧 | Publicado desde la cuenta de Lina (`linamarbg`) en <https://sipsa-corredores-e3-2uw9.onrender.com> (servicio creado con `render.yaml`; se redespliega con cada `git push` a `main`) |

## Bloque G. Jupyter Book (sección 9)

| Requisito | Estado | Evidencia |
|---|---|---|
| Capítulos 0 a 9 del cuadro 4, `_config.yml`, `_toc.yml`, `references.bib` | ✅ | `book/` (seis referencias nuevas en `references.bib`) |
| Celdas ejecutables, figuras con título y ejes, ecuaciones numeradas | ✅ | Cuadernos reejecutados en esta revisión |
| Construcción sin errores y publicación | 🔧 ⏳ | Se reconstruyó en local; falta publicarlo en GitHub Pages (`ghp-import`) |
| Declaración de autoría y uso de IA, firmada | ⏳ | Sección 8.5, actualizada con esta revisión; faltan las firmas |

## Acciones pendientes de los autores

1. Si llegan las observaciones del docente al Entregable 2, agregar una fila por observación en la tabla de trazabilidad.
2. Código y dashboard: hechos. El repositorio es <https://github.com/linamarbg/Entregable3MachineLearningProject> (el de
   Camilo queda como remoto `camilo`) y Render redespliega el dashboard con cada `git push` a `main`.
3. Publicar el libro: `~/.venvs/entregable3/bin/ghp-import -n -p -f book/_build/html` y, en GitHub, Settings → Pages →
   Deploy from a branch → `gh-pages` / `(root)`. Queda en <https://linamarbg.github.io/Entregable3MachineLearningProject/>.
4. Firmar la declaración de la sección 8.5.
5. Actualizar las diapositivas de la sustentación con la sección 5.13, las proposiciones 7 a 10 y el mapa del dashboard.
