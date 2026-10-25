"""Prueba de ida y vuelta de los artefactos serializados (guía, sección 4.2).

Cada modelo cargado desde artifacts/ debe reproducir las predicciones que produjo el modelo en memoria
antes de serializarse, sobre la misma muestra fija de 2.000 corredor-semanas de 2025. En la plataforma
donde se serializaron (la de metadata.json) la igualdad es exacta, bit a bit (scripts/13_artefactos.py la
verifica y la guarda en metadata.json). En otra plataforma (otro procesador, Python u OpenMP, como Render)
el orden de las sumas en punto flotante puede cambiar el último dígito: allí se exige |Δ| ≤ 1e-6.
"""
import platform
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
ART = RAIZ / "artifacts"
MODELOS = sorted(p.name for p in ART.iterdir() if (p / "modelo.joblib").exists()) if ART.exists() else []
CLAVES = {"librerias", "semilla", "hiperparametros", "esquema_entrada", "metricas_bucle_externo", "metricas_prueba_2025"}


@pytest.fixture(scope="module")
def muestra():
    return pd.read_parquet(ART / "muestra_referencia.parquet")


@pytest.mark.skipif(not MODELOS, reason="no hay artefactos")
@pytest.mark.parametrize("slug", MODELOS)
def test_predicciones_identicas(slug, muestra):
    import config as C
    modelo = joblib.load(ART / slug / "modelo.joblib")
    p = modelo.predict_proba(muestra[C.X_COLS])[:, 1]
    ref = np.load(ART / slug / "pred_referencia.npy")
    meta = json.loads((ART / slug / "metadata.json").read_text())
    misma_plataforma = (meta["librerias"].get("plataforma") == platform.platform()
                        and meta["librerias"].get("python") == platform.python_version())
    if misma_plataforma:
        assert np.array_equal(p, ref)
    else:
        np.testing.assert_allclose(p, ref, rtol=0, atol=1e-6)


@pytest.mark.skipif(not MODELOS, reason="no hay artefactos")
@pytest.mark.parametrize("slug", MODELOS)
def test_metadata_completa(slug):
    import config as C
    meta = json.loads((ART / slug / "metadata.json").read_text())
    assert CLAVES <= set(meta)
    assert [c["nombre"] for c in meta["esquema_entrada"]] == C.X_COLS
    assert meta["semilla"] == C.SEMILLA
    assert meta["prueba_ida_y_vuelta"]["identicas"]
