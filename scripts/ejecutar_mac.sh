#!/bin/sh
# Ejecuta un script de Python del proyecto en macOS sin instalar libomp con Homebrew.
# XGBoost y LightGBM buscan OpenMP en /opt/homebrew/opt/libomp; scikit-learn trae su propio libomp en
# sklearn/.dylibs, así que basta con ponerlo en DYLD_LIBRARY_PATH.
# Uso: scripts/ejecutar_mac.sh scripts/25_robustez_ajkl.py k4 semillas
#      PY=/ruta/a/python scripts/ejecutar_mac.sh app.py
PY=${PY:-$HOME/.venvs/entregable3/bin/python}
SKLEARN=$("$PY" -c "import os, sklearn; print(os.path.dirname(sklearn.__file__))") || exit 1
export DYLD_LIBRARY_PATH="$SKLEARN/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec "$PY" "$@"
