#!/bin/zsh
# Arma y ejecuta los cuadernos del libro y lo compila (Jupyter Book 1.x).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/Users/cjmolina/.venvs/ml-e2/bin/python}
# el kernel «ml-e2» que usan los cuadernos apunta al Python elegido
$PY -m ipykernel install --user --name ml-e2 --display-name "Python (ml-e2)" > /dev/null
# en macOS sin libomp de Homebrew, XGBoost y LightGBM usan el OpenMP de scikit-learn (lo hereda el kernel)
export DYLD_LIBRARY_PATH="$($PY -c 'import os, sklearn; print(os.path.dirname(sklearn.__file__))')/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
# el capítulo 5 une la teoría (markdown) y la parte práctica (código)
$PY - <<'P'
from pathlib import Path
t = Path("cuadernos/05_teoria.md").read_text().splitlines()
md = "# %% [markdown]\n" + "\n".join(("# " + l) if l else "#" for l in t) + "\n\n"
Path("cuadernos/05_modelo_original.py").write_text(md + Path("cuadernos/05_practica.py").read_text())
P
CAPS=(01_correcciones 02_datos 03_benchmarks 04_literatura 05_modelo_original 06_validacion 09_reproducibilidad)
(( $# )) && CAPS=("$@")
for c in $CAPS; do
  echo "== $c"; $PY src/py2nb.py cuadernos/$c.py book/$c.ipynb --ejecutar
done
"$(dirname $PY)/jupyter-book" build book/
