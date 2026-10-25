"""Datos del mapa de corredores del dashboard (data/dashboard/).

  mapa_corredores.parquet  un registro por corredor (municipio de origen → central): coordenadas de la cabecera
                           municipal de origen y de la central (DIVIPOLA), departamento, PDET, distancia, semanas
                           observadas, tasa de interrupción y abastecimiento medio (t/día, promedio de 8 semanas)
  mapa_centrales.parquet   un registro por central: coordenadas, corredores, tasa de interrupción y toneladas de 2025
Es descriptivo (panel completo 2018-2025): no entra a ningún modelo.
Uso: python scripts/26_mapa_dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402

SAL = RAIZ / "data" / "dashboard"


def main():
    p = D.cargar_panel()
    corr = (p.groupby(["mercado", "cod_mpio"], observed=True)
            .agg(mpio_origen=("mpio_origen", "first"), depto_origen=("depto_origen", "first"),
                 lat_o=("lat_o", "first"), lon_o=("lon_o", "first"), lat_m=("lat_m", "first"), lon_m=("lon_m", "first"),
                 pdet=("pdet_origen", "max"), dist_km=("dist_km", "first"), semanas=(C.OBJ_CLF, "size"),
                 tasa=(C.OBJ_CLF, "mean"), t_dia=("base8", "mean"),
                 grupo=("grupo_dom", lambda s: s.mode().iat[0] if len(s.mode()) else ""))
            .reset_index())
    ton = pd.read_parquet(SAL / "toneladas_mercado_anio.parquet")
    ton = ton[ton["anio"] == ton["anio"].max()].set_index("mercado")["toneladas"]
    cent = (corr.groupby("mercado")
            .agg(lat=("lat_m", "first"), lon=("lon_m", "first"), corredores=("cod_mpio", "size"))
            .join(p.groupby("mercado")[C.OBJ_CLF].mean().rename("tasa"))
            .join(ton.rename("toneladas_ultimo_anio")).reset_index())
    corr.to_parquet(SAL / "mapa_corredores.parquet", index=False)
    cent.to_parquet(SAL / "mapa_centrales.parquet", index=False)
    print(f"{len(corr)} corredores, {len(cent)} centrales; tasa media por corredor {corr['tasa'].mean():.3f}")


if __name__ == "__main__":
    main()
