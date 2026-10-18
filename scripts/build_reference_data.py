"""
scripts/build_reference_data.py
-------------------------------
Builds data/reference/neighborhoods.csv, the one place that ties together
how each source identifies a neighborhood. Run once; the output is committed.

    pip install -r scripts/requirements.txt
    python scripts/build_reference_data.py \\
        --mat data/raw/Rearranged_Data.mat \\
        --dengue data/raw/Dados_Modelagem.xlsx \\
        --ibge path/to/neighborhoods_2010.parquet

Inputs
- Rearranged_Data.mat, table ReducedTable (a MATLAB table object, readable
  with mat-io): the row order of EggIndice_Agregated, plus each
  neighborhood's municipal code and health district.
- Dados_Modelagem.xlsx, sheet 'Casos de Dengue': the canonical names used
  in the database.
- IBGE 2010 neighborhood boundaries (malha de bairros), as republished by
  Ipea's geobr project:
  https://github.com/ipea/geobr_prep_data/releases/download/v2.0.0/neighborhoods_2010.parquet

Output columns
  name               canonical name (dengue sheet spelling)
  code               municipal code from ReducedTable
  district           health district from ReducedTable
  egg_row            1-based row of this neighborhood in EggIndice_Agregated
  ibge_code, ibge_name
  latitude, longitude  centroid of the IBGE 2010 polygon (SIRGAS 2000)
  coord_method       'centroid', or 'representative_point' if the centroid
                     falls outside the polygon
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from matio import load_from_mat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.names import name_key  # noqa: E402

UTM_25S = "EPSG:31985"      # SIRGAS 2000 / UTM zone 25S — metric CRS for Natal
SIRGAS_2000 = "EPSG:4674"
NATAL_IBGE = 2408102


def _scalar(cell) -> str:
    """ReducedTable cells come back as 1-element arrays."""
    return str(cell[0] if not isinstance(cell, str) and hasattr(cell, "__len__") else cell).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=Path, required=True)
    ap.add_argument("--dengue", type=Path, required=True)
    ap.add_argument("--ibge", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reference" / "neighborhoods.csv")
    args = ap.parse_args()

    # 1. Egg-data row order, codes and districts from the MATLAB table
    table = load_from_mat(str(args.mat), variable_names=["ReducedTable"])["ReducedTable"]
    egg = pd.DataFrame({
        "egg_row":  range(1, len(table) + 1),
        "mat_name": [_scalar(v) for v in table["Neighborhood_Name"]],
        "code":     [_scalar(v) for v in table["Neighborhood_Cod"]],
        "district": [_scalar(v) for v in table["District"]],
    })

    # 2. Canonical names from the dengue sheet
    names = [str(n).strip() for n in pd.read_excel(args.dengue, sheet_name="Casos de Dengue", index_col=0).index]
    canonical = {name_key(n): n for n in names}

    # 3. IBGE 2010 polygons for Natal → centroids
    g = gpd.read_parquet(args.ibge)
    g = g[g["code_muni"] == NATAL_IBGE].to_crs(UTM_25S)
    centroid = g.geometry.centroid
    inside = centroid.within(g.geometry)
    point = centroid.where(inside, g.geometry.representative_point())
    pts = gpd.GeoSeries(point, crs=UTM_25S).to_crs(SIRGAS_2000)
    ibge = pd.DataFrame({
        "key":          [name_key(n) for n in g["name_neighborhood"]],
        "ibge_code":    g["code_neighborhood"].astype("int64").astype(str).values,
        "ibge_name":    g["name_neighborhood"].values,
        "latitude":     pts.y.round(6).values,
        "longitude":    pts.x.round(6).values,
        "coord_method": ["centroid" if ok else "representative_point" for ok in inside],
    })

    egg["key"] = [name_key(n) for n in egg["mat_name"]]
    out = egg.merge(ibge, on="key", how="left")
    out["name"] = out["key"].map(canonical)

    problems = []
    if out["name"].isna().any():
        problems.append(f"MATLAB names without a dengue-sheet match: {out.loc[out['name'].isna(), 'mat_name'].tolist()}")
    if out["ibge_code"].isna().any():
        problems.append(f"No IBGE polygon for: {out.loc[out['ibge_code'].isna(), 'mat_name'].tolist()}")
    if set(out["name"]) != set(names) or len(out) != len(names):
        problems.append("Neighborhood sets differ between the MATLAB table and the dengue sheet")
    if problems:
        raise SystemExit("\n".join(problems))

    cols = ["name", "code", "district", "egg_row", "ibge_code", "ibge_name", "latitude", "longitude", "coord_method"]
    out = out[cols].sort_values("name")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    digest = hashlib.sha256(args.ibge.read_bytes()).hexdigest()
    print(f"Wrote {len(out)} neighborhoods to {args.out}")
    print(f"IBGE file sha256: {digest}")
    print(f"Coordinates: {(out.coord_method == 'centroid').sum()} centroids, "
          f"{(out.coord_method != 'centroid').sum()} representative points")


if __name__ == "__main__":
    main()
