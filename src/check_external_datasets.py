from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from field_data_store import exact_ready_row_count


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

FILES = {
    "copernicus_physics.nc": EXTERNAL_DIR / "copernicus_physics.nc",
    "copernicus_chlorophyll.nc": EXTERNAL_DIR / "copernicus_chlorophyll.nc",
    "incois_pfz.csv": EXTERNAL_DIR / "incois_pfz.csv",
    "fishbase_maturity.csv": EXTERNAL_DIR / "fishbase_maturity.csv",
}


def check_netcdf(path: Path) -> None:
    dataset = xr.open_dataset(path)
    print(f"- {path.name}: FOUND")
    print(f"  coords: {list(dataset.coords)}")
    print(f"  variables: {list(dataset.data_vars)[:12]}")


def check_csv(path: Path, required_columns: list[str]) -> None:
    frame = pd.read_csv(path)
    print(f"- {path.name}: FOUND")
    print(f"  rows: {len(frame)}")
    print(f"  columns: {list(frame.columns)}")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        print(f"  missing required columns: {missing}")
    else:
        print("  required columns: OK")


def main() -> None:
    print(f"Checking external datasets in {EXTERNAL_DIR}")

    physics_path = FILES["copernicus_physics.nc"]
    if physics_path.exists():
        check_netcdf(physics_path)
    else:
        print("- copernicus_physics.nc: MISSING")

    chlorophyll_path = FILES["copernicus_chlorophyll.nc"]
    if chlorophyll_path.exists():
        check_netcdf(chlorophyll_path)
    else:
        print("- copernicus_chlorophyll.nc: MISSING")

    pfz_path = FILES["incois_pfz.csv"]
    if pfz_path.exists():
        check_csv(pfz_path, ["State", "Date"])
        print("  optional exact-juvenile columns: ['Species', 'Observed_Length_cm', 'Maturity_Length_cm']")
        print(f"  exact-ready rows: {exact_ready_row_count(pd.read_csv(pfz_path))}")
    else:
        print("- incois_pfz.csv: MISSING")

    maturity_path = FILES["fishbase_maturity.csv"]
    if maturity_path.exists():
        check_csv(maturity_path, ["Species", "Maturity_Length_cm"])
    else:
        print("- fishbase_maturity.csv: MISSING")


if __name__ == "__main__":
    main()
