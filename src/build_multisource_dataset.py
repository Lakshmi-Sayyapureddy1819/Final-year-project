from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr

from juvenile_risk_utils import attach_maturity_reference, maturity_risk_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATASET_PATH = PROJECT_ROOT / "data" / "real_world_training_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "multisource_training_data.csv"

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PHYSICS_NC_PATH = EXTERNAL_DIR / "copernicus_physics.nc"
CHL_NC_PATH = EXTERNAL_DIR / "copernicus_chlorophyll.nc"
PFZ_CSV_PATH = EXTERNAL_DIR / "incois_pfz.csv"
MATURITY_CSV_PATH = EXTERNAL_DIR / "fishbase_maturity.csv"


STATE_METADATA = {
    "West Bengal": {"lat": 21.62, "lon": 87.52},
    "Odisha": {"lat": 19.81, "lon": 85.84},
    "Andhra Pradesh": {"lat": 16.99, "lon": 82.25},
    "Tamil Nadu": {"lat": 13.08, "lon": 80.27},
    "Puducherry": {"lat": 11.94, "lon": 79.83},
    "Kerala": {"lat": 9.93, "lon": 76.27},
    "Karnataka": {"lat": 14.80, "lon": 74.13},
    "Goa": {"lat": 15.30, "lon": 74.12},
    "Maharashtra": {"lat": 19.08, "lon": 72.88},
    "Gujarat": {"lat": 21.64, "lon": 69.61},
}


def _find_first_name(candidates: Iterable[str], available: Iterable[str]) -> str | None:
    available_set = {item for item in available}
    for candidate in candidates:
        if candidate in available_set:
            return candidate
    return None


def _subset_to_point(dataset: xr.Dataset, latitude: float, longitude: float) -> xr.Dataset:
    lat_name = _find_first_name(["latitude", "lat", "nav_lat"], dataset.coords)
    lon_name = _find_first_name(["longitude", "lon", "nav_lon"], dataset.coords)
    if lat_name is None or lon_name is None:
        raise ValueError("Dataset missing latitude/longitude coordinates")

    return dataset.sel({lat_name: latitude, lon_name: longitude}, method="nearest")


def _yearly_frame_from_series(frame: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
    yearly = (
        frame.groupby("Year")[column_name]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{prefix}_Mean",
                "min": f"{prefix}_Min",
                "max": f"{prefix}_Max",
                "std": f"{prefix}_Std",
            }
        )
    )
    return yearly


def load_copernicus_physics_features() -> pd.DataFrame | None:
    if not PHYSICS_NC_PATH.exists():
        return None

    dataset = xr.open_dataset(PHYSICS_NC_PATH)
    time_name = _find_first_name(["time"], dataset.coords)
    sal_name = _find_first_name(["so", "sos", "salinity", "s"], dataset.data_vars)
    temp_name = _find_first_name(["thetao", "tos", "temperature"], dataset.data_vars)
    mld_name = _find_first_name(["mlotst", "mld", "mixed_layer_thickness"], dataset.data_vars)

    frames: list[pd.DataFrame] = []
    for state, meta in STATE_METADATA.items():
        point = _subset_to_point(dataset, meta["lat"], meta["lon"])
        base = pd.DataFrame({time_name: pd.to_datetime(point[time_name].values)})
        base["Year"] = base[time_name].dt.year
        base["State"] = state

        merged = base[["Year", "State"]].drop_duplicates().sort_values("Year")
        for var_name, prefix in [(sal_name, "Salinity"), (temp_name, "Temp"), (mld_name, "MLD")]:
            if var_name is None:
                continue
            values = point[var_name]
            while values.ndim > 1:
                values = values.isel({values.dims[-1]: 0})
            temp = pd.DataFrame({time_name: pd.to_datetime(point[time_name].values), var_name: values.values})
            temp["Year"] = temp[time_name].dt.year
            merged = merged.merge(_yearly_frame_from_series(temp, var_name, prefix), on="Year", how="left")

        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def load_copernicus_chlorophyll_features() -> pd.DataFrame | None:
    if not CHL_NC_PATH.exists():
        return None

    dataset = xr.open_dataset(CHL_NC_PATH)
    time_name = _find_first_name(["time"], dataset.coords)
    chl_name = _find_first_name(["chl", "CHL", "chla"], dataset.data_vars)
    if time_name is None or chl_name is None:
        raise ValueError("Chlorophyll file missing time or chlorophyll variable")

    frames: list[pd.DataFrame] = []
    for state, meta in STATE_METADATA.items():
        point = _subset_to_point(dataset, meta["lat"], meta["lon"])
        values = point[chl_name]
        while values.ndim > 1:
            values = values.isel({values.dims[-1]: 0})

        temp = pd.DataFrame({time_name: pd.to_datetime(point[time_name].values), chl_name: values.values})
        temp["Year"] = temp[time_name].dt.year
        yearly = _yearly_frame_from_series(temp, chl_name, "Chlorophyll")
        yearly["State"] = state
        frames.append(yearly)

    return pd.concat(frames, ignore_index=True)


def load_pfz_features() -> pd.DataFrame | None:
    if not PFZ_CSV_PATH.exists():
        return None

    frame = pd.read_csv(PFZ_CSV_PATH)
    required = {"State", "Date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{PFZ_CSV_PATH} must contain at least columns: {sorted(required)}")

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Year"] = frame["Date"].dt.year
    if "PFZ_Count" not in frame.columns:
        frame["PFZ_Count"] = 1

    if "Species" in frame.columns:
        frame = attach_maturity_reference(frame)

    if "Observed_Length_cm" in frame.columns:
        frame["Observed_Length_cm"] = pd.to_numeric(frame["Observed_Length_cm"], errors="coerce")
    if "Maturity_Length_cm" in frame.columns:
        frame["Maturity_Length_cm"] = pd.to_numeric(frame["Maturity_Length_cm"], errors="coerce")
    if {"Observed_Length_cm", "Maturity_Length_cm"}.issubset(frame.columns):
        frame["Exact_Maturity_Score"] = [
            maturity_risk_score(
                float(observed_length) if pd.notna(observed_length) else None,
                float(maturity_length) if pd.notna(maturity_length) else None,
            )
            for observed_length, maturity_length in zip(frame["Observed_Length_cm"], frame["Maturity_Length_cm"])
        ]

    aggregations = {"PFZ_Count": "sum"}
    if "Distance_km" in frame.columns:
        aggregations["Distance_km"] = "mean"
    if "Depth_m" in frame.columns:
        aggregations["Depth_m"] = "mean"
    if "Observed_Length_cm" in frame.columns:
        aggregations["Observed_Length_cm"] = "mean"
    if "Maturity_Length_cm" in frame.columns:
        aggregations["Maturity_Length_cm"] = "mean"
    if "Exact_Maturity_Score" in frame.columns:
        aggregations["Exact_Maturity_Score"] = "mean"

    grouped = frame.groupby(["State", "Year"]).agg(aggregations).reset_index()
    grouped = grouped.rename(
        columns={
            "PFZ_Count": "PFZ_Observations",
            "Distance_km": "PFZ_Mean_Distance_km",
            "Depth_m": "PFZ_Mean_Depth_m",
        }
    )
    return grouped


def load_maturity_lookup() -> pd.DataFrame | None:
    if not MATURITY_CSV_PATH.exists():
        return None

    frame = pd.read_csv(MATURITY_CSV_PATH)
    if not {"Species", "Maturity_Length_cm"}.issubset(frame.columns):
        raise ValueError(f"{MATURITY_CSV_PATH} must contain Species and Maturity_Length_cm columns")
    return frame


def main() -> None:
    if not BASE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing base dataset: {BASE_DATASET_PATH}")

    dataset = pd.read_csv(BASE_DATASET_PATH)
    source_notes = ["CMFRI", "NOAA OISST"]

    physics = load_copernicus_physics_features()
    if physics is not None:
        dataset = dataset.merge(physics, on=["State", "Year"], how="left")
        source_notes.append("Copernicus Physics")

    chlorophyll = load_copernicus_chlorophyll_features()
    if chlorophyll is not None:
        dataset = dataset.merge(chlorophyll, on=["State", "Year"], how="left")
        source_notes.append("Copernicus Chlorophyll")

    pfz = load_pfz_features()
    if pfz is not None:
        dataset = dataset.merge(pfz, on=["State", "Year"], how="left")
        source_notes.append("INCOIS PFZ")

    maturity = load_maturity_lookup()
    if maturity is not None:
        dataset.attrs["maturity_lookup_rows"] = len(maturity)
        source_notes.append("FishBase Maturity")

    dataset["Integrated_Sources"] = " + ".join(source_notes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved multisource dataset to {OUTPUT_PATH}")
    print(f"Rows: {len(dataset)} | Columns: {len(dataset.columns)}")
    print("Sources:", dataset["Integrated_Sources"].iloc[0])
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
