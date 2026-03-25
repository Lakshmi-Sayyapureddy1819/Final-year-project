from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDINGS_PATH = PROJECT_ROOT / "data" / "cmfri_state_landings.csv"
SST_PATH = PROJECT_ROOT / "data" / "indian_sst.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "real_world_training_data.csv"

STATE_METADATA = {
    "West Bengal": {"Latitude": 21.62, "Longitude": 87.52},
    "Odisha": {"Latitude": 19.81, "Longitude": 85.84},
    "Andhra Pradesh": {"Latitude": 16.99, "Longitude": 82.25},
    "Tamil Nadu": {"Latitude": 13.08, "Longitude": 80.27},
    "Puducherry": {"Latitude": 11.94, "Longitude": 79.83},
    "Kerala": {"Latitude": 9.93, "Longitude": 76.27},
    "Karnataka": {"Latitude": 14.80, "Longitude": 74.13},
    "Goa": {"Latitude": 15.30, "Longitude": 74.12},
    "Maharashtra": {"Latitude": 19.08, "Longitude": 72.88},
    "Gujarat": {"Latitude": 21.64, "Longitude": 69.61},
}


def nearest_grid_point(sst_frame: pd.DataFrame, latitude: float, longitude: float) -> tuple[float, float]:
    coordinates = sst_frame[["lat", "lon"]].drop_duplicates().copy()
    coordinates["distance"] = (coordinates["lat"] - latitude) ** 2 + (coordinates["lon"] - longitude) ** 2
    nearest = coordinates.nsmallest(1, "distance").iloc[0]
    return float(nearest["lat"]), float(nearest["lon"])


def yearly_sst_features(sst_frame: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    nearest_lat, nearest_lon = nearest_grid_point(sst_frame, latitude, longitude)
    subset = sst_frame[(sst_frame["lat"] == nearest_lat) & (sst_frame["lon"] == nearest_lon)].copy()
    subset["Year"] = pd.to_datetime(subset["time"]).dt.year

    features = (
        subset.groupby("Year")["sst"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "SST",
                "min": "SST_Min",
                "max": "SST_Max",
                "std": "SST_Std",
            }
        )
    )

    features["Latitude"] = latitude
    features["Longitude"] = longitude
    features["Nearest_SST_Latitude"] = nearest_lat
    features["Nearest_SST_Longitude"] = nearest_lon
    return features


def main() -> None:
    if not LANDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing landings file: {LANDINGS_PATH}")
    if not SST_PATH.exists():
        raise FileNotFoundError(f"Missing SST file: {SST_PATH}")

    landings = pd.read_csv(LANDINGS_PATH)
    sst = pd.read_csv(SST_PATH)

    state_frames: list[pd.DataFrame] = []
    for state, metadata in STATE_METADATA.items():
        state_landings = landings[landings["State"] == state].copy()
        if state_landings.empty:
            continue

        sst_features = yearly_sst_features(sst, metadata["Latitude"], metadata["Longitude"])
        merged = state_landings.merge(sst_features, on="Year", how="inner")
        merged["State"] = state
        state_frames.append(merged)

    if not state_frames:
        raise ValueError("No state-level rows were created. Check source files.")

    dataset = pd.concat(state_frames, ignore_index=True).sort_values(["State", "Year"]).reset_index(drop=True)
    dataset["Historical_Catch"] = dataset.groupby("State")["Landings_Tonnes"].shift(1)
    dataset["Historical_Catch"] = dataset["Historical_Catch"].fillna(dataset["Landings_Tonnes"])
    dataset["Availability"] = (
        dataset["Landings_Tonnes"] >= dataset.groupby("State")["Landings_Tonnes"].transform("median")
    ).astype(int)
    dataset["Month"] = dataset["Year"].astype(str) + "-06"
    dataset["Data_Source"] = "CMFRI state landings + Indian SST"

    ordered_columns = [
        "Year",
        "Month",
        "State",
        "Latitude",
        "Longitude",
        "Nearest_SST_Latitude",
        "Nearest_SST_Longitude",
        "SST",
        "SST_Min",
        "SST_Max",
        "SST_Std",
        "Historical_Catch",
        "Landings_Tonnes",
        "Availability",
        "Data_Source",
        "Source_URL",
    ]

    dataset = dataset[ordered_columns]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved real-world training dataset to {OUTPUT_PATH}")
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
