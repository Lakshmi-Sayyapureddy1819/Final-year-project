from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PFZ_DATA_PATH = PROJECT_ROOT / "data" / "external" / "incois_pfz.csv"

PFZ_COLUMNS = [
    "State",
    "Date",
    "PFZ_Count",
    "Distance_km",
    "Depth_m",
    "Species",
    "Observed_Length_cm",
    "Maturity_Length_cm",
    "Data_Source",
    "Recorded_By",
    "Notes",
]

COASTAL_STATES = [
    "Andhra Pradesh",
    "Tamil Nadu",
    "Kerala",
    "Karnataka",
    "Goa",
    "Maharashtra",
    "Gujarat",
    "Odisha",
    "West Bengal",
    "Puducherry",
    "Andaman",
    "Nicobar",
    "Lakshadweep",
]


def ensure_pfz_dataset() -> Path:
    PFZ_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PFZ_DATA_PATH.exists():
        pd.DataFrame(columns=PFZ_COLUMNS).to_csv(PFZ_DATA_PATH, index=False)
    return PFZ_DATA_PATH


def load_pfz_dataset() -> pd.DataFrame:
    path = ensure_pfz_dataset()
    frame = pd.read_csv(path)
    for column in PFZ_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[PFZ_COLUMNS]


def normalize_pfz_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    for column in PFZ_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = pd.NA

    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for numeric_column in ["PFZ_Count", "Distance_km", "Depth_m", "Observed_Length_cm", "Maturity_Length_cm"]:
        dataset[numeric_column] = pd.to_numeric(dataset[numeric_column], errors="coerce")

    dataset["PFZ_Count"] = dataset["PFZ_Count"].fillna(1).astype(int)
    return dataset[PFZ_COLUMNS]


def append_pfz_observation(
    *,
    state: str,
    observed_date: date,
    pfz_count: int,
    distance_km: float | None,
    depth_m: float | None,
    species: str | None,
    observed_length_cm: float | None,
    maturity_length_cm: float | None,
    data_source: str,
    recorded_by: str | None,
    notes: str | None,
) -> Path:
    path = ensure_pfz_dataset()
    current = normalize_pfz_dataset(load_pfz_dataset())
    new_row = pd.DataFrame(
        [
            {
                "State": state,
                "Date": observed_date.isoformat(),
                "PFZ_Count": int(pfz_count),
                "Distance_km": distance_km,
                "Depth_m": depth_m,
                "Species": species or pd.NA,
                "Observed_Length_cm": observed_length_cm,
                "Maturity_Length_cm": maturity_length_cm,
                "Data_Source": data_source,
                "Recorded_By": recorded_by or pd.NA,
                "Notes": notes or pd.NA,
            }
        ]
    )
    merged = normalize_pfz_dataset(pd.concat([current, new_row], ignore_index=True))
    merged = merged.drop_duplicates().reset_index(drop=True)
    merged.to_csv(path, index=False)
    return path


def import_pfz_observations(source_path: Path) -> dict[str, int]:
    target_path = ensure_pfz_dataset()
    current = normalize_pfz_dataset(load_pfz_dataset())
    incoming = normalize_pfz_dataset(pd.read_csv(source_path))
    before_count = len(current)
    combined = normalize_pfz_dataset(pd.concat([current, incoming], ignore_index=True))
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined.to_csv(target_path, index=False)
    after_count = len(combined)
    return {
        "imported_rows": int(len(incoming)),
        "new_rows_added": int(after_count - before_count),
        "total_rows": int(after_count),
    }


def exact_ready_row_count(frame: pd.DataFrame | None = None) -> int:
    dataset = load_pfz_dataset() if frame is None else frame.copy()
    required = ["Species", "Observed_Length_cm", "Maturity_Length_cm"]
    for column in required:
        if column not in dataset.columns:
            return 0
    ready = dataset[required].notna().all(axis=1)
    return int(ready.sum())


def observation_summary() -> dict[str, int]:
    frame = normalize_pfz_dataset(load_pfz_dataset())
    return {
        "rows": int(len(frame)),
        "exact_ready_rows": exact_ready_row_count(frame),
        "states": int(frame["State"].dropna().nunique()) if "State" in frame.columns else 0,
        "species": int(frame["Species"].dropna().nunique()) if "Species" in frame.columns else 0,
    }


def recent_observations(limit: int = 10) -> pd.DataFrame:
    frame = normalize_pfz_dataset(load_pfz_dataset())
    if frame.empty:
        return frame
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.sort_values("Date", ascending=False).head(limit).reset_index(drop=True)


def data_quality_issues(frame: pd.DataFrame | None = None) -> list[str]:
    dataset = normalize_pfz_dataset(load_pfz_dataset() if frame is None else frame)
    issues: list[str] = []

    if dataset.empty:
        issues.append("No field observation rows recorded yet.")
        return issues

    missing_state_or_date = dataset["State"].isna().sum() + dataset["Date"].isna().sum()
    if missing_state_or_date:
        issues.append(f"{int(missing_state_or_date)} required State/Date values are missing.")

    negative_lengths = 0
    for column in ["Observed_Length_cm", "Maturity_Length_cm"]:
        if column in dataset.columns:
            negative_lengths += int((dataset[column].fillna(0) < 0).sum())
    if negative_lengths:
        issues.append(f"{negative_lengths} negative biological length values detected.")

    duplicates = int(dataset.duplicated().sum())
    if duplicates:
        issues.append(f"{duplicates} duplicate field observation rows detected.")

    exact_ready = exact_ready_row_count(dataset)
    if exact_ready == 0:
        issues.append("No exact-ready rows yet. Add Species, Observed_Length_cm, and Maturity_Length_cm.")

    return issues
