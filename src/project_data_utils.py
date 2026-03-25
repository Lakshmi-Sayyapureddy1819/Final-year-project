from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from juvenile_risk_utils import attach_maturity_reference, maturity_risk_label


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTISOURCE_DATA_PATH = PROJECT_ROOT / "data" / "multisource_training_data.csv"
REAL_DATA_PATH = PROJECT_ROOT / "data" / "real_world_training_data.csv"
FALLBACK_DATA_PATH = PROJECT_ROOT / "data" / "final_training_data_fixed.csv"

MAIN_FEATURE_COLUMNS = [
    "SST",
    "Salinity",
    "Dissolved_Oxygen",
    "Historical_Catch",
    "Latitude",
    "Longitude",
    "MonthSin",
    "MonthCos",
    "ThermalStress",
    "OxygenStress",
    "SalinityAnomaly",
    "SST_Min",
    "SST_Max",
    "SST_Std",
    "PFZ_Observations",
    "PFZ_Mean_Distance_km",
    "PFZ_Mean_Depth_m",
    "YearNum",
    "CatchLog",
    "CatchPerThermal",
    "TempOxygenInteraction",
]

JUVENILE_FEATURE_COLUMNS = [
    "SST",
    "Salinity",
    "Historical_Catch",
    "Dissolved_Oxygen",
    "ThermalStress",
    "OxygenStress",
    "SalinityAnomaly",
    "CatchLog",
    "TempOxygenInteraction",
]

OBSERVED_LENGTH_COLUMNS = [
    "Observed_Length_cm",
    "Fish_Length_cm",
    "Length_cm",
    "Average_Length_cm",
    "Mean_Length_cm",
]


def _default_risk_labels(risk_score: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [risk_score >= 0.67, risk_score >= 0.4],
            ["High", "Medium"],
            default="Low",
        ),
        index=risk_score.index,
    )


def balanced_risk_labels(risk_score: pd.Series) -> pd.Series:
    score_series = pd.to_numeric(pd.Series(risk_score), errors="coerce")
    valid_scores = score_series.dropna()

    # When we have enough spread, use rank-based tertiles so the fallback
    # juvenile labels do not collapse into mostly one class.
    if len(valid_scores) >= 9 and valid_scores.nunique() >= 3:
        rank_percentile = score_series.rank(method="first", pct=True)
        return pd.Series(
            np.select(
                [rank_percentile > (2.0 / 3.0), rank_percentile > (1.0 / 3.0)],
                ["High", "Medium"],
                default="Low",
            ),
            index=score_series.index,
        )

    return _default_risk_labels(score_series)


def balance_classification_frame(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    target_name: str = "target",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    dataset = features.reset_index(drop=True).copy()
    target_series = pd.Series(target).reset_index(drop=True)
    label_name = target_series.name or target_name
    dataset[label_name] = target_series

    class_counts = dataset[label_name].value_counts(dropna=False)
    if class_counts.empty or len(class_counts) <= 1 or class_counts.max() == class_counts.min():
        return features.reset_index(drop=True), target_series

    max_count = int(class_counts.max())
    balanced_parts: list[pd.DataFrame] = []
    for _, class_frame in dataset.groupby(label_name, dropna=False, sort=False):
        if len(class_frame) < max_count:
            class_frame = class_frame.sample(n=max_count, replace=True, random_state=random_state)
        balanced_parts.append(class_frame)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced.drop(columns=[label_name]), balanced[label_name]


def resolve_primary_dataset_path() -> Path:
    for candidate in [MULTISOURCE_DATA_PATH, REAL_DATA_PATH, FALLBACK_DATA_PATH]:
        if candidate.exists():
            return candidate
    return FALLBACK_DATA_PATH


def first_available_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def prepare_main_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dataset = frame.copy()

    if "Salinity" not in dataset.columns:
        dataset["Salinity"] = 34.0
    if "Dissolved_Oxygen" not in dataset.columns:
        dataset["Dissolved_Oxygen"] = 5.5
    if "Latitude" not in dataset.columns:
        dataset["Latitude"] = 0.0
    if "Longitude" not in dataset.columns:
        dataset["Longitude"] = 0.0

    if "Availability" not in dataset.columns:
        target_reference = dataset["Landings_Tonnes"] if "Landings_Tonnes" in dataset.columns else dataset["Historical_Catch"]
        dataset["Availability"] = (target_reference >= target_reference.median()).astype(int)

    if "Month" in dataset.columns:
        month_values = pd.to_datetime(dataset["Month"], errors="coerce").dt.month.fillna(1)
    else:
        month_values = pd.Series(np.ones(len(dataset)), index=dataset.index)

    angle = (2 * np.pi * month_values) / 12.0
    dataset["MonthSin"] = np.sin(angle)
    dataset["MonthCos"] = np.cos(angle)
    dataset["ThermalStress"] = (dataset["SST"] - 27.0).abs()
    dataset["OxygenStress"] = np.clip(5.5 - dataset["Dissolved_Oxygen"], 0.0, None)
    dataset["SalinityAnomaly"] = (dataset["Salinity"] - 34.0).abs()
    dataset["YearNum"] = pd.to_numeric(dataset.get("Year"), errors="coerce")
    if dataset["YearNum"].isna().all():
        dataset["YearNum"] = 2024.0
    else:
        dataset["YearNum"] = dataset["YearNum"].fillna(dataset["YearNum"].median()).astype(float)

    dataset["SST_Min"] = pd.to_numeric(dataset.get("SST_Min"), errors="coerce")
    dataset["SST_Max"] = pd.to_numeric(dataset.get("SST_Max"), errors="coerce")
    dataset["SST_Std"] = pd.to_numeric(dataset.get("SST_Std"), errors="coerce")
    dataset["SST_Min"] = dataset["SST_Min"].fillna(dataset["SST"] - 1.5)
    dataset["SST_Max"] = dataset["SST_Max"].fillna(dataset["SST"] + 1.5)
    dataset["SST_Std"] = dataset["SST_Std"].fillna(1.0 + dataset["ThermalStress"] * 0.15)

    for column in ["PFZ_Observations", "PFZ_Mean_Distance_km", "PFZ_Mean_Depth_m"]:
        dataset[column] = pd.to_numeric(dataset.get(column), errors="coerce").fillna(0.0)

    dataset["CatchLog"] = np.log1p(np.clip(dataset["Historical_Catch"], 0.0, None))
    dataset["CatchPerThermal"] = dataset["Historical_Catch"] / (dataset["ThermalStress"] + 1.0)
    dataset["TempOxygenInteraction"] = dataset["SST"] * dataset["Dissolved_Oxygen"]

    for column in MAIN_FEATURE_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = 0.0

    return dataset, MAIN_FEATURE_COLUMNS.copy()


def prepare_juvenile_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    dataset = frame.copy()
    if "Salinity" not in dataset.columns:
        dataset["Salinity"] = 34.0
    if "Dissolved_Oxygen" not in dataset.columns:
        dataset["Dissolved_Oxygen"] = 5.5

    dataset["ThermalStress"] = (dataset["SST"] - 27.0).abs()
    dataset["OxygenStress"] = np.clip(5.5 - dataset["Dissolved_Oxygen"], 0.0, None)
    dataset["SalinityAnomaly"] = (dataset["Salinity"] - 34.0).abs()
    dataset["CatchLog"] = np.log1p(np.clip(dataset["Historical_Catch"], 0.0, None))
    dataset["TempOxygenInteraction"] = dataset["SST"] * dataset["Dissolved_Oxygen"]

    catch_percentile = dataset["Historical_Catch"].rank(pct=True)
    risk_score = (
        0.45 * (1.0 - catch_percentile)
        + 0.3 * np.clip(dataset["ThermalStress"] / 4.0, 0.0, 1.0)
        + 0.25 * np.clip(dataset["OxygenStress"] / 2.5, 0.0, 1.0)
    )

    dataset["Heuristic_Juvenile_Risk"] = balanced_risk_labels(risk_score)
    dataset["Juvenile_Risk_Score"] = risk_score

    dataset = attach_maturity_reference(dataset)
    observed_length_column = first_available_column(dataset, OBSERVED_LENGTH_COLUMNS)
    exact_count = 0

    if observed_length_column is not None:
        dataset["Observed_Length_cm"] = pd.to_numeric(dataset[observed_length_column], errors="coerce")
    if "Maturity_Length_cm" in dataset.columns:
        dataset["Maturity_Length_cm"] = pd.to_numeric(dataset["Maturity_Length_cm"], errors="coerce")

    exact_labels: list[str | None] = []
    exact_scores: list[float | None] = []
    if observed_length_column is not None and "Maturity_Length_cm" in dataset.columns:
        for observed_length, maturity_length in zip(dataset["Observed_Length_cm"], dataset["Maturity_Length_cm"]):
            label, score = maturity_risk_label(
                float(observed_length) if pd.notna(observed_length) else None,
                float(maturity_length) if pd.notna(maturity_length) else None,
            )
            exact_labels.append(label)
            exact_scores.append(score)
        dataset["Exact_Juvenile_Risk"] = exact_labels
        dataset["Exact_Maturity_Score"] = exact_scores
        exact_count = int(pd.Series(exact_labels).notna().sum())
    else:
        dataset["Exact_Juvenile_Risk"] = None
        dataset["Exact_Maturity_Score"] = np.nan

    dataset["Juvenile_Risk"] = dataset["Exact_Juvenile_Risk"].fillna(dataset["Heuristic_Juvenile_Risk"])
    dataset["Juvenile_Risk_Source"] = np.where(
        dataset["Exact_Juvenile_Risk"].notna(),
        "Exact maturity rule",
        "Environmental heuristic",
    )

    for column in JUVENILE_FEATURE_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = 0.0
    return dataset, exact_count
