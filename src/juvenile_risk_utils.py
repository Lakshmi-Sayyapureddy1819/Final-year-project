from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATURITY_REFERENCE_PATH = PROJECT_ROOT / "data" / "external" / "fishbase_maturity.csv"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


@lru_cache(maxsize=1)
def load_maturity_reference(path: Path = MATURITY_REFERENCE_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Species", "Maturity_Length_cm", "Source_URL"])
    return pd.read_csv(path)


def known_species(path: Path = MATURITY_REFERENCE_PATH) -> list[str]:
    frame = load_maturity_reference(path)
    if frame.empty or "Species" not in frame.columns:
        return []
    return sorted(frame["Species"].dropna().astype(str).unique().tolist())


def lookup_maturity_length(species: str | None, path: Path = MATURITY_REFERENCE_PATH) -> tuple[float | None, str | None]:
    if not species:
        return None, None

    frame = load_maturity_reference(path)
    if frame.empty:
        return None, None

    matches = frame[frame["Species"].astype(str).str.casefold() == species.casefold()]
    if matches.empty:
        return None, None

    row = matches.iloc[0]
    length_value = row.get("Maturity_Length_cm")
    source_url = row.get("Source_URL")
    length = float(length_value) if pd.notna(length_value) else None
    source = str(source_url) if pd.notna(source_url) else None
    return length, source


def maturity_risk_score(observed_length_cm: float | None, maturity_length_cm: float | None) -> float | None:
    if observed_length_cm is None or maturity_length_cm is None:
        return None
    if observed_length_cm <= 0 or maturity_length_cm <= 0:
        return None
    return _clamp(1.0 - (float(observed_length_cm) / float(maturity_length_cm)))


def maturity_risk_label(observed_length_cm: float | None, maturity_length_cm: float | None) -> tuple[str | None, float | None]:
    score = maturity_risk_score(observed_length_cm, maturity_length_cm)
    if score is None:
        return None, None

    if score >= 0.2:
        return "High", score
    if score > 0.0:
        return "Medium", score
    return "Low", score


def attach_maturity_reference(
    frame: pd.DataFrame,
    *,
    species_column: str = "Species",
    maturity_column: str = "Maturity_Length_cm",
    path: Path = MATURITY_REFERENCE_PATH,
) -> pd.DataFrame:
    dataset = frame.copy()
    if species_column not in dataset.columns:
        return dataset

    if maturity_column in dataset.columns and dataset[maturity_column].notna().any():
        return dataset

    reference = load_maturity_reference(path)
    if reference.empty:
        return dataset

    merged = dataset.merge(
        reference[["Species", "Maturity_Length_cm"]],
        left_on=species_column,
        right_on="Species",
        how="left",
        suffixes=("", "_reference"),
    )

    reference_column = f"{maturity_column}_reference"
    if maturity_column not in merged.columns:
        merged[maturity_column] = merged.get(reference_column)
    else:
        merged[maturity_column] = merged[maturity_column].fillna(merged.get(reference_column))

    for column in ["Species_reference", reference_column]:
        if column in merged.columns:
            merged = merged.drop(columns=column)

    return merged
