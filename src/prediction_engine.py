from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from math import cos, radians, sin
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from juvenile_risk_utils import known_species, lookup_maturity_length, maturity_risk_label


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LEGACY_MAIN_FEATURES = ["SST", "Salinity", "Dissolved_Oxygen", "Historical_Catch"]
LEGACY_JUVENILE_FEATURES = ["SST", "Salinity", "Historical_Catch"]
PRIME_LOCATIONS = {"vizag", "kakinada", "chennai", "goa", "kochi", "nellore", "mangalore"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


@dataclass(frozen=True)
class PredictionResult:
    location: str
    latitude: float | None
    longitude: float | None
    species: str | None
    model_pipeline: str
    availability: int
    availability_score: float
    quantity: float
    base_juvenile_risk: str
    juvenile_risk: str
    juvenile_method: str
    juvenile_score: float
    maturity_score: float | None
    maturity_length_cm: float | None
    advisory: str
    safe_zone_suggestions: list[dict[str, Any]]


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "rf_clf": joblib.load(MODELS_DIR / "availability_model.pkl"),
        "rf_reg": joblib.load(MODELS_DIR / "quantity_model.pkl"),
        "juvenile_model": joblib.load(MODELS_DIR / "juvenile_model.pkl"),
        "xgb_clf": None,
        "xgb_reg": None,
        "pca": None,
        "hyb_clf": None,
        "hyb_reg": None,
    }

    optional_files = {
        "xgb_clf": "xgb_availability_model.pkl",
        "xgb_reg": "xgb_quantity_model.pkl",
        "pca": "pca_transform.pkl",
        "hyb_clf": "hybrid_availability_model.pkl",
        "hyb_reg": "hybrid_quantity_model.pkl",
    }

    for key, filename in optional_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            try:
                models[key] = joblib.load(path)
            except Exception:
                models[key] = None

    return models


def build_feature_row(
    sst: float,
    salinity: float,
    dissolved_oxygen: float,
    historical_catch: float,
    latitude: float | None = None,
    longitude: float | None = None,
    month: int | None = None,
) -> dict[str, float]:
    month_index = int(month or datetime.now().month)
    theta = (2 * np.pi * month_index) / 12.0
    thermal_stress = float(abs(float(sst) - 27.0))
    oxygen_stress = float(max(0.0, 5.5 - float(dissolved_oxygen)))
    salinity_anomaly = float(abs(float(salinity) - 34.0))
    current_year = float(datetime.now().year)

    return {
        "SST": float(sst),
        "Salinity": float(salinity),
        "Dissolved_Oxygen": float(dissolved_oxygen),
        "Historical_Catch": float(historical_catch),
        "Latitude": float(latitude or 0.0),
        "Longitude": float(longitude or 0.0),
        "Month": float(month_index),
        "MonthSin": float(np.sin(theta)),
        "MonthCos": float(np.cos(theta)),
        "ThermalStress": thermal_stress,
        "OxygenStress": oxygen_stress,
        "SalinityAnomaly": salinity_anomaly,
        "SST_Min": float(sst) - 1.5,
        "SST_Max": float(sst) + 1.5,
        "SST_Std": float(1.0 + thermal_stress * 0.15),
        "PFZ_Observations": 0.0,
        "PFZ_Mean_Distance_km": 0.0,
        "PFZ_Mean_Depth_m": 0.0,
        "YearNum": current_year,
        "CatchLog": float(np.log1p(max(float(historical_catch), 0.0))),
        "CatchPerThermal": float(float(historical_catch) / (thermal_stress + 1.0)),
        "TempOxygenInteraction": float(float(sst) * float(dissolved_oxygen)),
    }


def _prepare_input(model: Any, feature_row: dict[str, float], default_columns: list[str]) -> Any:
    if hasattr(model, "feature_names_in_"):
        columns = [str(column) for column in model.feature_names_in_]
    else:
        columns = default_columns

    frame = pd.DataFrame(
        [{column: float(feature_row.get(column, 0.0)) for column in columns}],
        columns=columns,
    )
    return frame if hasattr(model, "feature_names_in_") else frame.to_numpy()


def _availability_probability(model: Any, features: Any, predicted_label: int) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        positive_candidates = [1, "1", True, "True", "yes", "YES"]

        for candidate in positive_candidates:
            if candidate in classes:
                return float(probabilities[classes.index(candidate)])

        if len(probabilities) == 2:
            return float(probabilities[-1])

    return float(predicted_label)


def _boosting_label(model: Any) -> str:
    module_name = getattr(model.__class__, "__module__", "")
    class_name = getattr(model.__class__, "__name__", "")
    descriptor = f"{module_name}.{class_name}".lower()
    if "xgboost" in descriptor or "xgb" in descriptor:
        return "XGBoost"
    if "gradientboost" in descriptor:
        return "Gradient Boosting"
    return "Boosting"


def _predict_with_pipeline(
    models: dict[str, Any],
    feature_row: dict[str, float],
    model_choice: str,
) -> tuple[int, float, float, str]:
    rf_clf = models["rf_clf"]
    rf_reg = models["rf_reg"]

    if (
        model_choice == "hybrid"
        and models["pca"] is not None
        and models["hyb_clf"] is not None
        and models["hyb_reg"] is not None
    ):
        transformed = models["pca"].transform(
            _prepare_input(models["pca"], feature_row, LEGACY_MAIN_FEATURES)
        )
        availability = int(models["hyb_clf"].predict(transformed)[0])
        availability_score = _availability_probability(models["hyb_clf"], transformed, availability)
        quantity = float(models["hyb_reg"].predict(transformed)[0])
        return availability, availability_score, quantity, "Hybrid (PCA + RF + ET + Boosting)"

    if model_choice == "xgboost" and models["xgb_clf"] is not None and models["xgb_reg"] is not None:
        xgb_features = _prepare_input(models["xgb_clf"], feature_row, LEGACY_MAIN_FEATURES)
        availability = int(models["xgb_clf"].predict(xgb_features)[0])
        availability_score = _availability_probability(models["xgb_clf"], xgb_features, availability)
        quantity = float(models["xgb_reg"].predict(_prepare_input(models["xgb_reg"], feature_row, LEGACY_MAIN_FEATURES))[0])
        return availability, availability_score, quantity, _boosting_label(models["xgb_clf"])

    rf_features = _prepare_input(rf_clf, feature_row, LEGACY_MAIN_FEATURES)
    availability = int(rf_clf.predict(rf_features)[0])
    availability_score = _availability_probability(rf_clf, rf_features, availability)
    quantity = float(rf_reg.predict(_prepare_input(rf_reg, feature_row, LEGACY_MAIN_FEATURES))[0])
    return availability, availability_score, quantity, "Random Forest"


def _predict_base_juvenile_risk(models: dict[str, Any], feature_row: dict[str, float]) -> tuple[str, float]:
    juvenile_model = models["juvenile_model"]
    juvenile_features = _prepare_input(juvenile_model, feature_row, LEGACY_JUVENILE_FEATURES)
    label = str(juvenile_model.predict(juvenile_features)[0])

    fallback_scores = {"Low": 0.2, "Medium": 0.5, "High": 0.85}
    score = fallback_scores.get(label, 0.5)

    if hasattr(juvenile_model, "predict_proba"):
        probabilities = juvenile_model.predict_proba(juvenile_features)[0]
        classes = [str(item) for item in getattr(juvenile_model, "classes_", range(len(probabilities)))]
        risk_scale = {"Low": 0.15, "Medium": 0.55, "High": 0.9}
        weighted_score = 0.0
        for class_name, probability in zip(classes, probabilities):
            weighted_score += float(probability) * risk_scale.get(class_name, 0.5)
        score = _clamp(weighted_score)

    return label, score


def _classify_risk_score(score: float) -> str:
    if score >= 0.67:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _apply_decision_rules(
    location: str,
    sst: float,
    salinity: float,
    dissolved_oxygen: float,
    historical_catch: float,
    availability: int,
    quantity: float,
    juvenile_risk: str,
) -> tuple[int, float]:
    if juvenile_risk == "High":
        return 0, max(0.0, float(quantity))

    good_conditions = 0
    if 22 <= sst <= 30:
        good_conditions += 1
    if 30 <= salinity <= 36:
        good_conditions += 1
    if 5 <= dissolved_oxygen <= 8:
        good_conditions += 1
    if historical_catch >= 150:
        good_conditions += 1

    if good_conditions >= 3:
        availability = 1
        quantity = max(float(quantity), 200.0)

    if location.strip().lower() in PRIME_LOCATIONS and availability == 0:
        availability = 1
        quantity = max(float(quantity), 220.0)

    return availability, max(0.0, float(quantity))


def _offset_coordinates(latitude: float, longitude: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    north_km = distance_km * cos(radians(bearing_deg))
    east_km = distance_km * sin(radians(bearing_deg))

    delta_lat = north_km / 111.0
    cos_lat = max(0.2, cos(radians(latitude)))
    delta_lon = east_km / (111.0 * cos_lat)
    return latitude + delta_lat, longitude + delta_lon


def generate_safe_zone_suggestions(
    latitude: float | None,
    longitude: float | None,
    juvenile_score: float,
    base_quantity: float,
) -> list[dict[str, Any]]:
    if latitude is None or longitude is None:
        return []

    directions = [
        ("North-East", 45.0, 8.0),
        ("East", 90.0, 10.0),
        ("South-East", 135.0, 12.0),
        ("North-West", 315.0, 15.0),
    ]

    baseline_quantity = max(float(base_quantity), 180.0)
    suggestions: list[dict[str, Any]] = []
    for index, (zone_name, bearing, distance_km) in enumerate(directions):
        zone_lat, zone_lon = _offset_coordinates(latitude, longitude, distance_km, bearing)
        expected_risk_score = _clamp(juvenile_score - (0.18 + index * 0.05), 0.12, 0.6)
        expected_risk = "Low" if expected_risk_score < 0.4 else "Medium"
        expected_quantity = baseline_quantity * (0.8 + index * 0.06)

        suggestions.append(
            {
                "zone": zone_name,
                "distance_km": round(distance_km, 1),
                "latitude": round(zone_lat, 5),
                "longitude": round(zone_lon, 5),
                "expected_juvenile_risk": expected_risk,
                "expected_quantity_kg": round(expected_quantity, 2),
            }
        )

    return suggestions


def _build_advisory(availability: int, juvenile_risk: str) -> str:
    if juvenile_risk == "High":
        return "High juvenile density detected. Avoid fishing here and shift 8-15 km toward a safer zone."
    if availability == 0:
        return "Ecological risk is acceptable, but fish availability is weak. Consider scanning nearby zones."
    if juvenile_risk == "Medium":
        return "Fishing is possible with caution. Use larger mesh sizes and monitor juvenile presence."
    return "Favorable fishing window with lower juvenile risk. Continue sustainable fishing practices."


def predict_fishing_zone(
    location: str,
    sst: float,
    salinity: float,
    dissolved_oxygen: float,
    historical_catch: float,
    species: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    month: int | None = None,
    model_choice: str = "random_forest",
    observed_length_cm: float | None = None,
    maturity_length_cm: float | None = None,
) -> PredictionResult:
    models = load_models()
    feature_row = build_feature_row(
        sst=sst,
        salinity=salinity,
        dissolved_oxygen=dissolved_oxygen,
        historical_catch=historical_catch,
        latitude=latitude,
        longitude=longitude,
        month=month,
    )

    availability, availability_score, quantity, model_pipeline = _predict_with_pipeline(
        models=models,
        feature_row=feature_row,
        model_choice=model_choice,
    )

    base_juvenile_risk, base_score = _predict_base_juvenile_risk(models, feature_row)
    looked_up_maturity_length, _ = lookup_maturity_length(species)
    effective_maturity_length = maturity_length_cm if maturity_length_cm is not None else looked_up_maturity_length
    maturity_risk, maturity_score = maturity_risk_label(observed_length_cm, effective_maturity_length)

    if maturity_risk is not None and maturity_score is not None:
        juvenile_risk = maturity_risk
        juvenile_score = maturity_score
        if maturity_length_cm is not None:
            juvenile_method = "Exact maturity rule (entered observed and maturity lengths)"
        elif species and looked_up_maturity_length is not None:
            juvenile_method = "Exact maturity rule (FishBase maturity reference + observed length)"
        else:
            juvenile_method = "Exact maturity rule"
    else:
        juvenile_risk = _classify_risk_score(base_score)
        juvenile_score = base_score
        juvenile_method = "Environmental juvenile model fallback"

    availability, quantity = _apply_decision_rules(
        location=location,
        sst=sst,
        salinity=salinity,
        dissolved_oxygen=dissolved_oxygen,
        historical_catch=historical_catch,
        availability=availability,
        quantity=quantity,
        juvenile_risk=juvenile_risk,
    )

    safe_zone_suggestions = []
    if juvenile_risk == "High" or availability == 0:
        safe_zone_suggestions = generate_safe_zone_suggestions(latitude, longitude, juvenile_score, quantity)

    return PredictionResult(
        location=location,
        latitude=latitude,
        longitude=longitude,
        species=species,
        model_pipeline=model_pipeline,
        availability=availability,
        availability_score=round(availability_score, 3),
        quantity=round(quantity, 2),
        base_juvenile_risk=base_juvenile_risk,
        juvenile_risk=juvenile_risk,
        juvenile_method=juvenile_method,
        juvenile_score=round(juvenile_score, 3),
        maturity_score=round(maturity_score, 3) if maturity_score is not None else None,
        maturity_length_cm=round(effective_maturity_length, 3) if effective_maturity_length is not None else None,
        advisory=_build_advisory(availability, juvenile_risk),
        safe_zone_suggestions=safe_zone_suggestions,
    )
