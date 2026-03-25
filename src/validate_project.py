from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from field_data_store import observation_summary
from prediction_engine import predict_fishing_zone
from project_data_utils import (
    JUVENILE_FEATURE_COLUMNS,
    prepare_juvenile_training_frame,
    prepare_main_training_frame,
    resolve_primary_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "latest_validation_report.md"
METRICS_PATH = REPORTS_DIR / "latest_metrics.json"


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _class_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | dict[str, int]]:
    true_series = pd.Series(y_true).reset_index(drop=True)
    pred_series = pd.Series(y_pred).reset_index(drop=True)

    # Keep labels type-consistent so confusion matrices remain valid for both
    # numeric availability targets and string juvenile-risk targets.
    if true_series.dtype.kind != pred_series.dtype.kind:
        true_eval = true_series.astype(str)
        pred_eval = pred_series.astype(str)
    else:
        true_eval = true_series
        pred_eval = pred_series

    labels = pd.concat([true_eval, pred_eval], ignore_index=True).drop_duplicates().tolist()
    labels = sorted(labels, key=str)
    matrix = confusion_matrix(true_eval, pred_eval, labels=labels)
    matrix_dict = {
        str(label): {str(other): int(matrix[i][j]) for j, other in enumerate(labels)}
        for i, label in enumerate(labels)
    }

    return {
        "accuracy": _round_metric(accuracy_score(true_eval, pred_eval)),
        "precision_weighted": _round_metric(precision_score(true_eval, pred_eval, average="weighted", zero_division=0)),
        "recall_weighted": _round_metric(recall_score(true_eval, pred_eval, average="weighted", zero_division=0)),
        "f1_weighted": _round_metric(f1_score(true_eval, pred_eval, average="weighted", zero_division=0)),
        "confusion_matrix": matrix_dict,
    }


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": _round_metric(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": _round_metric(mean_absolute_error(y_true, y_pred)),
        "r2": _round_metric(r2_score(y_true, y_pred)),
    }


def evaluate_main_models(dataset: pd.DataFrame) -> dict[str, object]:
    prepared, feature_columns = prepare_main_training_frame(dataset)
    target_column = "Landings_Tonnes" if "Landings_Tonnes" in prepared.columns else "Historical_Catch"

    X = prepared[feature_columns]
    y_class = prepared["Availability"]
    y_reg = prepared[target_column]

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X,
        y_reg,
        test_size=0.2,
        random_state=42,
    )

    metrics: dict[str, object] = {
        "dataset_rows": int(len(prepared)),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "availability_class_counts": {str(key): int(value) for key, value in prepared["Availability"].value_counts().sort_index().items()},
    }

    rf_clf = joblib.load(MODELS_DIR / "availability_model.pkl")
    rf_reg = joblib.load(MODELS_DIR / "quantity_model.pkl")
    rf_class_pred = rf_clf.predict(X_test_class)
    rf_reg_pred = rf_reg.predict(X_test_reg)
    metrics["random_forest"] = {
        "availability": _class_metrics(y_test_class, rf_class_pred),
        "quantity": _regression_metrics(y_test_reg, rf_reg_pred),
    }

    xgb_clf_path = MODELS_DIR / "xgb_availability_model.pkl"
    xgb_reg_path = MODELS_DIR / "xgb_quantity_model.pkl"
    if xgb_clf_path.exists() and xgb_reg_path.exists():
        xgb_clf = joblib.load(xgb_clf_path)
        xgb_reg = joblib.load(xgb_reg_path)
        xgb_class_pred = xgb_clf.predict(X_test_class)
        xgb_reg_pred = xgb_reg.predict(X_test_reg)
        metrics["boosting"] = {
            "availability": _class_metrics(y_test_class, xgb_class_pred),
            "quantity": _regression_metrics(y_test_reg, xgb_reg_pred),
            "model_class": f"{xgb_clf.__class__.__module__}.{xgb_clf.__class__.__name__}",
        }

    pca_path = MODELS_DIR / "pca_transform.pkl"
    hybrid_clf_path = MODELS_DIR / "hybrid_availability_model.pkl"
    hybrid_reg_path = MODELS_DIR / "hybrid_quantity_model.pkl"
    if pca_path.exists() and hybrid_clf_path.exists() and hybrid_reg_path.exists():
        pca = joblib.load(pca_path)
        hybrid_clf = joblib.load(hybrid_clf_path)
        hybrid_reg = joblib.load(hybrid_reg_path)
        X_test_class_pca = pca.transform(X_test_class)
        X_test_reg_pca = pca.transform(X_test_reg)
        hybrid_class_pred = hybrid_clf.predict(X_test_class_pca)
        hybrid_reg_pred = hybrid_reg.predict(X_test_reg_pca)
        metrics["hybrid"] = {
            "availability": _class_metrics(y_test_class, hybrid_class_pred),
            "quantity": _regression_metrics(y_test_reg, hybrid_reg_pred),
        }

    return metrics


def evaluate_juvenile_model(dataset: pd.DataFrame) -> dict[str, object]:
    prepared, exact_label_count = prepare_juvenile_training_frame(dataset)
    X = prepared[JUVENILE_FEATURE_COLUMNS]
    y = prepared["Juvenile_Risk"]
    stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    juvenile_model = joblib.load(MODELS_DIR / "juvenile_model.pkl")
    predictions = juvenile_model.predict(X_test)
    source_counts = prepared["Juvenile_Risk_Source"].value_counts(dropna=False).to_dict()

    return {
        "dataset_rows": int(len(prepared)),
        "exact_label_rows": int(exact_label_count),
        "class_counts": {str(key): int(value) for key, value in prepared["Juvenile_Risk"].value_counts().items()},
        "risk_source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "metrics": _class_metrics(y_test, predictions),
    }


def run_demo_checks() -> list[dict[str, object]]:
    demo_cases = [
        {
            "name": "Exact maturity high-risk case",
            "kwargs": {
                "location": "Vizag",
                "sst": 28.0,
                "salinity": 34.0,
                "dissolved_oxygen": 6.2,
                "historical_catch": 250.0,
                "species": "Sardinella longiceps",
                "observed_length_cm": 12.0,
                "model_choice": "random_forest",
            },
            "expect": lambda result: result.juvenile_risk == "High" and "Exact maturity rule" in result.juvenile_method,
        },
        {
            "name": "Exact maturity low-risk case",
            "kwargs": {
                "location": "Vizag",
                "sst": 28.0,
                "salinity": 34.0,
                "dissolved_oxygen": 6.2,
                "historical_catch": 250.0,
                "species": "Sardinella longiceps",
                "observed_length_cm": 18.0,
                "model_choice": "random_forest",
            },
            "expect": lambda result: result.juvenile_risk == "Low" and "Exact maturity rule" in result.juvenile_method,
        },
        {
            "name": "Environmental fallback case",
            "kwargs": {
                "location": "Vizag",
                "sst": 28.0,
                "salinity": 34.0,
                "dissolved_oxygen": 6.2,
                "historical_catch": 250.0,
                "model_choice": "random_forest",
            },
            "expect": lambda result: result.juvenile_method == "Environmental juvenile model fallback",
        },
    ]

    outputs: list[dict[str, object]] = []
    for case in demo_cases:
        result = predict_fishing_zone(**case["kwargs"])
        passed = bool(case["expect"](result))
        outputs.append(
            {
                "name": case["name"],
                "passed": passed,
                "juvenile_risk": result.juvenile_risk,
                "juvenile_method": result.juvenile_method,
                "maturity_length_cm": result.maturity_length_cm,
                "availability": int(result.availability),
            }
        )
    return outputs


def build_markdown_report(metrics: dict[str, object]) -> str:
    main_metrics = metrics["main_models"]
    juvenile_metrics = metrics["juvenile_model"]
    demo_checks = metrics["demo_checks"]
    field_summary = metrics["field_data"]

    lines = [
        "# Validation Report",
        "",
        f"- Dataset path: `{metrics['dataset_path']}`",
        f"- Dataset rows: `{main_metrics['dataset_rows']}`",
        f"- Juvenile exact-label rows in training data: `{juvenile_metrics['exact_label_rows']}`",
        f"- Field observation rows: `{field_summary['rows']}`",
        f"- Field exact-ready rows: `{field_summary['exact_ready_rows']}`",
        "",
        "## Main model metrics",
        "",
        f"- Availability class counts: `{main_metrics['availability_class_counts']}`",
        f"- Random Forest availability accuracy: `{main_metrics['random_forest']['availability']['accuracy']}`",
        f"- Random Forest quantity RMSE: `{main_metrics['random_forest']['quantity']['rmse']}`",
    ]

    if "boosting" in main_metrics:
        lines.extend(
            [
                f"- Boosting availability accuracy: `{main_metrics['boosting']['availability']['accuracy']}`",
                f"- Boosting quantity RMSE: `{main_metrics['boosting']['quantity']['rmse']}`",
                f"- Boosting implementation class: `{main_metrics['boosting']['model_class']}`",
            ]
        )

    if "hybrid" in main_metrics:
        lines.extend(
            [
                f"- Hybrid availability accuracy: `{main_metrics['hybrid']['availability']['accuracy']}`",
                f"- Hybrid quantity RMSE: `{main_metrics['hybrid']['quantity']['rmse']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Juvenile model metrics",
            "",
            f"- Juvenile class counts: `{juvenile_metrics['class_counts']}`",
            f"- Juvenile accuracy: `{juvenile_metrics['metrics']['accuracy']}`",
            f"- Juvenile weighted F1: `{juvenile_metrics['metrics']['f1_weighted']}`",
            f"- Juvenile risk-source counts: `{juvenile_metrics['risk_source_counts']}`",
            "",
            "## Demo verification checks",
            "",
        ]
    )

    for case in demo_checks:
        status = "PASS" if case["passed"] else "FAIL"
        lines.append(
            f"- {status}: `{case['name']}` -> risk `{case['juvenile_risk']}`, method `{case['juvenile_method']}`"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    dataset_path = resolve_primary_dataset_path()
    dataset = pd.read_csv(dataset_path)

    metrics = {
        "dataset_path": str(dataset_path),
        "main_models": evaluate_main_models(dataset),
        "juvenile_model": evaluate_juvenile_model(dataset),
        "field_data": observation_summary(),
        "demo_checks": run_demo_checks(),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = build_markdown_report(metrics)
    REPORT_PATH.write_text(report_content, encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(report_content)
    print(f"Saved report to {REPORT_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
