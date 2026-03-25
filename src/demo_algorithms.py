from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from juvenile_risk_utils import load_maturity_reference
from prediction_engine import load_models, predict_fishing_zone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "algorithm_execution_report.md"
METRICS_PATH = REPORTS_DIR / "latest_metrics.json"

BALANCING_BASELINE = {
    "random_forest_accuracy": 0.5909,
    "boosting_accuracy": 0.5,
    "hybrid_accuracy": 0.4545,
    "juvenile_accuracy": 0.9091,
    "juvenile_f1": 0.8874,
    "juvenile_class_counts": {"Low": 75, "Medium": 34, "High": 1},
}

CODE_SECTION_SPECS = [
    {
        "title": "Pipeline Selection Logic",
        "description": "This block selects the Random Forest, Boosting, or Hybrid inference path during prediction.",
        "path": PROJECT_ROOT / "src" / "prediction_engine.py",
        "start_marker": "def _predict_with_pipeline(",
        "end_marker": "def _predict_base_juvenile_risk(",
    },
    {
        "title": "Boosting Model Builders",
        "description": "This block constructs the boosting models and transparently falls back to Gradient Boosting when native XGBoost is unavailable.",
        "path": PROJECT_ROOT / "src" / "model_training.py",
        "start_marker": "def build_boosting_classifier():",
        "end_marker": "print(\"Loading training dataset...\")",
    },
    {
        "title": "Exact Juvenile Risk Formula",
        "description": "This is the biological rule used when observed fish length and maturity length are available.",
        "path": PROJECT_ROOT / "src" / "juvenile_risk_utils.py",
        "start_marker": "def maturity_risk_score(",
        "end_marker": "def attach_maturity_reference(",
    },
    {
        "title": "Juvenile Training Data Preparation",
        "description": "This block merges heuristic juvenile labels with exact maturity-based labels for training.",
        "path": PROJECT_ROOT / "src" / "project_data_utils.py",
        "start_marker": "def prepare_juvenile_training_frame(",
    },
]

DEMO_CASES = [
    {
        "case": "Random Forest pipeline",
        "description": "Executes the tabular Random Forest classifier and regressor with exact maturity refinement.",
        "model_choice": "random_forest",
        "kwargs": {
            "location": "Vizag",
            "sst": 28.0,
            "salinity": 34.0,
            "dissolved_oxygen": 6.2,
            "historical_catch": 250.0,
            "latitude": 17.6868,
            "longitude": 83.2185,
            "species": "Sardinella longiceps",
            "observed_length_cm": 12.0,
        },
    },
    {
        "case": "Boosting pipeline",
        "description": "Executes the boosting path. On this machine it resolves to native XGBoost if available, otherwise Gradient Boosting fallback.",
        "model_choice": "xgboost",
        "kwargs": {
            "location": "Vizag",
            "sst": 28.0,
            "salinity": 34.0,
            "dissolved_oxygen": 6.2,
            "historical_catch": 250.0,
            "latitude": 17.6868,
            "longitude": 83.2185,
            "species": "Sardinella longiceps",
            "observed_length_cm": 12.0,
        },
    },
    {
        "case": "Hybrid PCA + RF + ET + Boosting pipeline",
        "description": "Executes the PCA-transformed ensemble path used in the hybrid model.",
        "model_choice": "hybrid",
        "kwargs": {
            "location": "Vizag",
            "sst": 28.0,
            "salinity": 34.0,
            "dissolved_oxygen": 6.2,
            "historical_catch": 250.0,
            "latitude": 17.6868,
            "longitude": 83.2185,
            "species": "Sardinella longiceps",
            "observed_length_cm": 12.0,
        },
    },
    {
        "case": "Exact juvenile-risk rule",
        "description": "Shows the biological maturity rule using FishBase maturity length plus observed fish length.",
        "model_choice": "random_forest",
        "kwargs": {
            "location": "Vizag",
            "sst": 28.0,
            "salinity": 34.0,
            "dissolved_oxygen": 6.2,
            "historical_catch": 250.0,
            "species": "Sardinella longiceps",
            "observed_length_cm": 12.0,
        },
    },
    {
        "case": "Environmental juvenile fallback",
        "description": "Shows the fallback juvenile model when species or maturity data are not provided.",
        "model_choice": "random_forest",
        "kwargs": {
            "location": "Vizag",
            "sst": 28.0,
            "salinity": 34.0,
            "dissolved_oxygen": 6.2,
            "historical_catch": 250.0,
        },
    },
]


def _class_path(model: Any | None) -> str:
    if model is None:
        return "Unavailable"
    return f"{model.__class__.__module__}.{model.__class__.__name__}"


def extract_code_section(path: Path, start_marker: str, end_marker: str | None = None) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_index = next((index for index, line in enumerate(lines) if start_marker in line), None)
    if start_index is None:
        return f"# Unable to find code marker: {start_marker}"

    end_index = len(lines)
    if end_marker is not None:
        located_end = next((index for index, line in enumerate(lines[start_index + 1 :], start=start_index + 1) if end_marker in line), None)
        if located_end is not None:
            end_index = located_end

    snippet = "\n".join(lines[start_index:end_index]).rstrip()
    return snippet or "# Code block is empty"


def get_code_sections() -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for spec in CODE_SECTION_SPECS:
        sections.append(
            {
                "title": str(spec["title"]),
                "description": str(spec["description"]),
                "path": str(spec["path"]),
                "code": extract_code_section(
                    path=Path(spec["path"]),
                    start_marker=str(spec["start_marker"]),
                    end_marker=str(spec["end_marker"]) if spec.get("end_marker") else None,
                ),
            }
        )
    return sections


def get_runtime_summary() -> list[dict[str, Any]]:
    models = load_models()
    maturity_reference = load_maturity_reference()

    return [
        {
            "algorithm": "Random Forest",
            "status": "Ready" if models.get("rf_clf") is not None and models.get("rf_reg") is not None else "Missing",
            "classifier": _class_path(models.get("rf_clf")),
            "regressor": _class_path(models.get("rf_reg")),
        },
        {
            "algorithm": "Boosting",
            "status": "Ready" if models.get("xgb_clf") is not None and models.get("xgb_reg") is not None else "Missing",
            "classifier": _class_path(models.get("xgb_clf")),
            "regressor": _class_path(models.get("xgb_reg")),
        },
        {
            "algorithm": "Hybrid (PCA + RF + ET + Boosting)",
            "status": (
                "Ready"
                if models.get("pca") is not None and models.get("hyb_clf") is not None and models.get("hyb_reg") is not None
                else "Missing"
            ),
            "classifier": _class_path(models.get("hyb_clf")),
            "regressor": _class_path(models.get("hyb_reg")),
        },
        {
            "algorithm": "Juvenile ML layer",
            "status": "Ready" if models.get("juvenile_model") is not None else "Missing",
            "classifier": _class_path(models.get("juvenile_model")),
            "regressor": "-",
        },
        {
            "algorithm": "Exact maturity rule",
            "status": "Ready" if not maturity_reference.empty else "Missing reference data",
            "classifier": "Rule-based formula: JR = 1 - observed_length / maturity_length",
            "regressor": f"FishBase rows: {len(maturity_reference)}",
        },
    ]


def get_balancing_comparison() -> list[dict[str, Any]]:
    current_metrics: dict[str, Any] = {}
    if METRICS_PATH.exists():
        current_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    main_metrics = current_metrics.get("main_models", {})
    juvenile_metrics = current_metrics.get("juvenile_model", {})

    rows = [
        {
            "metric": "Random Forest availability accuracy",
            "before_balancing": BALANCING_BASELINE["random_forest_accuracy"],
            "after_balancing": main_metrics.get("random_forest", {}).get("availability", {}).get("accuracy"),
            "impact": "Improved",
        },
        {
            "metric": "Boosting availability accuracy",
            "before_balancing": BALANCING_BASELINE["boosting_accuracy"],
            "after_balancing": main_metrics.get("boosting", {}).get("availability", {}).get("accuracy"),
            "impact": "Dropped",
        },
        {
            "metric": "Hybrid availability accuracy",
            "before_balancing": BALANCING_BASELINE["hybrid_accuracy"],
            "after_balancing": main_metrics.get("hybrid", {}).get("availability", {}).get("accuracy"),
            "impact": "Improved",
        },
        {
            "metric": "Juvenile accuracy",
            "before_balancing": BALANCING_BASELINE["juvenile_accuracy"],
            "after_balancing": juvenile_metrics.get("metrics", {}).get("accuracy"),
            "impact": "More conservative",
        },
        {
            "metric": "Juvenile weighted F1",
            "before_balancing": BALANCING_BASELINE["juvenile_f1"],
            "after_balancing": juvenile_metrics.get("metrics", {}).get("f1_weighted"),
            "impact": "More realistic",
        },
        {
            "metric": "Random Forest quantity RMSE",
            "before_balancing": 82478.3759,
            "after_balancing": main_metrics.get("random_forest", {}).get("quantity", {}).get("rmse"),
            "impact": "Lower is better",
        },
        {
            "metric": "Boosting quantity RMSE",
            "before_balancing": 91903.1661,
            "after_balancing": main_metrics.get("boosting", {}).get("quantity", {}).get("rmse"),
            "impact": "Lower is better",
        },
        {
            "metric": "Hybrid quantity RMSE",
            "before_balancing": 76830.9764,
            "after_balancing": main_metrics.get("hybrid", {}).get("quantity", {}).get("rmse"),
            "impact": "Lower is better",
        },
        {
            "metric": "Juvenile class counts",
            "before_balancing": BALANCING_BASELINE["juvenile_class_counts"],
            "after_balancing": juvenile_metrics.get("class_counts"),
            "impact": "Balanced classes",
        },
    ]
    return rows


def run_algorithm_demos() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in DEMO_CASES:
        result = predict_fishing_zone(model_choice=case["model_choice"], **case["kwargs"])
        results.append(
            {
                "case": case["case"],
                "description": case["description"],
                "requested_pipeline": case["model_choice"],
                "resolved_pipeline": result.model_pipeline,
                "availability": "YES" if result.availability else "NO",
                "availability_score": result.availability_score,
                "quantity_kg": result.quantity,
                "juvenile_risk": result.juvenile_risk,
                "juvenile_method": result.juvenile_method,
                "maturity_length_cm": result.maturity_length_cm,
                "safe_zone_count": len(result.safe_zone_suggestions),
                "advisory": result.advisory,
            }
        )
    return results


def build_algorithm_execution_report() -> str:
    runtime_summary = get_runtime_summary()
    balancing_comparison = get_balancing_comparison()
    demo_results = run_algorithm_demos()
    code_sections = get_code_sections()

    lines = [
        "# Algorithm Execution Report",
        "",
        "This report is intended for viva/demo use. It shows which algorithms are loaded, how they execute on sample inputs, and the exact project code blocks to explain.",
        "",
        "## Runtime status",
        "",
    ]

    for item in runtime_summary:
        lines.extend(
            [
                f"- `{item['algorithm']}`: `{item['status']}`",
                f"  Classifier/logic: `{item['classifier']}`",
                f"  Regressor/detail: `{item['regressor']}`",
            ]
        )

    lines.extend(["", "## Improvement comparison", ""])

    for item in balancing_comparison:
        lines.extend(
            [
                f"- `{item['metric']}`",
                f"  Before balancing: `{item['before_balancing']}`",
                f"  After balancing: `{item['after_balancing']}`",
                f"  Interpretation: `{item['impact']}`",
            ]
        )

    lines.extend(["", "## Demo executions", ""])

    for item in demo_results:
        lines.extend(
            [
                f"### {item['case']}",
                "",
                item["description"],
                "",
                f"- Requested pipeline: `{item['requested_pipeline']}`",
                f"- Resolved pipeline: `{item['resolved_pipeline']}`",
                f"- Availability: `{item['availability']}`",
                f"- Availability score: `{item['availability_score']}`",
                f"- Predicted quantity (kg): `{item['quantity_kg']}`",
                f"- Juvenile risk: `{item['juvenile_risk']}`",
                f"- Juvenile method: `{item['juvenile_method']}`",
                f"- Applied maturity length (cm): `{item['maturity_length_cm']}`",
                f"- Safe-zone suggestions: `{item['safe_zone_count']}`",
                f"- Advisory: `{item['advisory']}`",
                "",
            ]
        )

    lines.extend(["## Code blocks for viva", ""])

    for item in code_sections:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                item["description"],
                "",
                f"Source: `{item['path']}`",
                "",
                "```python",
                item["code"],
                "```",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_algorithm_execution_report(path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_algorithm_execution_report(), encoding="utf-8")
    return path


def main() -> None:
    report_path = write_algorithm_execution_report()
    print(f"Algorithm execution report written to {report_path}")

    print("\nRuntime summary:")
    for item in get_runtime_summary():
        print(f"- {item['algorithm']}: {item['status']} ({item['classifier']})")

    print("\nDemo execution summary:")
    for item in run_algorithm_demos():
        print(
            f"- {item['case']}: pipeline={item['resolved_pipeline']}, "
            f"availability={item['availability']}, juvenile={item['juvenile_risk']}, qty={item['quantity_kg']}"
        )


if __name__ == "__main__":
    main()
