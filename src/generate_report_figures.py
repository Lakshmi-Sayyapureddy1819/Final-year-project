from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplconfig")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_ROOT / "reports" / "latest_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

BASELINE = {
    "availability_accuracy": {
        "Random Forest": 0.5909,
        "Boosting": 0.5000,
        "Hybrid": 0.4545,
    },
    "quantity_rmse": {
        "Random Forest": 82478.3759,
        "Boosting": 91903.1661,
        "Hybrid": 76830.9764,
    },
    "juvenile_accuracy": 0.6818,
}


def _load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_availability_chart(metrics: dict) -> None:
    values = {
        "Random Forest": metrics["main_models"]["random_forest"]["availability"]["accuracy"],
        "Boosting": metrics["main_models"]["boosting"]["availability"]["accuracy"],
        "Hybrid": metrics["main_models"]["hybrid"]["availability"]["accuracy"],
    }

    labels = list(values.keys())
    y = [values[label] * 100 for label in labels]

    plt.figure(figsize=(7.2, 4.4))
    bars = plt.bar(labels, y, color=["#0b6e4f", "#1f78b4", "#f28e2b"])
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Availability Prediction Accuracy")
    for bar, value in zip(bars, y):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    _save_current_figure(FIGURES_DIR / "availability_accuracy.png")


def generate_quantity_rmse_chart(metrics: dict) -> None:
    values = {
        "Random Forest": metrics["main_models"]["random_forest"]["quantity"]["rmse"],
        "Boosting": metrics["main_models"]["boosting"]["quantity"]["rmse"],
        "Hybrid": metrics["main_models"]["hybrid"]["quantity"]["rmse"],
    }

    labels = list(values.keys())
    y = [values[label] for label in labels]

    plt.figure(figsize=(7.2, 4.4))
    bars = plt.bar(labels, y, color=["#3b5b92", "#c0392b", "#8e6c8a"])
    plt.ylabel("RMSE")
    plt.title("Catch Quantity Prediction Error")
    for bar, value in zip(bars, y):
        plt.text(bar.get_x() + bar.get_width() / 2, value + max(y) * 0.02, f"{value:.0f}", ha="center", va="bottom", fontsize=9)

    _save_current_figure(FIGURES_DIR / "quantity_rmse.png")


def generate_juvenile_confusion_matrix(metrics: dict) -> None:
    confusion = metrics["juvenile_model"]["metrics"]["confusion_matrix"]
    labels = ["High", "Medium", "Low"]
    matrix = np.array([[confusion[row].get(col, 0) for col in labels] for row in labels], dtype=float)

    plt.figure(figsize=(6.2, 5.0))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Juvenile Risk Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(fraction=0.046, pad=0.04)
    _save_current_figure(FIGURES_DIR / "juvenile_confusion_matrix.png")


def generate_improvement_chart(metrics: dict) -> None:
    labels = ["RF Acc", "Boost Acc", "Hybrid Acc", "Juvenile Acc"]
    before = [
        BASELINE["availability_accuracy"]["Random Forest"] * 100,
        BASELINE["availability_accuracy"]["Boosting"] * 100,
        BASELINE["availability_accuracy"]["Hybrid"] * 100,
        BASELINE["juvenile_accuracy"] * 100,
    ]
    after = [
        metrics["main_models"]["random_forest"]["availability"]["accuracy"] * 100,
        metrics["main_models"]["boosting"]["availability"]["accuracy"] * 100,
        metrics["main_models"]["hybrid"]["availability"]["accuracy"] * 100,
        metrics["juvenile_model"]["metrics"]["accuracy"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.34

    plt.figure(figsize=(8.2, 4.6))
    plt.bar(x - width / 2, before, width=width, label="Earlier baseline", color="#9aa5b1")
    plt.bar(x + width / 2, after, width=width, label="Current improved", color="#1f78b4")
    plt.xticks(x, labels)
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.title("Model Improvement Comparison")
    plt.legend()

    _save_current_figure(FIGURES_DIR / "improvement_comparison.png")


def main() -> None:
    metrics = _load_metrics()
    generate_availability_chart(metrics)
    generate_quantity_rmse_chart(metrics)
    generate_juvenile_confusion_matrix(metrics)
    generate_improvement_chart(metrics)
    print(f"Saved report figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
