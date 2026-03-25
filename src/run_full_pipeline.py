from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full fish-catch project pipeline.")
    parser.add_argument(
        "--refresh-cmfri",
        action="store_true",
        help="Refresh CMFRI landings and rebuild the real-world base dataset before training.",
    )
    parser.add_argument(
        "--check-xgboost",
        action="store_true",
        help="Run the local XGBoost runtime readiness check before training.",
    )
    args = parser.parse_args()

    python_bin = sys.executable

    if args.refresh_cmfri:
        run_step("Fetch CMFRI state landings", [python_bin, "src/fetch_cmfri_state_landings.py"])
        run_step("Build real-world dataset", [python_bin, "src/build_real_world_dataset.py"])

    if args.check_xgboost:
        run_step("Check XGBoost runtime", [python_bin, "src/check_xgboost_runtime.py"])

    run_step("Check external datasets", [python_bin, "src/check_external_datasets.py"])
    run_step("Build multisource dataset", [python_bin, "src/build_multisource_dataset.py"])
    run_step("Train juvenile-risk model", [python_bin, "src/juvenile_risk_model.py"])
    run_step("Train main models", [python_bin, "src/model_training.py"])
    run_step("Validate project", [python_bin, "src/validate_project.py"])
    run_step("Generate algorithm execution report", [python_bin, "src/demo_algorithms.py"])


if __name__ == "__main__":
    main()
