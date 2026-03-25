from __future__ import annotations

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMON_LIBOMP_PATHS = [
    Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
    Path("/usr/local/opt/libomp/lib/libomp.dylib"),
]


def main() -> None:
    print("Checking XGBoost runtime...")
    brew_path = shutil.which("brew")
    print(f"Homebrew available: {'YES' if brew_path else 'NO'}")

    libomp_path = next((path for path in COMMON_LIBOMP_PATHS if path.exists()), None)
    print(f"libomp detected: {'YES' if libomp_path else 'NO'}")
    if libomp_path:
        print(f"libomp path: {libomp_path}")

    try:
        import xgboost as xgb

        print("XGBoost import: OK")
        print(f"XGBoost version: {xgb.__version__}")
        print("Status: native XGBoost is ready on this machine.")
    except Exception as error:
        print("XGBoost import: FAILED")
        print(f"Error: {error}")
        print("Status: the project will use Gradient Boosting fallback until libomp is installed.")
        print("Recommended next step on macOS:")
        if brew_path:
            print("  brew install libomp")
        else:
            print("  1. Install Homebrew")
            print("  2. Run: brew install libomp")
        print("Then retrain with:")
        print("  .venv/bin/python src/model_training.py")


if __name__ == "__main__":
    main()
