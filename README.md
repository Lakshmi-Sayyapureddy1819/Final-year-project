# AI-Driven Fish Catch Prediction System

This repository contains the working code for a final-year project on fish availability prediction, juvenile-risk screening, catch quantity estimation, and safe-zone recommendation.

## What the code now includes

- A shared prediction engine in [src/prediction_engine.py](/Users/lakshmis/Final-year-project/src/prediction_engine.py) that keeps the CLI app, Streamlit app, and map heatmap aligned.
- A juvenile-risk layer that combines the trained juvenile model with maturity-based scoring using `JR = 1 - observed_length / maturity_length`.
- FishBase maturity-length lookup support for common marine species, so the exact juvenile-risk rule can run directly in the app when species and observed length are available.
- Safe-zone suggestions 8-15 km away when the selected zone is high-risk or not recommended.
- Training scripts that use robust project-relative paths instead of fragile `../models/...` assumptions.
- A real-world data pipeline based on official CMFRI state-wise landing data plus the SST dataset already present in the repository.

## Current project structure

- [src/app.py](/Users/lakshmis/Final-year-project/src/app.py): main Streamlit application.
- [src/map_app.py](/Users/lakshmis/Final-year-project/src/map_app.py): heatmap explorer for safer candidate zones.
- [src/main.py](/Users/lakshmis/Final-year-project/src/main.py): command-line interface.
- [src/model_training.py](/Users/lakshmis/Final-year-project/src/model_training.py): Random Forest, XGBoost, and Hybrid PCA + RF + XGBoost training flow.
- [src/juvenile_risk_model.py](/Users/lakshmis/Final-year-project/src/juvenile_risk_model.py): juvenile-risk model training flow.

## Final-year project methodology

This project now follows a no-CNN methodology:

- Fish availability prediction using machine learning classifiers such as Random Forest and Boosting models.
- Catch quantity prediction using regression models such as Random Forest Regressor and Boosting regressors.
- Juvenile-risk assessment using the exact maturity rule `JR = 1 - observed_length / maturity_length` whenever real species and length data are available, with an environmental fallback model when they are not.
- Safe-zone recommendation by shifting 8-15 km toward lower-risk nearby coordinates.

## Algorithm note

- The codebase is wired for `Random Forest`, `XGBoost`, and `Hybrid (PCA + RF + XGBoost)`.
- On this Mac, `xgboost` currently needs the `libomp.dylib` runtime to load. Until that runtime is installed, the training script automatically uses `Gradient Boosting` as the boosting fallback.
- The juvenile-risk layer is exact at inference time whenever you provide `observed length` and either a `species` from FishBase or a manual `maturity length`.
- The Streamlit app now includes a `Viva Algorithm Demo` section that executes all algorithm paths and shows the corresponding project code blocks.

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train or retrain models:

```bash
python src/fetch_cmfri_state_landings.py
python src/build_real_world_dataset.py
python src/check_external_datasets.py
python src/build_multisource_dataset.py
python src/juvenile_risk_model.py
python src/model_training.py
python src/validate_project.py
```

3. Launch the main app:

```bash
streamlit run src/app.py
```

4. Launch the heatmap app:

```bash
streamlit run src/map_app.py
```

## Report helper

Use [REPORT_METHODOLOGY_NO_CNN.md](/Users/lakshmis/Final-year-project/REPORT_METHODOLOGY_NO_CNN.md) as the replacement methodology text for your report.

## Testing and validation

- Use [VALIDATION_GUIDE.md](/Users/lakshmis/Final-year-project/VALIDATION_GUIDE.md) for demo, viva, and verification steps.
- Run `python src/run_full_pipeline.py` to rebuild the dataset, retrain the models, and generate a validation report in [reports/latest_validation_report.md](/Users/lakshmis/Final-year-project/reports/latest_validation_report.md).
- Run `.venv/bin/python src/demo_algorithms.py` to generate [algorithm_execution_report.md](/Users/lakshmis/Final-year-project/reports/algorithm_execution_report.md) with live algorithm outputs plus viva-ready code blocks.
- Run `.venv/bin/python -m unittest discover -s tests` to verify the exact juvenile-risk logic and fallback behavior.
- Use the Streamlit sidebar field-data form to save real `Species + Observed_Length_cm` rows into [data/external/incois_pfz.csv](/Users/lakshmis/Final-year-project/data/external/incois_pfz.csv), then rerun the pipeline.
- Import batch CSV files with `.venv/bin/python src/import_pfz_observations.py your_file.csv`.
- Check native XGBoost readiness with `.venv/bin/python src/check_xgboost_runtime.py`.
