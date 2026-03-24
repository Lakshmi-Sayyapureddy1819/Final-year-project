# AI-Driven Fish Catch Prediction System

This repository contains the working code for a final-year project on fish availability prediction, juvenile-risk screening, catch quantity estimation, and safe-zone recommendation.

## What the code now includes

- A shared prediction engine in [src/prediction_engine.py](/Users/lakshmis/Final-year-project/src/prediction_engine.py) that keeps the CLI app, Streamlit app, and map heatmap aligned.
- A juvenile-risk layer that combines the trained juvenile model with maturity-based scoring using `JR = 1 - observed_length / maturity_length`.
- Safe-zone suggestions 8-15 km away when the selected zone is high-risk or not recommended.
- Training scripts that use robust project-relative paths instead of fragile `../models/...` assumptions.

## Current project structure

- [src/app.py](/Users/lakshmis/Final-year-project/src/app.py): main Streamlit application.
- [src/map_app.py](/Users/lakshmis/Final-year-project/src/map_app.py): heatmap explorer for safer candidate zones.
- [src/main.py](/Users/lakshmis/Final-year-project/src/main.py): command-line interface.
- [src/model_training.py](/Users/lakshmis/Final-year-project/src/model_training.py): Random Forest, XGBoost, and Hybrid PCA + RF + XGBoost training flow.
- [src/juvenile_risk_model.py](/Users/lakshmis/Final-year-project/src/juvenile_risk_model.py): juvenile-risk model training flow.

## Final-year project methodology

This project now follows a no-CNN methodology:

- Fish availability prediction using machine learning classifiers such as Random Forest and XGBoost.
- Catch quantity prediction using regression models such as Random Forest Regressor and XGBoost Regressor.
- Juvenile-risk assessment using species maturity length and observed fish length instead of any video-based detection pipeline.
- Safe-zone recommendation by shifting 8-15 km toward lower-risk nearby coordinates.

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train or retrain models:

```bash
python src/juvenile_risk_model.py
python src/model_training.py
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
