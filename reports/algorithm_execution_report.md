# Algorithm Execution Report

This report is intended for viva/demo use. It shows which algorithms are loaded, how they execute on sample inputs, and the exact project code blocks to explain.

## Runtime status

- `Random Forest`: `Ready`
  Classifier/logic: `sklearn.ensemble._forest.RandomForestClassifier`
  Regressor/detail: `sklearn.ensemble._forest.RandomForestRegressor`
- `Boosting`: `Ready`
  Classifier/logic: `sklearn.ensemble._gb.GradientBoostingClassifier`
  Regressor/detail: `sklearn.ensemble._gb.GradientBoostingRegressor`
- `Hybrid (PCA + RF + ET + Boosting)`: `Ready`
  Classifier/logic: `sklearn.ensemble._voting.VotingClassifier`
  Regressor/detail: `sklearn.ensemble._voting.VotingRegressor`
- `Juvenile ML layer`: `Ready`
  Classifier/logic: `sklearn.ensemble._forest.ExtraTreesClassifier`
  Regressor/detail: `-`
- `Exact maturity rule`: `Ready`
  Classifier/logic: `Rule-based formula: JR = 1 - observed_length / maturity_length`
  Regressor/detail: `FishBase rows: 7`

## Improvement comparison

- `Random Forest availability accuracy`
  Before balancing: `0.5909`
  After balancing: `0.6364`
  Interpretation: `Improved`
- `Boosting availability accuracy`
  Before balancing: `0.5`
  After balancing: `0.6818`
  Interpretation: `Dropped`
- `Hybrid availability accuracy`
  Before balancing: `0.4545`
  After balancing: `0.5909`
  Interpretation: `Improved`
- `Juvenile accuracy`
  Before balancing: `0.9091`
  After balancing: `0.7727`
  Interpretation: `More conservative`
- `Juvenile weighted F1`
  Before balancing: `0.8874`
  After balancing: `0.7538`
  Interpretation: `More realistic`
- `Random Forest quantity RMSE`
  Before balancing: `82478.3759`
  After balancing: `37039.0236`
  Interpretation: `Lower is better`
- `Boosting quantity RMSE`
  Before balancing: `91903.1661`
  After balancing: `50117.505`
  Interpretation: `Lower is better`
- `Hybrid quantity RMSE`
  Before balancing: `76830.9764`
  After balancing: `43769.41`
  Interpretation: `Lower is better`
- `Juvenile class counts`
  Before balancing: `{'Low': 75, 'Medium': 34, 'High': 1}`
  After balancing: `{'Medium': 37, 'High': 37, 'Low': 36}`
  Interpretation: `Balanced classes`

## Demo executions

### Random Forest pipeline

Executes the tabular Random Forest classifier and regressor with exact maturity refinement.

- Requested pipeline: `random_forest`
- Resolved pipeline: `Random Forest`
- Availability: `NO`
- Availability score: `0.463`
- Predicted quantity (kg): `93857.15`
- Juvenile risk: `High`
- Juvenile method: `Exact maturity rule (FishBase maturity reference + observed length)`
- Applied maturity length (cm): `16.3`
- Safe-zone suggestions: `4`
- Advisory: `High juvenile density detected. Avoid fishing here and shift 8-15 km toward a safer zone.`

### Boosting pipeline

Executes the boosting path. On this machine it resolves to native XGBoost if available, otherwise Gradient Boosting fallback.

- Requested pipeline: `xgboost`
- Resolved pipeline: `Gradient Boosting`
- Availability: `NO`
- Availability score: `0.37`
- Predicted quantity (kg): `53354.96`
- Juvenile risk: `High`
- Juvenile method: `Exact maturity rule (FishBase maturity reference + observed length)`
- Applied maturity length (cm): `16.3`
- Safe-zone suggestions: `4`
- Advisory: `High juvenile density detected. Avoid fishing here and shift 8-15 km toward a safer zone.`

### Hybrid PCA + RF + ET + Boosting pipeline

Executes the PCA-transformed ensemble path used in the hybrid model.

- Requested pipeline: `hybrid`
- Resolved pipeline: `Hybrid (PCA + RF + ET + Boosting)`
- Availability: `NO`
- Availability score: `0.515`
- Predicted quantity (kg): `78116.18`
- Juvenile risk: `High`
- Juvenile method: `Exact maturity rule (FishBase maturity reference + observed length)`
- Applied maturity length (cm): `16.3`
- Safe-zone suggestions: `4`
- Advisory: `High juvenile density detected. Avoid fishing here and shift 8-15 km toward a safer zone.`

### Exact juvenile-risk rule

Shows the biological maturity rule using FishBase maturity length plus observed fish length.

- Requested pipeline: `random_forest`
- Resolved pipeline: `Random Forest`
- Availability: `NO`
- Availability score: `0.46`
- Predicted quantity (kg): `55158.15`
- Juvenile risk: `High`
- Juvenile method: `Exact maturity rule (FishBase maturity reference + observed length)`
- Applied maturity length (cm): `16.3`
- Safe-zone suggestions: `0`
- Advisory: `High juvenile density detected. Avoid fishing here and shift 8-15 km toward a safer zone.`

### Environmental juvenile fallback

Shows the fallback juvenile model when species or maturity data are not provided.

- Requested pipeline: `random_forest`
- Resolved pipeline: `Random Forest`
- Availability: `NO`
- Availability score: `0.46`
- Predicted quantity (kg): `55158.15`
- Juvenile risk: `High`
- Juvenile method: `Environmental juvenile model fallback`
- Applied maturity length (cm): `None`
- Safe-zone suggestions: `0`
- Advisory: `High juvenile density detected. Avoid fishing here and shift 8-15 km toward a safer zone.`

## Code blocks for viva

### Pipeline Selection Logic

This block selects the Random Forest, Boosting, or Hybrid inference path during prediction.

Source: `/Users/lakshmis/Final-year-project/src/prediction_engine.py`

```python
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
```

### Boosting Model Builders

This block constructs the boosting models and transparently falls back to Gradient Boosting when native XGBoost is unavailable.

Source: `/Users/lakshmis/Final-year-project/src/model_training.py`

```python
def build_boosting_classifier():
    if xgb is not None:
        return xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    return GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)


def build_boosting_regressor():
    if xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
    return GradientBoostingRegressor(n_estimators=250, learning_rate=0.04, random_state=42)


def model_label() -> str:
    return "XGBoost" if xgb is not None else "Gradient Boosting"
```

### Exact Juvenile Risk Formula

This is the biological rule used when observed fish length and maturity length are available.

Source: `/Users/lakshmis/Final-year-project/src/juvenile_risk_utils.py`

```python
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
```

### Juvenile Training Data Preparation

This block merges heuristic juvenile labels with exact maturity-based labels for training.

Source: `/Users/lakshmis/Final-year-project/src/project_data_utils.py`

```python
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
```
