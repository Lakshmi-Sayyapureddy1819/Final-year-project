from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except Exception:
    xgb = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "final_training_data_fixed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MAX_TRAINING_ROWS = 60000


def build_boosting_classifier():
    if xgb is not None:
        return xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    return GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, random_state=42)


def build_boosting_regressor():
    if xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
    return GradientBoostingRegressor(n_estimators=80, learning_rate=0.08, random_state=42)


def model_label() -> str:
    return "XGBoost" if xgb is not None else "Gradient Boosting"


def prepare_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dataset = frame.copy()

    if "Availability" not in dataset.columns:
        dataset["Availability"] = dataset["Historical_Catch"].apply(lambda value: 1 if value > 300 else 0)

    if "Month" in dataset.columns:
        month_values = pd.to_datetime(dataset["Month"], errors="coerce").dt.month.fillna(1)
    else:
        month_values = pd.Series(np.ones(len(dataset)), index=dataset.index)

    angle = (2 * np.pi * month_values) / 12.0
    dataset["MonthSin"] = np.sin(angle)
    dataset["MonthCos"] = np.cos(angle)
    dataset["ThermalStress"] = (dataset["SST"] - 27.0).abs()
    dataset["OxygenStress"] = np.clip(5.5 - dataset["Dissolved_Oxygen"], 0.0, None)
    dataset["SalinityAnomaly"] = (dataset["Salinity"] - 34.0).abs()

    feature_columns = [
        "SST",
        "Salinity",
        "Dissolved_Oxygen",
        "Historical_Catch",
        "Latitude",
        "Longitude",
        "MonthSin",
        "MonthCos",
        "ThermalStress",
        "OxygenStress",
        "SalinityAnomaly",
    ]

    for column in feature_columns:
        if column not in dataset.columns:
            dataset[column] = 0.0

    return dataset, feature_columns


print("Loading training dataset...")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA_PATH)
df, feature_columns = prepare_training_frame(df)

if len(df) > MAX_TRAINING_ROWS:
    df = df.sample(n=MAX_TRAINING_ROWS, random_state=42).sort_index().reset_index(drop=True)
    print(f"Using a sampled training subset of {len(df)} rows for faster local retraining.")

X = df[feature_columns]
y_class = df["Availability"]
y_reg = df["Historical_Catch"]

X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X,
    y_class,
    y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_class,
)

print("\nTraining Random Forest models...")
rf_clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train_class)
rf_class_pred = rf_clf.predict(X_test)
print("Random Forest Availability Accuracy:", round(accuracy_score(y_test_class, rf_class_pred), 4))

rf_reg = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train_reg)
rf_reg_pred = rf_reg.predict(X_test)
rf_rmse = float(np.sqrt(mean_squared_error(y_test_reg, rf_reg_pred)))
print("Random Forest Quantity RMSE:", round(rf_rmse, 4))

print(f"\nTraining {model_label()} models...")
xgb_clf = build_boosting_classifier()
xgb_clf.fit(X_train, y_train_class)
xgb_class_pred = xgb_clf.predict(X_test)
print(f"{model_label()} Availability Accuracy:", round(accuracy_score(y_test_class, xgb_class_pred), 4))

xgb_reg = build_boosting_regressor()
xgb_reg.fit(X_train, y_train_reg)
xgb_reg_pred = xgb_reg.predict(X_test)
xgb_rmse = float(np.sqrt(mean_squared_error(y_test_reg, xgb_reg_pred)))
print(f"{model_label()} Quantity RMSE:", round(xgb_rmse, 4))

joblib.dump(rf_clf, MODELS_DIR / "availability_model.pkl")
joblib.dump(rf_reg, MODELS_DIR / "quantity_model.pkl")
joblib.dump(xgb_clf, MODELS_DIR / "xgb_availability_model.pkl")
joblib.dump(xgb_reg, MODELS_DIR / "xgb_quantity_model.pkl")

print(f"\nTraining Hybrid PCA + RF + {model_label()} models...")
pca_components = min(6, X_train.shape[1])
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

rf_clf_pca = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
xgb_clf_pca = build_boosting_classifier()
hybrid_clf = VotingClassifier([("rf", rf_clf_pca), ("xgb", xgb_clf_pca)], voting="soft")
hybrid_clf.fit(X_train_pca, y_train_class)
hybrid_class_pred = hybrid_clf.predict(X_test_pca)
print("Hybrid Availability Accuracy:", round(accuracy_score(y_test_class, hybrid_class_pred), 4))

rf_reg_pca = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
xgb_reg_pca = build_boosting_regressor()
hybrid_reg = VotingRegressor([("rf", rf_reg_pca), ("xgb", xgb_reg_pca)])
hybrid_reg.fit(X_train_pca, y_train_reg)
hybrid_reg_pred = hybrid_reg.predict(X_test_pca)
hybrid_rmse = float(np.sqrt(mean_squared_error(y_test_reg, hybrid_reg_pred)))
print("Hybrid Quantity RMSE:", round(hybrid_rmse, 4))

joblib.dump(pca, MODELS_DIR / "pca_transform.pkl")
joblib.dump(hybrid_clf, MODELS_DIR / "hybrid_availability_model.pkl")
joblib.dump(hybrid_reg, MODELS_DIR / "hybrid_quantity_model.pkl")

print("\nAll models saved successfully.")
