from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from project_data_utils import balance_classification_frame, prepare_main_training_frame, resolve_primary_dataset_path

try:
    import xgboost as xgb
except Exception:
    xgb = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MAX_TRAINING_ROWS = 60000


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


print("Loading training dataset...")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
data_path = resolve_primary_dataset_path()
df = pd.read_csv(data_path)
df, feature_columns = prepare_main_training_frame(df)
target_column = "Landings_Tonnes" if "Landings_Tonnes" in df.columns else "Historical_Catch"
print(f"Using dataset: {data_path}")
if xgb is None:
    print("XGBoost runtime not available on this machine; training Boosting models with Gradient Boosting fallback.")

if len(df) > MAX_TRAINING_ROWS:
    df = df.sample(n=MAX_TRAINING_ROWS, random_state=42).sort_index().reset_index(drop=True)
    print(f"Using a sampled training subset of {len(df)} rows for faster local retraining.")

X = df[feature_columns]
y_class = df["Availability"]
y_reg = df[target_column]

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

print("Availability class counts before balancing:", y_train_class.value_counts().to_dict())
X_train_class_balanced, y_train_class_balanced = balance_classification_frame(
    X_train_class,
    y_train_class,
    target_name="Availability",
    random_state=42,
)
print("Availability class counts after balancing:", y_train_class_balanced.value_counts().to_dict())

print("\nTraining Random Forest models...")
rf_clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
rf_clf.fit(X_train_class_balanced, y_train_class_balanced)
rf_class_pred = rf_clf.predict(X_test_class)
print("Random Forest Availability Accuracy:", round(accuracy_score(y_test_class, rf_class_pred), 4))

rf_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg, y_train_reg)
rf_reg_pred = rf_reg.predict(X_test_reg)
rf_rmse = float(np.sqrt(mean_squared_error(y_test_reg, rf_reg_pred)))
print("Random Forest Quantity RMSE:", round(rf_rmse, 4))

print(f"\nTraining {model_label()} models...")
xgb_clf = build_boosting_classifier()
xgb_clf.fit(X_train_class_balanced, y_train_class_balanced)
xgb_class_pred = xgb_clf.predict(X_test_class)
print(f"{model_label()} Availability Accuracy:", round(accuracy_score(y_test_class, xgb_class_pred), 4))

xgb_reg = build_boosting_regressor()
xgb_reg.fit(X_train_reg, y_train_reg)
xgb_reg_pred = xgb_reg.predict(X_test_reg)
xgb_rmse = float(np.sqrt(mean_squared_error(y_test_reg, xgb_reg_pred)))
print(f"{model_label()} Quantity RMSE:", round(xgb_rmse, 4))

joblib.dump(rf_clf, MODELS_DIR / "availability_model.pkl")
joblib.dump(rf_reg, MODELS_DIR / "quantity_model.pkl")
joblib.dump(xgb_clf, MODELS_DIR / "xgb_availability_model.pkl")
joblib.dump(xgb_reg, MODELS_DIR / "xgb_quantity_model.pkl")

print(f"\nTraining Hybrid PCA + RF + {model_label()} models...")
pca_components = min(8, X_train_class.shape[1])
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train_reg)
X_test_pca = pca.transform(X_test_reg)
X_train_class_balanced_pca = pca.transform(X_train_class_balanced)
X_test_class_pca = pca.transform(X_test_class)

rf_clf_pca = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
et_clf_pca = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
xgb_clf_pca = build_boosting_classifier()
hybrid_clf = VotingClassifier([("rf", rf_clf_pca), ("et", et_clf_pca), ("xgb", xgb_clf_pca)], voting="soft")
hybrid_clf.fit(X_train_class_balanced_pca, y_train_class_balanced)
hybrid_class_pred = hybrid_clf.predict(X_test_class_pca)
print("Hybrid Availability Accuracy:", round(accuracy_score(y_test_class, hybrid_class_pred), 4))

rf_reg_pca = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
et_reg_pca = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)
xgb_reg_pca = build_boosting_regressor()
hybrid_reg = VotingRegressor([("rf", rf_reg_pca), ("et", et_reg_pca), ("xgb", xgb_reg_pca)])
hybrid_reg.fit(X_train_pca, y_train_reg)
hybrid_reg_pred = hybrid_reg.predict(X_test_pca)
hybrid_rmse = float(np.sqrt(mean_squared_error(y_test_reg, hybrid_reg_pred)))
print("Hybrid Quantity RMSE:", round(hybrid_rmse, 4))

joblib.dump(pca, MODELS_DIR / "pca_transform.pkl")
joblib.dump(hybrid_clf, MODELS_DIR / "hybrid_availability_model.pkl")
joblib.dump(hybrid_reg, MODELS_DIR / "hybrid_quantity_model.pkl")

print("\nAll models saved successfully.")
