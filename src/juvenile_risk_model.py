from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from project_data_utils import (
    JUVENILE_FEATURE_COLUMNS,
    balance_classification_frame,
    prepare_juvenile_training_frame,
    resolve_primary_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "juvenile_model.pkl"


print("Loading training data for juvenile-risk model...")
data_path = resolve_primary_dataset_path()
df = pd.read_csv(data_path)
print(f"Using dataset: {data_path}")
df, exact_label_count = prepare_juvenile_training_frame(df)
if exact_label_count:
    print(f"Applied exact maturity-based juvenile labels to {exact_label_count} rows.")
else:
    print("No observed-length maturity pairs found; using environmental juvenile labels for training.")

X = df[JUVENILE_FEATURE_COLUMNS]
y = df["Juvenile_Risk"]

stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_target,
)

print("Juvenile class counts before balancing:", y_train.value_counts().to_dict())
X_train_balanced, y_train_balanced = balance_classification_frame(
    X_train,
    y_train,
    target_name="Juvenile_Risk",
    random_state=42,
)
print("Juvenile class counts after balancing:", y_train_balanced.value_counts().to_dict())

model = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
model.fit(X_train_balanced, y_train_balanced)

predictions = model.predict(X_test)
print("Juvenile Risk Accuracy:", round(accuracy_score(y_test, predictions), 4))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Juvenile-risk model saved to {MODEL_PATH}")
