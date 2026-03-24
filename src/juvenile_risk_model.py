from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "final_training_data_fixed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "juvenile_model.pkl"


def prepare_juvenile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    dataset["ThermalStress"] = (dataset["SST"] - 27.0).abs()
    dataset["OxygenStress"] = np.clip(5.5 - dataset["Dissolved_Oxygen"], 0.0, None)
    dataset["SalinityAnomaly"] = (dataset["Salinity"] - 34.0).abs()

    catch_percentile = dataset["Historical_Catch"].rank(pct=True)
    risk_score = (
        0.45 * (1.0 - catch_percentile)
        + 0.3 * np.clip(dataset["ThermalStress"] / 4.0, 0.0, 1.0)
        + 0.25 * np.clip(dataset["OxygenStress"] / 2.5, 0.0, 1.0)
    )

    dataset["Juvenile_Risk"] = np.select(
        [risk_score >= 0.67, risk_score >= 0.4],
        ["High", "Medium"],
        default="Low",
    )
    return dataset


print("Loading training data for juvenile-risk model...")
df = pd.read_csv(DATA_PATH)
df = prepare_juvenile_frame(df)

feature_columns = [
    "SST",
    "Salinity",
    "Historical_Catch",
    "Dissolved_Oxygen",
    "ThermalStress",
    "OxygenStress",
    "SalinityAnomaly",
]

X = df[feature_columns]
y = df["Juvenile_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Juvenile Risk Accuracy:", round(accuracy_score(y_test, predictions), 4))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Juvenile-risk model saved to {MODEL_PATH}")
