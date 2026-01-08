import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)

rows = 1000

data = {
    "SST": np.random.uniform(20, 32, rows),
    "Salinity": np.random.uniform(28, 36, rows),
    "Historical_Catch": np.random.uniform(50, 1000, rows),
}

df = pd.DataFrame(data)

# Create synthetic labels for juvenile risk
conditions = [
    (df["Historical_Catch"] < 250),
    (df["Historical_Catch"] >= 250) & (df["Historical_Catch"] <= 600),
    (df["Historical_Catch"] > 600)
]

labels = ["High", "Medium", "Low"]  # Label order based on fishing risk

df["Juvenile_Risk"] = np.select(conditions, labels, default="Medium")


X = df[["SST", "Salinity", "Historical_Catch"]]
y = df["Juvenile_Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Juvenile Risk Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "../models/juvenile_model.pkl")
print("Juvenile Risk Model saved successfully!")
