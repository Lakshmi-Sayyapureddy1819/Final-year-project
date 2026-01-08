import joblib
import numpy as np

# Load Models
clf = joblib.load("../models/availability_model.pkl")
reg = joblib.load("../models/quantity_model.pkl")
j_model = joblib.load("../models/juvenile_model.pkl")

print("---- AI-Driven Fish Catch Prediction System ----")

# User Inputs
SST = float(input("Enter Sea Surface Temperature (°C): "))
Salinity = float(input("Enter Salinity (PSU): "))
DO = float(input("Enter Dissolved Oxygen (mg/l): "))
History = float(input("Enter Previous Average Catch (kg): "))

# Main model features
features = np.array([[SST, Salinity, DO, History]])

# Juvenile model uses only 3 features
juvenile_features = np.array([[SST, Salinity, DO]])

# Predictions
availability = clf.predict(features)[0]
quantity = reg.predict(features)[0]
juvenile_risk = j_model.predict(juvenile_features)[0]

print("\n---------- FINAL RESULT ----------")
print("Juvenile Risk Level:", juvenile_risk)

# Decision Engine
if juvenile_risk == "High":
    print("⚠️ High Juvenile Density Detected — Fishing Not Recommended!")
    print("➡ Suggested shift: Move 8–15 km from the current location.")
else:
    if availability == 1:
        print("Fish Availability: YES 🟢")
        print(f"Predicted Catch Quantity: {quantity:.2f} kg")
    else:
        print("Fish Availability: NO 🔴")
