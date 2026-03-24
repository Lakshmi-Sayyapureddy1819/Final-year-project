from prediction_engine import predict_fishing_zone


PIPELINES = {
    "1": ("Random Forest", "random_forest"),
    "2": ("XGBoost", "xgboost"),
    "3": ("Hybrid (PCA + RF + Boosting)", "hybrid"),
}


def read_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a numeric value.")


def read_optional_float(prompt: str) -> float | None:
    raw = input(prompt).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid number. Skipping this optional input.")
        return None


print("AI-Driven Fish Catch Prediction System")
print("--------------------------------------")

location = input("Enter location name [Vizag]: ").strip() or "Vizag"
latitude = read_optional_float("Latitude (optional): ")
longitude = read_optional_float("Longitude (optional): ")

sst = read_float("Sea Surface Temperature (C): ")
salinity = read_float("Salinity (PSU): ")
dissolved_oxygen = read_float("Dissolved Oxygen (mg/l): ")
historical_catch = read_float("Previous Average Catch (kg): ")

print("\nChoose ML pipeline:")
for key, (label, _) in PIPELINES.items():
    print(f"{key}. {label}")
pipeline_key = input("Selection [1]: ").strip() or "1"
_, model_choice = PIPELINES.get(pipeline_key, PIPELINES["1"])

use_refinement = input("Add maturity-based juvenile risk? [y/N]: ").strip().lower() == "y"
observed_length = None
maturity_length = None

if use_refinement:
    observed_length = read_optional_float("Observed fish length in cm (optional): ")
    maturity_length = read_optional_float("Species maturity length in cm (optional): ")

result = predict_fishing_zone(
    location=location,
    sst=sst,
    salinity=salinity,
    dissolved_oxygen=dissolved_oxygen,
    historical_catch=historical_catch,
    latitude=latitude,
    longitude=longitude,
    model_choice=model_choice,
    observed_length_cm=observed_length,
    maturity_length_cm=maturity_length,
)

print("\nFinal Result")
print("------------")
print(f"Location: {result.location}")
print(f"Pipeline: {result.model_pipeline}")
print(f"Fish Availability: {'YES' if result.availability else 'NO'}")
print(f"Availability Score: {result.availability_score:.2f}")
print(f"Predicted Catch Quantity: {result.quantity:.2f} kg")
print(f"Juvenile Risk Level: {result.juvenile_risk}")
print(f"Base Juvenile Layer: {result.base_juvenile_risk}")

if result.maturity_score is not None:
    print(f"Maturity Risk Score: {result.maturity_score:.2f}")

print(f"Advisory: {result.advisory}")

if result.safe_zone_suggestions:
    print("\nSuggested Safer Zones")
    for zone in result.safe_zone_suggestions:
        print(
            f"- {zone['zone']}: {zone['distance_km']} km away at "
            f"({zone['latitude']}, {zone['longitude']}) | "
            f"Risk {zone['expected_juvenile_risk']} | "
            f"Expected catch {zone['expected_quantity_kg']} kg"
        )
