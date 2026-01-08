import pandas as pd

print("Merging SST data with fishing port locations...")

# Load cleaned SST dataset
sst = pd.read_csv("data/clean_sst.csv")

# Define fishing port locations
ports = [
    ["Vizag", 17.6868, 83.2185],
    ["Kakinada", 16.9891, 82.2475],
    ["Machilipatnam", 16.1875, 81.1381],
    ["Chennai", 13.0827, 80.2707],
    ["Kochi", 9.9312, 76.2673],
    ["Goa", 15.2993, 74.1240],
    ["Mumbai", 19.0760, 72.8777],
]

ports_df = pd.DataFrame(ports, columns=["Location", "Latitude", "Longitude"])

# Round coordinates to match SST grid alignment
sst["Latitude"] = sst["Latitude"].round(1)
sst["Longitude"] = sst["Longitude"].round(1)
ports_df["Latitude"] = ports_df["Latitude"].round(1)
ports_df["Longitude"] = ports_df["Longitude"].round(1)

# Merge SST with port coordinates
merged = pd.merge(sst, ports_df, on=["Latitude", "Longitude"], how="inner")

# Approximate historical catch estimation (sample real-like assumptions)
merged["Historical_Catch"] = (merged["SST"] - merged["SST"].min()) * 50  # temp-based scaling example

# Fish Availability Rule: SST Range 25–30°C is best
merged["Availability"] = merged["SST"].apply(lambda x: 1 if 25 <= x <= 30 else 0)

# Juvenile Risk heuristic (SST extremes correlate with juvenile presence)
merged["Juvenile_Risk"] = merged["SST"].apply(
    lambda x: "High" if x > 30 or x < 23 else ("Medium" if 23 <= x < 25 else "Low")
)

merged.to_csv("data/final_training_data.csv", index=False)

print("Merged dataset created successfully -> data/final_training_data.csv")
print(merged.head())
