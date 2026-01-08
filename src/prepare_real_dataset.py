import pandas as pd

print("Preparing cleaned SST dataset...")

# Load Indian SST real dataset
df = pd.read_csv("data/indian_sst.csv")

# Keep only necessary columns
df = df[["time", "lat", "lon", "sst"]]

# Drop missing values
df = df.dropna()

# Convert date column to datetime format
df["time"] = pd.to_datetime(df["time"])

# Monthly average SST for stable prediction features
df = df.groupby([df["time"].dt.to_period("M"), "lat", "lon"])["sst"].mean().reset_index()

# Rename columns
df.rename(columns={
    "time": "Month",
    "lat": "Latitude",
    "lon": "Longitude",
    "sst": "SST"
}, inplace=True)

df["Month"] = df["Month"].astype(str)

# Save
df.to_csv("data/clean_sst.csv", index=False)

print("Cleaning complete. File saved as data/clean_sst.csv")
print(df.head())
