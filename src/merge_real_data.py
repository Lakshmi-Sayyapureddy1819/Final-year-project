import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("📌 Loading datasets...")

sst = pd.read_csv("data/indian_sst.csv")
fish = pd.read_csv("data/clean_sst.csv")

# Rename for consistency
sst.rename(columns={"lat": "Latitude", "lon": "Longitude", "sst": "SST"}, inplace=True)

print("🔎 Matching nearest SST coordinates...")

# Extract coordinate pairs
sst_coords = np.array(list(zip(sst["Latitude"], sst["Longitude"])))
fish_coords = np.array(list(zip(fish["Latitude"], fish["Longitude"])))

# Nearest neighbor model
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(sst_coords)

distances, indices = nn.kneighbors(fish_coords)
matched_sst = sst.iloc[indices.flatten()][["SST"]].reset_index(drop=True)

merged = pd.concat([fish.reset_index(drop=True), matched_sst], axis=1)

merged.to_csv("data/final_training_data.csv", index=False)

print("🎉 Merged dataset created successfully!")
print("Rows:", len(merged))
print("Saved as data/final_training_data.csv")
