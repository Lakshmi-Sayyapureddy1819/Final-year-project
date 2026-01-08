import pandas as pd
import numpy as np

df = pd.read_csv("data/final_training_data.csv")

# Rename duplicate SST column
df = df.rename(columns={"SST.1": "SST_Extra"})

# Add synthetic values (temporary placeholders)
np.random.seed(42)
df["Salinity"] = np.random.uniform(28, 36, size=len(df))
df["Dissolved_Oxygen"] = np.random.uniform(4.5, 7.5, size=len(df))
df["Historical_Catch"] = np.random.randint(50, 800, size=len(df))

# Save updated dataset
df.to_csv("data/final_training_data_fixed.csv", index=False)
print("Final dataset prepared successfully! Rows:", len(df))
