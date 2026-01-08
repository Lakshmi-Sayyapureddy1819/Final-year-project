import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

data = {
    "SST": np.random.uniform(20, 32, rows),
    "Salinity": np.random.uniform(28, 36, rows),
    "Dissolved_Oxygen": np.random.uniform(3, 9, rows),
    "Season": np.random.choice(["Summer", "Winter", "Monsoon"], rows),
    "Historical_Catch": np.random.uniform(50, 1000, rows),
}

df = pd.DataFrame(data)
df.to_csv("../data/fish_dataset.csv", index=False)

print("Dataset generated successfully!")
