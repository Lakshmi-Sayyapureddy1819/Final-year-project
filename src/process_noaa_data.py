import xarray as xr
import pandas as pd

print("Downloading & Extracting SST dataset...")

url = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/sst.mnmean.nc"
data = xr.open_dataset(url)

subset = data.sel(
    lat=slice(25, 5),     # 25°N to 5°N
    lon=slice(65, 90)     # 65°E to 90°E
)

sst_df = subset['sst'].to_dataframe().reset_index()

sst_df.to_csv("data/indian_sst.csv", index=False)

print("Completed. File saved: data/indian_sst.csv")
print(sst_df.head())
