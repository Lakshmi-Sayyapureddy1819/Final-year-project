import xarray as xr

print("Downloading NOAA SST Dataset... Please wait...")

url = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/sst.mnmean.nc"
data = xr.open_dataset(url)

print("Dataset Structure:")
print(data)

df = data.to_dataframe().reset_index()
df.to_csv("../data/noaa_sst.csv", index=False)

print("NOAA SST dataset saved successfully to data/noaa_sst.csv")
