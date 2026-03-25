# Optional External Datasets

Place manually downloaded official files here to enrich the training dataset beyond CMFRI + NOAA SST.

## Supported file names

- `copernicus_physics.nc`
  Use a Copernicus Marine physics product containing salinity and ideally mixed layer depth or temperature.
- `copernicus_chlorophyll.nc`
  Use a Copernicus chlorophyll product containing a chlorophyll variable such as `chl`, `CHL`, or `chla`.
- `incois_pfz.csv`
  Expected columns:
  - `State`
  - `Date`
  Optional:
  - `PFZ_Count`
  - `Distance_km`
  - `Depth_m`
  - `Species`
  - `Observed_Length_cm`
  - `Maturity_Length_cm`
- `fishbase_maturity.csv`
  Expected columns:
  - `Species`
  - `Maturity_Length_cm`

## Included templates

- [incois_pfz.template.csv](/Users/lakshmis/Final-year-project/data/external/incois_pfz.template.csv)
- [fishbase_maturity.template.csv](/Users/lakshmis/Final-year-project/data/external/fishbase_maturity.template.csv)

## Build command

After placing the files here, run:

```bash
python src/check_external_datasets.py
python src/build_multisource_dataset.py
```

If `incois_pfz.csv` contains `Species` and `Observed_Length_cm`, the build step will also derive exact maturity-based juvenile features by joining FishBase maturity lengths.

You can also create and append rows directly from the Streamlit app sidebar. The app writes real observation rows into [incois_pfz.csv](/Users/lakshmis/Final-year-project/data/external/incois_pfz.csv).
