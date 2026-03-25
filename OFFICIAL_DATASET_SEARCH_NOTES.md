# Official Dataset Search Notes

These notes summarize what was found from the official sources during setup.

## NOAA OISST

- Official page: [NOAA OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
- Metadata page: [NOAA OISST v2.1 metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC01606)
- What was found:
  - 0.25 degree grid
  - daily resolution
  - temporal span from September 1981 to present
  - NetCDF access is available
- Repo support:
  - [src/download_noaa_oisst_subset.py](/Users/lakshmis/Final-year-project/src/download_noaa_oisst_subset.py)

## Copernicus Marine Physics

- Official product page: [GLOBAL_MULTIYEAR_PHY_001_030](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
- What was found:
  - daily and monthly data
  - variables include salinity, temperature, currents, mixed layer depth
  - NetCDF-4 format
  - temporal span from January 1, 1993 to February 24, 2026 on the product page seen during search
- Important constraint:
  - download typically requires a Copernicus Marine account or toolbox login
- Repo support:
  - [src/copernicus_subset_example.py](/Users/lakshmis/Final-year-project/src/copernicus_subset_example.py)

## Copernicus Chlorophyll

- Official product page: [OCEANCOLOUR_GLO_BGC_L3_MY_009_103](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_MY_009_103/description)
- What was found:
  - daily chlorophyll data
  - 4 km resolution
  - NetCDF-4 format
  - temporal span from September 4, 1997 to late 2025 on the product page seen during search
- Important constraint:
  - download typically requires a Copernicus Marine account or toolbox login
- Repo support:
  - [src/copernicus_subset_example.py](/Users/lakshmis/Final-year-project/src/copernicus_subset_example.py)

## CMFRI Fish Landings

- Official page: [CMFRI Fish Catch Estimates](https://www.cmfri.org.in/fish-catch-estimates)
- Methodology page: [CMFRI Methodology](https://www.cmfri.org.in/methodology)
- What was found:
  - public state-wise annual landing pages exist for 2013 through 2024
  - methodology page states CMFRI maintains richer monthly, district, and fishing-zone-style fisheries data internally
- Repo support:
  - [src/fetch_cmfri_state_landings.py](/Users/lakshmis/Final-year-project/src/fetch_cmfri_state_landings.py)
  - [src/build_real_world_dataset.py](/Users/lakshmis/Final-year-project/src/build_real_world_dataset.py)

## INCOIS PFZ

- Official text-data page: [INCOIS PFZ text data](https://services.incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en)
- Marine fisheries advisory page: [INCOIS PFZ advisory](https://services.incois.gov.in/MarineFisheries/PfzAdvisory.action)
- What was found:
  - public advisory pages and sector lists exist, including North and South Andhra Pradesh sectors
  - the page shows current forecast/advisory access
- Important constraint:
  - a clean public bulk archive export was not confirmed during search
  - this source may need manual collection/export into CSV
- Repo support:
  - place a manually prepared CSV at [incois_pfz.csv](/Users/lakshmis/Final-year-project/data/external/incois_pfz.template.csv)

## FishBase Maturity

- Official glossary: [FishBase size at first maturity](https://www.fishbase.se/glossary/Glossary.php?q=size-at-first-maturity)
- What was found:
  - FishBase publicly defines size at first maturity
  - this is suitable for maturity-based juvenile-risk scoring
- Important constraint:
  - a simple official bulk CSV download was not confirmed during search
  - a small project-specific CSV may need to be prepared manually
- Repo support:
  - place a manually prepared CSV at [fishbase_maturity.csv](/Users/lakshmis/Final-year-project/data/external/fishbase_maturity.template.csv)
