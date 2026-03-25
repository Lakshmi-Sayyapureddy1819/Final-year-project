# Multi-Source Dataset Guide

This guide maps the official data sources to the files supported by the project.

## Official sources

- NOAA OISST: [OISST official page](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
- Copernicus Marine Physics Reanalysis: [GLOBAL_MULTIYEAR_PHY_001_030](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
- Copernicus Ocean Colour / Chlorophyll: [OCEANCOLOUR_GLO_BGC_L3_MY_009_103](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_MY_009_103/description) and [MULTIOBS_GLO_BIO_BGC_3D_REP_015_010](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_BIO_BGC_3D_REP_015_010/description)
- CMFRI Fish Catch Estimates: [CMFRI fish catch estimates](https://www.cmfri.org.in/fish-catch-estimates)
- CMFRI methodology page: [CMFRI methodology](https://www.cmfri.org.in/methodology)
- INCOIS PFZ advisory: [INCOIS PFZ](https://services.incois.gov.in/MarineFisheries/PfzAdvisory.action)
- INCOIS PFZ text data: [INCOIS PFZ text data](https://services.incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en)
- FishBase maturity glossary: [FishBase length at first maturity](https://www.fishbase.se/glossary/Glossary.php?q=length+at+first+maturity)

## What is already implemented

- CMFRI annual state-wise landings are scraped into [data/cmfri_state_landings.csv](/Users/lakshmis/Final-year-project/data/cmfri_state_landings.csv).
- NOAA-derived SST is already used in [data/real_world_training_data.csv](/Users/lakshmis/Final-year-project/data/real_world_training_data.csv).
- Optional enrichment is supported via [src/build_multisource_dataset.py](/Users/lakshmis/Final-year-project/src/build_multisource_dataset.py).

## Why the row count is still limited

The current public CMFRI pages available to this repo are annual state-wise totals for 2013-2024. That gives a real but small table. Adding Copernicus and FishBase increases feature richness, but not the number of target rows, unless we obtain monthly or district-level landing data or PFZ archives.

## Best way to generate more real rows

Use one of these:

- CMFRI district-wise or month-wise landing files
- CMFRI fishing-zone data exports
- INCOIS PFZ archive files with dates and coordinates

Once any of those are available locally, the project can be upgraded from annual state-level rows to monthly or zone-level rows.
