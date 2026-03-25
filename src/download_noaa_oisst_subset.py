from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "indian_sst_from_noaa.csv"

NOAA_OISST_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.nc"

# Rough India-region bounds used for this project.
LAT_MIN = 5.0
LAT_MAX = 25.0
LON_MIN = 65.0
LON_MAX = 95.0
START_DATE = "2013-01-01"
END_DATE = "2024-12-31"


def main() -> None:
    print("Opening NOAA OISST via OPeNDAP...")
    dataset = xr.open_dataset(NOAA_OISST_URL)
    subset = dataset.sel(
        time=slice(START_DATE, END_DATE),
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )

    frame = subset[["sst"]].to_dataframe().reset_index().dropna()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved NOAA OISST subset to {OUTPUT_PATH}")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
