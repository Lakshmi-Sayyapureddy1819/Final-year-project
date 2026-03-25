# Real-World Data Sources

This project can now be upgraded from placeholder data to a real-data pipeline using these sources:

- CMFRI official state-wise marine fish landing estimates: used for annual landings by Indian coastal state.
- Indian SST dataset already present in the repository: used as the environmental SST signal.
- Maturity-length data from FishBase or project-specific fisheries references: used at inference time for maturity-based juvenile risk.

## Implemented data scripts

- [src/fetch_cmfri_state_landings.py](/Users/lakshmis/Final-year-project/src/fetch_cmfri_state_landings.py)
  Downloads and parses official CMFRI state-wise landings pages for 2013-2024.
- [src/build_real_world_dataset.py](/Users/lakshmis/Final-year-project/src/build_real_world_dataset.py)
  Merges CMFRI landings with yearly SST features from [data/indian_sst.csv](/Users/lakshmis/Final-year-project/data/indian_sst.csv) and produces [data/real_world_training_data.csv](/Users/lakshmis/Final-year-project/data/real_world_training_data.csv).

## Expected outputs

- `data/cmfri_state_landings.csv`
- `data/real_world_training_data.csv`

## Notes

- The real-world dataset is annual state-level data, not frame-level or video-level data.
- Juvenile-risk remains maturity-based, which is more defensible than claiming a video model without labeled video data.
