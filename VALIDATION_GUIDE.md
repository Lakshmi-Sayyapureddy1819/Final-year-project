# Validation Guide

This guide shows how to test, verify, and explain the fish-catch prediction project during demo and viva.

## 1. Run the full pipeline

Use the local virtual environment:

```bash
.venv/bin/python src/run_full_pipeline.py
```

If you want to refresh the CMFRI dataset first:

```bash
.venv/bin/python src/run_full_pipeline.py --refresh-cmfri
```

## 2. Run the automated validation report

```bash
.venv/bin/python src/validate_project.py
```

This generates:

- [latest_validation_report.md](/Users/lakshmis/Final-year-project/reports/latest_validation_report.md)
- [latest_metrics.json](/Users/lakshmis/Final-year-project/reports/latest_metrics.json)

## 3. Generate the viva algorithm report

```bash
.venv/bin/python src/demo_algorithms.py
```

This generates:

- [algorithm_execution_report.md](/Users/lakshmis/Final-year-project/reports/algorithm_execution_report.md)

This report is useful in viva because it:

- proves that `Random Forest`, `Boosting`, `Hybrid`, and both juvenile-risk paths executed
- shows the actual implementation class used at runtime
- includes project code blocks you can explain to the teachers

## 4. Run the logic tests

```bash
.venv/bin/python -m unittest discover -s tests
```

These tests verify:

- FishBase species maturity lookup works.
- Exact juvenile formula is applied when species and observed length are available.
- Environmental fallback works when exact biological inputs are missing.
- All three inference pipelines execute without error.

## 4A. Add real field observations

You can now record real observation rows directly from the Streamlit sidebar in [src/app.py](/Users/lakshmis/Final-year-project/src/app.py).

Each row is saved into:

- [incois_pfz.csv](/Users/lakshmis/Final-year-project/data/external/incois_pfz.csv)

You can also batch import a CSV file:

```bash
.venv/bin/python src/import_pfz_observations.py your_file.csv
```

To make a row usable for exact juvenile-risk training, fill:

- `State`
- `Date`
- `Species`
- `Observed_Length_cm`
- `Maturity_Length_cm`

After adding new rows, rerun:

```bash
.venv/bin/python src/run_full_pipeline.py
```

If you want to check whether native XGBoost can run on this Mac:

```bash
.venv/bin/python src/check_xgboost_runtime.py
```

## 5. What to verify in the app

Open [src/app.py](/Users/lakshmis/Final-year-project/src/app.py) in Streamlit and use the `Viva Algorithm Demo` section to:

- run all algorithms with one button
- show the runtime backend for each pipeline
- open the exact code blocks for pipeline selection, boosting, and juvenile logic

Then try these domain-specific cases:

1. Species: `Sardinella longiceps`, Observed length: `12 cm`
   Expected result:
   - Juvenile method should say `Exact maturity rule`
   - Juvenile risk should be `High`
   - Maturity length should show about `16.3 cm`

2. Species: `Sardinella longiceps`, Observed length: `18 cm`
   Expected result:
   - Juvenile method should say `Exact maturity rule`
   - Juvenile risk should be `Low`

3. No species and no maturity inputs
   Expected result:
   - Juvenile method should say `Environmental juvenile model fallback`

4. Record one real field observation in the sidebar
   Expected result:
   - [incois_pfz.csv](/Users/lakshmis/Final-year-project/data/external/incois_pfz.csv) row count increases
   - `exact-ready rows` increases if species and lengths are filled
   - after rerunning the pipeline, the validation report should reflect the new field row counts

## 6. How to explain validation in the report

Use these three validation levels:

- Data validation:
  verify dataset source, row counts, required columns, and missing external files.
- Model validation:
  measure availability accuracy, juvenile accuracy, quantity RMSE/MAE/R2.
- Logic validation:
  confirm the exact maturity formula behaves correctly on known biological cases.

## 7. How to explain current limitations honestly

- The current merged training dataset is real, but still small.
- Exact juvenile-risk training labels need real `Species` + `Observed_Length_cm` records in PFZ or fisheries survey data.
- Until those rows are available, the trained juvenile model uses the environmental heuristic layer, while the live app still applies the exact maturity rule whenever species and observed length are entered.
