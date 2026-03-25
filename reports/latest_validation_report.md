# Validation Report

- Dataset path: `/Users/lakshmis/Final-year-project/data/multisource_training_data.csv`
- Dataset rows: `110`
- Juvenile exact-label rows in training data: `3`
- Field observation rows: `4`
- Field exact-ready rows: `3`

## Main model metrics

- Availability class counts: `{'0': 50, '1': 60}`
- Random Forest availability accuracy: `0.6364`
- Random Forest quantity RMSE: `37039.0236`
- Boosting availability accuracy: `0.6818`
- Boosting quantity RMSE: `50117.505`
- Boosting implementation class: `sklearn.ensemble._gb.GradientBoostingClassifier`
- Hybrid availability accuracy: `0.5909`
- Hybrid quantity RMSE: `43769.41`

## Juvenile model metrics

- Juvenile class counts: `{'Medium': 37, 'High': 37, 'Low': 36}`
- Juvenile accuracy: `0.7727`
- Juvenile weighted F1: `0.7538`
- Juvenile risk-source counts: `{'Environmental heuristic': 107, 'Exact maturity rule': 3}`

## Demo verification checks

- PASS: `Exact maturity high-risk case` -> risk `High`, method `Exact maturity rule (FishBase maturity reference + observed length)`
- PASS: `Exact maturity low-risk case` -> risk `Low`, method `Exact maturity rule (FishBase maturity reference + observed length)`
- PASS: `Environmental fallback case` -> risk `High`, method `Environmental juvenile model fallback`
