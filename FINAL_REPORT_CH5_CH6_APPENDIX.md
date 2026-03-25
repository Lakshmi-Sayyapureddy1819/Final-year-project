# Chapter 5 Conclusions and Future Scope of Work

## 5.1 Conclusion

This project presents an AI-Driven Fish Catch Prediction System with Juvenile Risk Assessment for Sustainable Fisheries. The implemented system integrates real-world fisheries data, SST-linked environmental indicators, machine learning models, maturity-based biological logic, and a deployable user interface. Unlike unsupported image-based drafts, the final implementation is grounded in verifiable datasets and executable algorithms.

The project demonstrates that a layered tabular machine learning architecture can support fish availability prediction, catch quantity estimation, juvenile-risk assessment, and safe-zone recommendation in a single workflow. Based on the latest validation results, the system achieved:

- Boosting availability accuracy: `0.6818`
- Random Forest quantity RMSE: `37039.0236`
- Juvenile-risk classification accuracy: `0.7727`
- Juvenile-risk weighted F1: `0.7538`

These results show that the proposed system is suitable as a final-year project prototype and offers a realistic base for further research and field deployment.

## 5.2 Contributions of the Proposed System

- Developed a real-data fisheries prediction workflow using CMFRI landings, SST-derived environmental features, FishBase maturity references, and PFZ-style field observation support.
- Implemented multiple machine learning pipelines for fish availability and catch quantity prediction.
- Introduced an exact maturity-based juvenile-risk rule:

```text
JR = 1 - (Observed Length / Maturity Length)
```

- Built a safe-zone recommendation engine for sustainability-aware decision support.
- Developed an interactive Streamlit application supporting manual, region-based, and map-based predictions.
- Added field-data recording, validation reporting, automated tests, and report-figure generation.

## 5.3 Practical Implications and Applications

The practical importance of the proposed system lies in its balance between productivity and sustainability. The application can be used as:

- an academic prototype for intelligent fisheries decision support,
- a sustainability-awareness tool that warns users about juvenile-risk before recommending fishing activity,
- a comparative study platform for machine learning models on real marine tabular data,
- a foundation for future integration with PFZ, Copernicus, and advanced oceanographic sources.

### Suggested Figures

![Improvement Comparison](/Users/lakshmis/Final-year-project/reports/figures/improvement_comparison.png)

Figure 5.1: Overall model improvement after feature engineering, class balancing, and retraining.

## 5.4 Recommendations for Future Work

- Integrate monthly, district-level, or fishing-zone-level CMFRI landing data.
- Integrate Copernicus salinity, chlorophyll, currents, and mixed-layer-depth variables.
- Expand PFZ and field observation records with more exact-ready species and observed-length entries.
- Enable native XGBoost by installing the OpenMP runtime on macOS.
- Introduce temporal forecasting, spatial interpolation, and uncertainty reporting.
- Conduct validation using real harbour or onboard observation campaigns.

# Chapter 6 References

## References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189-1232.
3. Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely Randomized Trees. *Machine Learning*, 63, 3-42.
4. CMFRI Fish Catch Estimates. [https://www.cmfri.org.in/fish-catch-estimates](https://www.cmfri.org.in/fish-catch-estimates)
5. CMFRI Methodology. [https://www.cmfri.org.in/methodology](https://www.cmfri.org.in/methodology)
6. NOAA Optimum Interpolation Sea Surface Temperature. [https://www.ncei.noaa.gov/products/optimum-interpolation-sst](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
7. Copernicus Marine Physics Reanalysis. [https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
8. Copernicus Ocean Colour Chlorophyll Product. [https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_MY_009_103/description](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_MY_009_103/description)
9. INCOIS PFZ Advisory. [https://services.incois.gov.in/MarineFisheries/PfzAdvisory.action](https://services.incois.gov.in/MarineFisheries/PfzAdvisory.action)
10. INCOIS Marine Fisheries Text Data. [https://services.incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en](https://services.incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en)
11. FishBase Glossary: Length at First Maturity. [https://www.fishbase.se/glossary/Glossary.php?q=length+at+first+maturity](https://www.fishbase.se/glossary/Glossary.php?q=length+at+first+maturity)
12. scikit-learn Documentation. [https://scikit-learn.org/](https://scikit-learn.org/)
13. Streamlit Documentation. [https://streamlit.io/](https://streamlit.io/)

# Appendix

## Appendix A: Key Algorithms

- Random Forest Classifier for fish availability prediction.
- Random Forest Regressor for catch quantity prediction.
- Boosting Classifier and Regressor with Gradient Boosting fallback on the current system.
- Hybrid PCA + RF + ET + Boosting ensemble.
- Extra Trees juvenile-risk classifier.
- Exact maturity-based juvenile-risk formula.
- Safe-zone recommendation logic.

## Appendix B: Project Execution Commands

```bash
.venv/bin/python src/check_external_datasets.py
.venv/bin/python src/run_full_pipeline.py
.venv/bin/python src/validate_project.py
.venv/bin/python src/demo_algorithms.py
.venv/bin/python src/generate_report_figures.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/streamlit run src/app.py --server.headless true --server.port 8501
```

## Appendix C: Validation Procedure

Validation is performed at three levels:

- Dataset validation: verify row counts, required columns, and exact-ready field rows.
- Model validation: measure availability accuracy, juvenile-risk accuracy, weighted F1, RMSE, MAE, and R2.
- Logic validation: verify exact maturity-rule behavior, fallback behavior, and safe-zone response.

Current validated status:

- Field observation rows: `4`
- Field exact-ready rows: `3`
- Juvenile exact-label rows in training: `3`
- Automated tests passed: `9`

### Suggested Validation Figures

![Availability Accuracy](/Users/lakshmis/Final-year-project/reports/figures/availability_accuracy.png)

Figure C.1: Availability prediction accuracy of the implemented models.

![Quantity RMSE](/Users/lakshmis/Final-year-project/reports/figures/quantity_rmse.png)

Figure C.2: Catch quantity prediction RMSE comparison.

![Juvenile Confusion Matrix](/Users/lakshmis/Final-year-project/reports/figures/juvenile_confusion_matrix.png)

Figure C.3: Juvenile-risk confusion matrix after class balancing and model refinement.
