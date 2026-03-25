# AI Driven Fish Catch Prediction System with Juvenile Risk Assessment for Sustainable Fisheries

This file is a corrected replacement for the report content from the Table of Contents onward. It matches the project currently implemented in [/Users/lakshmis/Final-year-project](/Users/lakshmis/Final-year-project).

Important correction:

- The old report still contains unrelated `Wildlife Conservation`, `CNN`, and `EfficientNetB7` content.
- The corrected project is a `no-CNN`, `tabular machine learning`, `real-data`, `juvenile-risk aware` fisheries prediction system.

## Table of Contents

### Chapter 1 Introduction

- 1.1 Introduction
- 1.2 Artificial Intelligence and Machine Learning in Fisheries
- 1.3 Sustainable Fisheries and Juvenile Risk Assessment
- 1.4 Objectives of the Project
- 1.5 Problem Statement
- 1.6 Scope of the Work

### Chapter 2 Literature Review

- 2.1 Review of Literature on Fish Catch Prediction
- 2.2 Oceanographic and Geographic Considerations
- 2.3 Factors Affecting Fish Availability and Juvenile Presence
- 2.4 Integration of Technology for Fisheries Decision Support
- 2.5 Recent Advancements in Fisheries Informatics
- 2.6 Research Gaps
- 2.7 Motivation
- 2.8 Problem Statement and Research Objectives
- 2.9 Summary of Literature Review

### Chapter 3 Methodology

- 3.1 Research Design and Approach
- 3.2 Data Collection and Preprocessing
- 3.3 Fish Availability Prediction using Machine Learning
- 3.4 Catch Quantity Estimation
- 3.5 Juvenile Risk Assessment Layer
- 3.6 Safe-Zone Recommendation and System Deployment

### Chapter 4 Results and Discussion

- 4.1 Case Study: AI Driven Fish Catch Prediction for Sustainable Fisheries
- 4.1.1 System Configuration and Technical Details
- 4.1.2 Performance Evaluation and Validation
- 4.2 Results and Discussion
- 4.2.1 Fish Availability Prediction Results
- 4.2.2 Catch Quantity Estimation Results
- 4.2.3 Juvenile Risk Assessment Results
- 4.2.4 Safe-Zone Recommendation Results
- 4.2.5 Application Interface and Deployment
- 4.3 Summary of Results

### Chapter 5 Conclusions and Future Scope of Work

- 5.1 Conclusion
- 5.2 Contributions of the Proposed System
- 5.3 Practical Implications and Applications
- 5.4 Recommendations for Future Work

### Chapter 6 References

- References

### Appendix

- Appendix A: Key Algorithms
- Appendix B: Project Execution Commands
- Appendix C: Validation Procedure

## Chapter 1 Introduction

### 1.1 Introduction

Fisheries play an important role in food security, coastal livelihoods, and regional economies. However, overfishing, environmental variability, and the capture of juvenile fish reduce long-term sustainability. Traditional fishing decisions often rely on experience, seasonal memory, and limited marine advisories. While these approaches are valuable, they do not fully exploit the predictive potential of modern data-driven systems.

This project proposes an AI Driven Fish Catch Prediction System with Juvenile Risk Assessment for Sustainable Fisheries. The system predicts whether fishing conditions are favorable, estimates the expected catch quantity, and evaluates whether juvenile fish are likely to dominate the catch. When ecological risk is high, the system recommends safer nearby fishing zones rather than simply maximizing catch.

The final implemented work focuses on tabular machine learning and biologically meaningful juvenile-risk assessment. It uses fisheries and oceanographic data rather than image or video pipelines. This makes the project more practical, more defensible, and more closely aligned with the available real-world datasets.

### 1.2 Artificial Intelligence and Machine Learning in Fisheries

Artificial Intelligence and Machine Learning are increasingly used in marine informatics for catch forecasting, fishing-ground recommendation, anomaly detection, and resource planning. In fisheries, important predictive variables include sea surface temperature, salinity, dissolved oxygen, season, geographic location, and historical landings. These features can be modeled using supervised learning algorithms.

In this project, fish availability is treated as a classification task and catch quantity is treated as a regression task. Ensemble machine learning methods such as Random Forest and boosting-based models are used because they perform well on nonlinear structured data and require fewer assumptions than traditional linear models. A hybrid model using PCA and ensemble learning is also included to compare predictive performance.

### 1.3 Sustainable Fisheries and Juvenile Risk Assessment

A key limitation of many prediction systems is that they focus only on catch maximization. In sustainable fisheries, this is not sufficient. Catching juvenile fish reduces future stock strength and affects ecological balance. Therefore, a useful advisory system should not only identify productive zones but also identify biologically unsafe zones.

The juvenile-risk layer introduced in this project uses maturity-based assessment. Instead of relying on image-based fish behavior or video-based pose estimation, the system compares the observed fish length to the maturity length of the species. This approach is practical, explainable, and grounded in fisheries biology.

### 1.4 Objectives of the Project

The main objectives of the project are:

- to predict fish availability in a selected marine zone
- to estimate the expected catch quantity in favorable zones
- to assess juvenile-risk using maturity-based logic
- to recommend safer nearby fishing zones when ecological risk is high
- to build a usable decision-support interface for demonstration and validation

### 1.5 Problem Statement

Fishermen and marine stakeholders need a practical system that can identify productive fishing opportunities without encouraging the capture of juvenile fish. Existing academic prototypes often depend on image datasets, video datasets, or highly specialized annotations that are difficult to obtain in real fishing scenarios. There is a need for a reliable alternative that uses real fisheries data, oceanographic variables, and biological maturity information to balance productivity and conservation.

### 1.6 Scope of the Work

The current scope includes:

- annual state-level fish landing data from CMFRI
- SST-linked environmental data
- optional PFZ and maturity-based enrichment
- machine learning models for availability and quantity prediction
- a maturity-based juvenile-risk layer
- a Streamlit application for manual, region-based, and map-based prediction

The system does not currently implement CNN, R-CNN, DeepLabCut, or video-based behavioral analysis due to the absence of suitable annotated video datasets.

## Chapter 2 Literature Review

### 2.1 Review of Literature on Fish Catch Prediction

Fish catch prediction has been studied using oceanographic indicators, remote sensing data, statistical models, and machine learning algorithms. Many studies show that variables such as SST, chlorophyll concentration, salinity, and previous catch records are useful predictors of fish distribution and abundance. In fisheries decision support, the most successful approaches often combine environmental suitability with historical catch behavior.

### 2.2 Oceanographic and Geographic Considerations

Marine productivity is strongly influenced by oceanographic conditions. SST affects species migration and habitat suitability. Salinity and dissolved oxygen affect survival and physiological stress. Latitude, longitude, and seasonal cycles also influence fish aggregation. Therefore, fisheries prediction should not rely on one feature alone; it should combine multiple environmental and spatial indicators.

### 2.3 Factors Affecting Fish Availability and Juvenile Presence

Fish availability depends on seasonal movement, habitat conditions, ocean productivity, and previous stock behavior. Juvenile presence depends on spawning cycles, recruitment patterns, fishing pressure, and species-specific maturity length. A sustainable system must therefore evaluate both catch opportunity and ecological sensitivity.

### 2.4 Integration of Technology for Fisheries Decision Support

Modern fisheries systems increasingly integrate public datasets, predictive models, geospatial interfaces, and advisory outputs. Government agencies provide PFZ advisories and marine condition updates, while machine learning systems can personalize and extend these advisories. The combination of public marine data and interactive prediction tools is therefore a strong direction for applied fisheries research.

### 2.5 Recent Advancements in Fisheries Informatics

Recent advancements include satellite-driven fish habitat models, ensemble learning for catch forecasting, and spatial advisory systems. There is also growing interest in sustainability-aware prediction, where the objective is not only yield estimation but also stock protection and responsible harvesting.

### 2.6 Research Gaps

The literature reveals several gaps:

- many studies focus on prediction accuracy but not juvenile protection
- some methods depend on high-resolution datasets that are not easily available to students or local stakeholders
- systems that combine catch prediction, ecological blocking, and safe-zone recommendation are relatively limited
- real-world project implementations often suffer from mismatch between proposed methodology and available data

### 2.7 Motivation

The project is motivated by the need to build a realistic, demo-ready, and sustainability-aware fisheries system using accessible public datasets. Instead of claiming unsupported deep-learning behavior models, the implemented system focuses on what can be validated with available data: tabular ML, maturity-based risk scoring, and operational decision support.

### 2.8 Problem Statement and Research Objectives

The research objective is to design a fisheries advisory system that integrates:

- fish availability prediction
- catch quantity estimation
- juvenile-risk assessment
- safe-zone recommendation

while using real-world data sources that are actually obtainable and verifiable.

### 2.9 Summary of Literature Review

The literature supports the use of environmental and historical features for fisheries prediction, but it also highlights the importance of sustainability. This project responds to that gap by adding a juvenile-risk layer and a safe-zone recommendation mechanism to a practical machine learning pipeline.

## Chapter 3 Methodology

### 3.1 Research Design and Approach

The proposed system follows a layered machine learning design:

1. collect and integrate fisheries and marine datasets
2. preprocess and engineer predictive features
3. classify whether a zone is likely to be fish-available
4. estimate the expected catch quantity
5. assess juvenile risk using maturity-based scoring
6. recommend safer nearby zones when risk is high

This design keeps the project modular and allows each layer to be validated independently.

### 3.2 Data Collection and Preprocessing

The implemented project uses the following sources:

- CMFRI state-wise marine fish landing estimates
- Indian SST data already available in the project
- FishBase maturity-length reference data for selected species
- PFZ-style observation rows stored in the project for exact juvenile-risk enrichment

The preprocessing steps include:

- cleaning missing and inconsistent values
- year-wise and state-wise alignment of CMFRI and SST data
- generation of engineered features such as thermal stress, oxygen stress, and salinity anomaly
- month-based cyclical encoding using sine and cosine transformation
- integration of observed length and maturity-length fields when available

### 3.3 Fish Availability Prediction using Machine Learning

Fish availability is formulated as a binary classification problem. The input feature set includes SST, salinity, dissolved oxygen, historical catch, location-based fields, and engineered seasonal and stress indicators.

The implemented classifiers are:

- Random Forest Classifier
- Boosting Classifier
- Hybrid PCA + Random Forest + Boosting pipeline

The system is designed for XGBoost support, but on the current development machine the boosting layer falls back to Gradient Boosting because the native XGBoost runtime requires `libomp.dylib`.

### 3.4 Catch Quantity Estimation

After availability prediction, the system estimates expected catch quantity through regression. The implemented regressors are:

- Random Forest Regressor
- Boosting Regressor
- Hybrid PCA + Random Forest + Boosting Regressor

This layer provides approximate catch estimates for planning and comparative evaluation of zones.

### 3.5 Juvenile Risk Assessment Layer

The juvenile-risk layer combines environmental fallback logic with exact biological assessment whenever species and length data are available.

The exact maturity-based formula is:

`JR = 1 - (Observed_Length / Maturity_Length)`

Risk categorization is performed as follows:

- `High` if the score is greater than or equal to `0.2`
- `Medium` if the score is greater than `0` and less than `0.2`
- `Low` if the observed length is at or above maturity length

When exact biological fields are unavailable, the project uses an environmental juvenile-risk model derived from SST stress, oxygen stress, and historical catch behavior. This fallback allows the system to remain operational while still prioritizing the exact rule whenever possible.

### 3.6 Safe-Zone Recommendation and System Deployment

The decision layer integrates availability, quantity, and juvenile risk. If juvenile risk is high, the system blocks fishing recommendations and suggests safer nearby zones within approximately `8-15 km`. These suggestions are generated through directional offsets from the selected location and adjusted expected-risk logic.

The system is deployed as a Streamlit application supporting:

- manual prediction
- region-based prediction
- map-based GPS input
- field data collection for PFZ-style observation enrichment

## Chapter 4 Results and Discussion

### 4.1 Case Study: AI Driven Fish Catch Prediction for Sustainable Fisheries

The project was implemented and validated using the current multi-source dataset generated inside the repository. The dataset includes CMFRI landings, SST-linked features, PFZ observation enrichment, and FishBase maturity support.

### 4.1.1 System Configuration and Technical Details

The project is implemented in Python using:

- pandas and numpy for preprocessing
- scikit-learn for Random Forest, Gradient Boosting, PCA, and evaluation
- joblib for model serialization
- Streamlit and Folium for application deployment and map visualization

The current validated dataset status is:

- total training rows: `110`
- field observation rows: `3`
- exact-ready field rows: `3`
- juvenile exact-label rows used in training: `3`

### 4.1.2 Performance Evaluation and Validation

The project includes automated validation, logic tests, and a reproducible pipeline. The latest baseline validation results are:

- Random Forest availability accuracy: `0.5909`
- Random Forest quantity RMSE: `82478.3759`
- Boosting availability accuracy: `0.5000`
- Boosting quantity RMSE: `91903.1661`
- Hybrid availability accuracy: `0.4545`
- Hybrid quantity RMSE: `76830.9764`
- Juvenile-risk accuracy: `0.9091`
- Juvenile weighted F1 score: `0.8874`

In addition, the project includes automated demo verification checks confirming:

- exact maturity high-risk case passes
- exact maturity low-risk case passes
- environmental fallback case passes

### 4.2 Results and Discussion

### 4.2.1 Fish Availability Prediction Results

Random Forest produced the strongest availability classification among the currently trained models. The accuracy is moderate because the dataset is still relatively small and coarse-grained, consisting mainly of annual state-level rows rather than monthly fishing-zone-level targets. Even so, the model provides usable advisory behavior in the deployed interface.

### 4.2.2 Catch Quantity Estimation Results

Catch quantity regression remains challenging because fisheries yields vary substantially across years and regions. The Hybrid model produced the lowest RMSE among the available pipelines, indicating that dimensionality reduction plus ensemble regression may help stabilize quantity estimation in small structured datasets.

### 4.2.3 Juvenile Risk Assessment Results

The most important improvement in the project is the juvenile-risk layer. The system no longer depends only on heuristic environmental risk. Instead, it now uses exact maturity-based labels when species and fish-length records are available. With the addition of exact-ready observation rows, the juvenile model now trains with both:

- environmental heuristic labels
- exact maturity-rule labels

The current distribution is:

- environmental heuristic rows: `107`
- exact maturity rule rows: `3`

This confirms that the project has moved beyond a purely heuristic juvenile-risk layer.

### 4.2.4 Safe-Zone Recommendation Results

When a selected zone is identified as high juvenile risk or weak availability, the application suggests nearby alternative zones and provides expected risk and quantity indicators. This adds practical value to the system because it avoids returning only negative results and instead offers a sustainability-aware recommendation.

### 4.2.5 Application Interface and Deployment

The Streamlit interface provides:

- manual prediction for user-defined marine conditions
- region-based prediction for preconfigured coastal zones
- map-based prediction through geographic point selection
- field-data entry for adding exact juvenile-risk observations

The deployed application preserves the latest prediction summary using session state so that results remain visible after submission.

### 4.3 Summary of Results

The results show that the system is operational, reproducible, and aligned with the actual implemented methodology. The strongest contribution is not only prediction, but the integration of ecological blocking and safe-zone advisory into the decision process.

## Chapter 5 Conclusions and Future Scope of Work

### 5.1 Conclusion

This project presents a practical AI Driven Fish Catch Prediction System with Juvenile Risk Assessment for Sustainable Fisheries. It combines real fisheries data, environmental indicators, machine learning models, and maturity-based risk scoring in a single decision-support workflow. The system predicts fish availability, estimates catch quantity, evaluates juvenile risk, and recommends safer nearby zones when necessary.

Unlike the earlier mismatched draft report, the final implemented work does not rely on CNN, EfficientNetB7, or wildlife image classification. Instead, it uses tabular machine learning and biologically meaningful juvenile-risk logic, which is more appropriate for the available datasets and the actual deployed codebase.

### 5.2 Contributions of the Proposed System

The major contributions are:

- a real-data fisheries prediction pipeline using CMFRI and SST data
- a maturity-based juvenile-risk layer integrated into decision making
- a safe-zone recommendation mechanism for sustainable fishing
- a validation workflow with reproducible tests and metrics
- a deployable Streamlit interface with data-entry support

### 5.3 Practical Implications and Applications

The proposed system can support:

- fishermen seeking safer and more productive zones
- student and academic demonstrations of sustainability-aware fisheries AI
- marine advisory prototypes for ecological decision support
- future integration of PFZ records and richer oceanographic variables

### 5.4 Recommendations for Future Work

Future work may include:

- acquiring monthly or district-level CMFRI landings
- integrating Copernicus salinity, chlorophyll, and physical ocean products
- expanding exact juvenile observations using PFZ and field measurements
- enabling native XGBoost after OpenMP runtime installation
- adding more coastal sectors, richer maps, and time-series forecasting

If suitable annotated video datasets become available later, a separate future extension may explore behavioral or vision-based juvenile detection. However, that is not part of the current implemented system.

## Chapter 6 References

Use these references in the final report:

1. CMFRI Fish Catch Estimates. [https://www.cmfri.org.in/fish-catch-estimates](https://www.cmfri.org.in/fish-catch-estimates)
2. CMFRI Methodology. [https://www.cmfri.org.in/methodology](https://www.cmfri.org.in/methodology)
3. NOAA Optimum Interpolation Sea Surface Temperature. [https://www.ncei.noaa.gov/products/optimum-interpolation-sst](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
4. Copernicus Marine Physics Reanalysis. [https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
5. Copernicus Ocean Colour Chlorophyll Product. [https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_MY_009_103/description](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_MY_009_103/description)
6. INCOIS PFZ Advisory. [https://services.incois.gov.in/MarineFisheries/PfzAdvisory.action](https://services.incois.gov.in/MarineFisheries/PfzAdvisory.action)
7. INCOIS Marine Fisheries Text Data. [https://services.incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en](https://services.incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en)
8. FishBase Glossary: Length at First Maturity. [https://www.fishbase.se/glossary/Glossary.php?q=length+at+first+maturity](https://www.fishbase.se/glossary/Glossary.php?q=length+at+first+maturity)
9. scikit-learn Documentation. [https://scikit-learn.org/](https://scikit-learn.org/)
10. Streamlit Documentation. [https://streamlit.io/](https://streamlit.io/)

## Appendix A: Key Algorithms

Implemented algorithms:

- Random Forest Classifier
- Random Forest Regressor
- Boosting Classifier
- Boosting Regressor
- PCA + Hybrid Voting Classifier
- PCA + Hybrid Voting Regressor
- Exact maturity-based juvenile-risk rule

## Appendix B: Project Execution Commands

```bash
.venv/bin/python src/check_external_datasets.py
.venv/bin/python src/run_full_pipeline.py
.venv/bin/python src/validate_project.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/streamlit run src/app.py --server.headless true --server.port 8501
```

## Appendix C: Validation Procedure

Validation of the project should be presented at three levels:

- dataset validation:
  verify row counts, source coverage, and exact-ready observation rows
- model validation:
  report classification accuracy, F1 score, and regression RMSE
- logic validation:
  verify that the exact maturity rule returns high risk for sub-maturity fish and low risk for mature fish

Latest validated status from the implemented project:

- field observation rows: `3`
- field exact-ready rows: `3`
- juvenile exact-label rows in training: `3`
- automated tests passed: `5`

## What To Correct in the Existing PDF

The following parts of the old report should be replaced or corrected:

- Certificate page:
  replace `WildLife Conservation using Deep Learning` with the actual fish-catch project title
- Table of Contents:
  replace all wildlife/CNN/EfficientNetB7 items with the corrected contents above
- Literature Review:
  remove wildlife monitoring sections
- Methodology:
  remove CNN, EfficientNetB7, R-CNN, DeepLabCut, and behavioral analysis
- Results:
  replace image-classification results with fisheries model metrics and validation
- Figures and Tables:
  replace wildlife image figures with system architecture, dataset flow, validation charts, and application screenshots
