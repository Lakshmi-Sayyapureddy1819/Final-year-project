# No-CNN Methodology for Final Report

## Methodology

The proposed Fish Catch Prediction System follows a multi-layer machine learning approach for sustainable fisheries support without using CNN, R-CNN, DeepLabCut, or behavioral analysis. The system is divided into four stages: data acquisition, preprocessing and feature engineering, predictive modeling, and decision support.

### 1. Data Acquisition

The system uses tabular marine and fisheries datasets collected from reliable public sources. Oceanographic variables such as Sea Surface Temperature, salinity, and related marine conditions are used to describe the environmental suitability of fishing zones. Historical catch values and location information are used to represent past fishing outcomes. Species maturity information is used to estimate juvenile risk in a biologically meaningful way.

### 2. Data Preprocessing and Feature Engineering

The collected datasets are cleaned to remove missing values, duplicates, and inconsistent records. Temporal and spatial records are aligned using month, latitude, and longitude. Additional engineered features are derived from the raw inputs, including seasonal encodings, thermal stress, oxygen stress, and salinity anomaly. These steps improve model stability and help capture variations in marine conditions across fishing regions.

### 3. Fish Availability Prediction

Fish availability is treated as a classification problem. Machine learning classifiers such as Random Forest and XGBoost are trained on environmental and historical catch features to predict whether fish are likely to be available in a selected zone. A hybrid pipeline using PCA with ensemble learning is also used to compare performance and improve generalization.

### 4. Catch Quantity Prediction

For zones predicted as favorable for fishing, the expected catch quantity is estimated using regression models such as Random Forest Regressor and XGBoost Regressor. This layer supports economic planning by providing an approximate catch volume for the selected location.

### 5. Juvenile-Risk Assessment

Juvenile risk is assessed using maturity-based analysis instead of image or video models. The risk score is calculated by comparing the observed fish length with the maturity length of the species:

`JR = 1 - (Observed_Length / Maturity_Length)`

If the observed fish length is far below the maturity length, the zone is considered high risk because juvenile fish are likely to dominate the catch. Based on the final score, juvenile risk is categorized as Low, Medium, or High.

### 6. Decision Support and Safe-Zone Recommendation

The final decision engine combines fish availability, juvenile risk, and catch quantity. If juvenile risk is high, fishing is not recommended even when fish availability is predicted. In such cases, the system suggests safer nearby zones within approximately 8-15 km. If risk is low or medium, the system displays fish availability and expected catch quantity with suitable fishing guidance.

## Final Method Summary

The final implemented methodology is:

- Data sources: oceanographic data, catch history, maturity-length data
- Availability model: Random Forest / XGBoost classifier
- Quantity model: Random Forest / XGBoost regressor
- Juvenile-risk model: maturity-based scoring
- Recommendation layer: safe-zone suggestion based on lower-risk nearby coordinates

## What To Remove From Your Old Report

Delete these terms from the methodology and architecture sections:

- CNN
- R-CNN
- DeepLabCut
- behavioral analysis
- video-based juvenile detection

## What To Replace Them With

Replace them with:

- machine learning classification
- machine learning regression
- maturity-based juvenile-risk assessment
- sustainable safe-zone recommendation
