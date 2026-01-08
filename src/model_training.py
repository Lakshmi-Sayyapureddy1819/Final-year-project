import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

print("📌 Loading final real training dataset...")
df = pd.read_csv("data/final_training_data_fixed.csv")

# Create Availability label from Catch
df["Availability"] = df["Historical_Catch"].apply(lambda x: 1 if x > 300 else 0)

# Feature selection
X = df[["SST", "Salinity", "Dissolved_Oxygen", "Historical_Catch"]]
y_class = df["Availability"]
y_reg = df["Historical_Catch"]

# ONE unified split
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ===== RANDOM FOREST MODELS =====
print("\n🌲 Training Random Forest Models...")
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train_class)
pred_class_rf = clf.predict(X_test)
print("Random Forest Availability Accuracy:", accuracy_score(y_test_class, pred_class_rf))

reg = RandomForestRegressor(n_estimators=50)
reg.fit(X_train, y_train_reg)
pred_reg_rf = reg.predict(X_test)
print("Random Forest Quantity RMSE:", mean_squared_error(y_test_reg, pred_reg_rf))

# ===== XGBOOST MODELS =====
print("\n🚀 Training XGBoost Models...")
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_clf.fit(X_train, y_train_class)
pred_class_xgb = xgb_clf.predict(X_test)
print("XGBoost Availability Accuracy:", accuracy_score(y_test_class, pred_class_xgb))

xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train, y_train_reg)
pred_reg_xgb = xgb_reg.predict(X_test)
print("XGBoost Quantity RMSE:", mean_squared_error(y_test_reg, pred_reg_xgb))

# Save all models
joblib.dump(clf, "models/availability_model.pkl")
joblib.dump(reg, "models/quantity_model.pkl")
joblib.dump(xgb_clf, "models/xgb_availability_model.pkl")
joblib.dump(xgb_reg, "models/xgb_quantity_model.pkl")

# ===== HYBRID PCA + RF + SG BOOST (GradientBoosting) =====
print("\n4️⃣ Hybrid ML — PCA + RF + SG Boost...")

# choose PCA components (at most number of features)
n_components = min(3, X_train.shape[1])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Classification ensemble (Random Forest + Gradient Boosting)
rf_clf_pca = RandomForestClassifier(n_estimators=50, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
voting_clf = VotingClassifier([('rf', rf_clf_pca), ('gb', gb_clf)], voting='soft')
voting_clf.fit(X_train_pca, y_train_class)
pred_class_hybrid = voting_clf.predict(X_test_pca)
print("Hybrid Availability Accuracy:", accuracy_score(y_test_class, pred_class_hybrid))

# Regression ensemble (Random Forest + Gradient Boosting)
rf_reg_pca = RandomForestRegressor(n_estimators=50, random_state=42)
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
voting_reg = VotingRegressor([('rf', rf_reg_pca), ('gb', gb_reg)])
voting_reg.fit(X_train_pca, y_train_reg)
pred_reg_hybrid = voting_reg.predict(X_test_pca)
print("Hybrid Quantity RMSE:", mean_squared_error(y_test_reg, pred_reg_hybrid))

# Save PCA and hybrid models
joblib.dump(pca, "models/pca_transform.pkl")
joblib.dump(voting_clf, "models/hybrid_availability_model.pkl")
joblib.dump(voting_reg, "models/hybrid_quantity_model.pkl")
print("\n✅ All models saved successfully!")
print("Models: availability_model.pkl, quantity_model.pkl, xgb_availability_model.pkl, xgb_quantity_model.pkl")
