import streamlit as st
import joblib
import numpy as np
import pandas as pd
import folium
import os
from streamlit_folium import st_folium

# ======================= LOAD ML MODELS =======================
# Construct paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "..", "models")

clf = joblib.load(os.path.join(models_dir, "availability_model.pkl"))
reg = joblib.load(os.path.join(models_dir, "quantity_model.pkl"))
j_model = joblib.load(os.path.join(models_dir, "juvenile_model.pkl"))
# Optional hybrid models (PCA + RF + GradientBoosting ensembles)
try:
    pca = joblib.load(os.path.join(models_dir, "pca_transform.pkl"))
    hyb_clf = joblib.load(os.path.join(models_dir, "hybrid_availability_model.pkl"))
    hyb_reg = joblib.load(os.path.join(models_dir, "hybrid_quantity_model.pkl"))
except Exception:
    pca = None
    hyb_clf = None
    hyb_reg = None

# ======================= PAGE CONFIG =======================
st.set_page_config(page_title="AI-Driven Fish Catch Prediction System", layout="wide")

# ======================= CSS THEME =======================
st.markdown("""
<style>
body {background-color: #e6f4ff;}
.card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 6px solid #0077b6;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ======================= PROJECT HEADER =======================
st.markdown("<h1 style='text-align:center; color:#005f99;'>🌊 AI-Driven Fish Catch Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#0077b6;'>With Juvenile Risk Assessment for Sustainable Fisheries</h3>",
            unsafe_allow_html=True)
st.write("---")

# ======================= PREDICTION METHODS NAVIGATION =======================
menu = st.radio(
    "Choose Prediction Method",
    ["Manual Input", "Select Region", "Map Based GPS Input"],
    horizontal=True
)

# Choose ML pipeline
model_choice = st.selectbox("Choose ML Pipeline", ["Default (RF/XGB)", "Hybrid (PCA + RF + GB)"], index=0)

st.write("---")

# ======================= OUTPUT FUNCTION =======================
def display_output(location, availability, quantity, juvenile_risk):
    st.markdown("### 🎯 Prediction Summary")

    st.markdown(f"<div class='card'><h3>📍 Location: {location}</h3></div>", unsafe_allow_html=True)

    if availability == 0:
        st.markdown("<div class='card'><h3>🔴 Fish Availability: NO</h3></div>", unsafe_allow_html=True)
        st.error("📌 Fishing Not Recommended")
        st.info("➡ Try shifting to nearby zones (5–10 km).")
    else:
        st.markdown("<div class='card'><h3>🟢 Fish Availability: YES</h3></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><h3>🎣 Predicted Catch Quantity: {quantity:.2f} kg</h3></div>", unsafe_allow_html=True)

        if juvenile_risk == 'High':
            st.markdown("<div class='card'><h3>🔴 Juvenile Risk: HIGH ⚠</h3></div>", unsafe_allow_html=True)
            st.error("❌ Fishing Not Recommended — High Juvenile Density")
            st.info("➡ Suggested shift: Move 8–15 km away.")
        elif juvenile_risk == 'Medium':
            st.markdown("<div class='card'><h3>🟡 Juvenile Risk: MEDIUM</h3></div>", unsafe_allow_html=True)
            st.warning("👉 Fishing Allowed With Caution")
            st.info("➡ Recommended mesh size ≥ 45 mm.")
        else:
            st.markdown("<div class='card'><h3>🟢 Juvenile Risk: LOW</h3></div>", unsafe_allow_html=True)
            st.success("✔ Safe Fishing Zone")
            st.info("➡ Continue sustainable fishing practices")

# ======================= METHOD A: MANUAL INPUT =======================
if menu == "Manual Input":
    st.header("📝 Manual Parameter Entry")

    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Enter Location", "Vizag")
        SST = st.number_input("Sea Surface Temperature (°C)", 20, 35, 28)
        Salinity = st.number_input("Salinity (PSU)", 20, 40, 33)
    with col2:
        DO = st.number_input("Dissolved Oxygen (mg/l)", 1.0, 10.0, 6.4)
        History = st.number_input("Previous Avg Catch (kg)", 10, 1000, 200)

    if st.button("🔍 Predict (Manual)"):
        features = np.array([[SST, Salinity, DO, History]])
        juvenile_features = np.array([[SST, Salinity, History]])

        juvenile_risk = j_model.predict(juvenile_features)[0]

        if model_choice == "Hybrid (PCA + RF + GB)" and pca is not None and hyb_clf is not None and hyb_reg is not None:
            features_pca = pca.transform(features)
            availability = hyb_clf.predict(features_pca)[0]
            quantity = hyb_reg.predict(features_pca)[0]
        else:
            availability = clf.predict(features)[0]
            quantity = reg.predict(features)[0]


        # ---------------- HYBRID DECISION RULES ----------------
        good_conditions = 0
        if 22 <= SST <= 30: good_conditions += 1
        if 30 <= Salinity <= 36: good_conditions += 1
        if 5 <= DO <= 8: good_conditions += 1
        if History >= 150: good_conditions += 1

        # Rule 1: If environment conditions are mostly good → force YES
        if good_conditions >= 3 and juvenile_risk != "High":
            availability = 1
            quantity = max(quantity, 200)


            
        # Rule 2: Override for major fishing hubs
        prime_locations = ["Vizag", "Kakinada", "Chennai", "Goa", "Kochi", "Nellore", "Mangalore"]
        if location in prime_locations and availability == 0:
            availability = 1
            quantity = max(quantity, 220)

         # ---------------- DISPLAY RESULT ----------------
        st.subheader("🎣 Prediction Results")
        st.write(f"📍 **Location:** {location}")

        if availability == 0:
            st.error("🔴 **Fish Availability: NO**")
            st.info("➡ Try shifting to nearby zones (5–10 km).")
        else:
            st.success("🟢 **Fish Availability: YES**")
            st.write(f"🎣 **Predicted Catch Quantity:** {quantity:.2f} kg")

            if juvenile_risk == "High":
                st.error("⚠ **Juvenile Risk Level: HIGH — Fishing Not Recommended!**")
            elif juvenile_risk == "Medium":
                st.warning("🟡 Juvenile Risk Level: MEDIUM — Use caution")
            else:
                st.success("🟢 Juvenile Risk Level: LOW")
                st.info("✔ Fishing Recommended")


# ======================= METHOD B: REGION INPUT =======================
elif menu == "Select Region":
    st.header("📍 Region-Based Prediction")

    regions = {
        "Vizag": [29, 33, 6.2, 300],
        "Kakinada": [28, 34, 6.5, 260],
        "Machilipatnam": [27, 32, 6.8, 210],
        "Goa": [30, 35, 5.7, 280],
        "Kochi": [29, 36, 6.0, 330],
    }

    region = st.selectbox("Select Coastal Zone", regions.keys())
    SST, Salinity, DO, History = regions[region]

    if st.button("🔍 Predict (Region Based)"):
        features = np.array([[SST, Salinity, DO, History]])
        juvenile_features = np.array([[SST, Salinity, History]])

        juvenile_risk = j_model.predict(juvenile_features)[0]
        if model_choice == "Hybrid (PCA + RF + GB)" and pca is not None and hyb_clf is not None and hyb_reg is not None:
            features_pca = pca.transform(features)
            availability = hyb_clf.predict(features_pca)[0]
            quantity = hyb_reg.predict(features_pca)[0]
        else:
            availability = clf.predict(features)[0]
            quantity = reg.predict(features)[0]

        display_output(region, availability, quantity, juvenile_risk)

# ======================= METHOD C: MAP GPS INPUT =======================
elif menu == "Map Based GPS Input":
    st.header("🗺 Select Location from Map")

    map_center = [16.9891, 82.2475]
    m = folium.Map(location=map_center, zoom_start=6)

    map_output = st_folium(m, width=900, height=500)

    if map_output and map_output.get("last_clicked"):
        lat = map_output["last_clicked"]["lat"]
        lon = map_output["last_clicked"]["lng"]

        st.success(f"Selected Location → Lat: {lat:.3f}, Lon: {lon:.3f}")

        if st.button("🔍 Predict from Map"):
            SST, Salinity, DO, History = 28, 33, 6.2, 250
            features = np.array([[SST, Salinity, DO, History]])
            juvenile_features = np.array([[SST, Salinity, History]])

            juvenile_risk = j_model.predict(juvenile_features)[0]
            if model_choice == "Hybrid (PCA + RF + GB)" and pca is not None and hyb_clf is not None and hyb_reg is not None:
                features_pca = pca.transform(features)
                availability = hyb_clf.predict(features_pca)[0]
                quantity = hyb_reg.predict(features_pca)[0]
            else:
                availability = clf.predict(features)[0]
                quantity = reg.predict(features)[0]

            display_output(f"Lat:{lat}, Lon:{lon}", availability, quantity, juvenile_risk)

# ======================== END ========================
