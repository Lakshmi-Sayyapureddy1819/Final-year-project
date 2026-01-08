# src/map_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Load models (adjust filenames if different)
clf = joblib.load("models/availability_model.pkl")
reg = joblib.load("models/quantity_model.pkl")
j_model = joblib.load("models/juvenile_model.pkl")

st.set_page_config(page_title="Fish Map Heatmap", layout="wide")
st.title("Live Map Heatmap Overlay — Fish Prediction & Juvenile Risk")

# Controls
with st.sidebar:
    st.header("Heatmap Settings")
    center_lat = st.number_input("Center Latitude", value=15.0, format="%.6f")
    center_lon = st.number_input("Center Longitude", value=83.0, format="%.6f")
    radius_km = st.slider("Radius (km)", min_value=5, max_value=100, value=25)
    grid_res = st.slider("Grid resolution (points per side)", min_value=10, max_value=80, value=30)
    heat_type = st.selectbox("Heatmap Value", ["juvenile_risk_probability", "availability_prob", "predicted_quantity"])
    run_btn = st.button("Generate Heatmap")

def make_latlon_grid(center_lat, center_lon, radius_km, n):
    # approximate degrees per km (valid for small regions) ~ 1 deg lat ~ 111 km
    deg = radius_km / 111.0
    lat_min = center_lat - deg
    lat_max = center_lat + deg
    lon_min = center_lon - deg
    lon_max = center_lon + deg
    lats = np.linspace(lat_min, lat_max, n)
    lons = np.linspace(lon_min, lon_max, n)
    points = []
    for la in lats:
        for lo in lons:
            points.append((la, lo))
    return points

def compute_scores(points):
    # Build feature arrays for the models.
    # Here: we need SST, Salinity, DO, History for availability/quantity models.
    # Without real environmental values, use reasonable defaults or synthetic variation.
    scores = []
    for (lat, lon) in points:
        # Simple synthetic environmental variation for demo:
        # (Replace this with real environmental raster or API values)
        sst = 27.5 + (lat - center_lat) * 0.1 + (lon - center_lon) * 0.05
        sal = 34.0 + (lat - center_lat) * 0.02
        do = 6.0 - abs(lat - center_lat) * 0.05
        history = 400 + (np.sin(lat*3.14/180)*50)  # synthetic past catch proxy

        main_feat = np.array([[sst, sal, do, history]])
        juv_feat = np.array([[sst, sal, do]])

        # availability probability (if classifier supports predict_proba)
        try:
            avail_prob = float(clf.predict_proba(main_feat)[0,1])
        except Exception:
            avail_prob = float(clf.predict(main_feat)[0])

        # juvenile risk probability (we can convert class to numeric)
        try:
            # if juvenile model has predict_proba -> use probability of "High"
            probs = j_model.predict_proba(juv_feat)
            # find index of 'High' if label encoder used; otherwise assume order
            # fallback: average probability of non-Low classes
            if probs.shape[1] == 3:
                juv_prob = float(probs[0, 0])  # may need mapping depending on training
            else:
                juv_prob = float(probs[0].max())
        except Exception:
            # If juvenile model outputs label, map it
            lab = j_model.predict(juv_feat)[0]
            juv_prob = {"Low": 0.1, "Medium": 0.5, "High": 0.9}.get(lab, 0.5)

        # predicted quantity
        try:
            qty = float(reg.predict(main_feat)[0])
        except Exception:
            qty = 0.0

        scores.append({
            "lat": lat, "lon": lon,
            "avail_prob": avail_prob,
            "juv_prob": juv_prob,
            "qty": qty
        })
    return pd.DataFrame(scores)

if run_btn:
    with st.spinner("Computing grid scores (this may take a few seconds)..."):
        points = make_latlon_grid(center_lat, center_lon, radius_km, grid_res)
        df = compute_scores(points)

    st.success("Computed scores for %d points" % len(df))

    # choose heat values
    if heat_type == "juvenile_risk_probability":
        heat_vals = df[["lat","lon","juv_prob"]].values.tolist()
    elif heat_type == "availability_prob":
        heat_vals = df[["lat","lon","avail_prob"]].values.tolist()
    else:
        # normalized quantity
        q = df["qty"].values
        qn = (q - q.min()) / (q.max() - q.min() + 1e-6)
        heat_vals = np.column_stack([df["lat"].values, df["lon"].values, qn]).tolist()

    # Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")
    HeatMap(heat_vals, radius=15, blur=10, max_zoom=13).add_to(m)

    # add some recommended safe points (low juvenile risk and high availability)
    candidates = df[(df["juv_prob"] < 0.4) & (df["avail_prob"] > 0.5)].sort_values(by=["avail_prob","qty"], ascending=False).head(8)
    for _, r in candidates.iterrows():
        folium.CircleMarker(location=[r.lat, r.lon],
                            radius=5,
                            color="green",
                            fill=True,
                            fill_opacity=0.8,
                            tooltip=f"Avail: {r.avail_prob:.2f}, Juv: {r.juv_prob:.2f}, Qty: {r.qty:.0f}").add_to(m)

    st_folium(m, width=900, height=600)
    st.dataframe(df.head(20))
