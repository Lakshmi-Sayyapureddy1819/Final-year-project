import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from prediction_engine import predict_fishing_zone


PIPELINE_OPTIONS = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Hybrid (PCA + RF + Boosting)": "hybrid",
}


st.set_page_config(page_title="Fish Map Heatmap", layout="wide")
st.title("Fish Prediction Heatmap and Safe-Zone Explorer")

with st.sidebar:
    st.header("Heatmap Settings")
    center_lat = st.number_input("Center Latitude", value=15.0, format="%.6f")
    center_lon = st.number_input("Center Longitude", value=83.0, format="%.6f")
    radius_km = st.slider("Radius (km)", min_value=5, max_value=100, value=25)
    grid_res = st.slider("Grid resolution (points per side)", min_value=10, max_value=60, value=25)
    heat_type = st.selectbox("Heatmap Value", ["juvenile_score", "availability_score", "predicted_quantity"])
    pipeline = st.selectbox("ML pipeline", list(PIPELINE_OPTIONS.keys()), index=0)
    run_btn = st.button("Generate Heatmap")


def make_latlon_grid(center_latitude: float, center_longitude: float, radius: float, resolution: int) -> list[tuple[float, float]]:
    degrees = radius / 111.0
    latitudes = np.linspace(center_latitude - degrees, center_latitude + degrees, resolution)
    longitudes = np.linspace(center_longitude - degrees, center_longitude + degrees, resolution)
    return [(lat, lon) for lat in latitudes for lon in longitudes]


def synthetic_environment(lat: float, lon: float) -> tuple[float, float, float, float]:
    sst = 27.5 + (lat - center_lat) * 0.12 + (lon - center_lon) * 0.05
    salinity = 34.0 + (lat - center_lat) * 0.03
    dissolved_oxygen = 6.2 - abs(lat - center_lat) * 0.06
    historical_catch = 350 + (np.sin(np.radians(lat)) * 60) + (np.cos(np.radians(lon)) * 25)
    return float(sst), float(salinity), float(dissolved_oxygen), float(max(historical_catch, 50.0))


def compute_scores(points: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for latitude, longitude in points:
        sst, salinity, dissolved_oxygen, historical_catch = synthetic_environment(latitude, longitude)
        result = predict_fishing_zone(
            location=f"{latitude:.3f}, {longitude:.3f}",
            sst=sst,
            salinity=salinity,
            dissolved_oxygen=dissolved_oxygen,
            historical_catch=historical_catch,
            latitude=latitude,
            longitude=longitude,
            model_choice=PIPELINE_OPTIONS[pipeline],
        )

        rows.append(
            {
                "lat": latitude,
                "lon": longitude,
                "availability_score": result.availability_score,
                "juvenile_score": result.juvenile_score,
                "quantity": result.quantity,
                "juvenile_risk": result.juvenile_risk,
            }
        )

    return pd.DataFrame(rows)


if run_btn:
    with st.spinner("Computing grid scores..."):
        score_frame = compute_scores(make_latlon_grid(center_lat, center_lon, radius_km, grid_res))

    st.success(f"Computed scores for {len(score_frame)} grid points.")

    if heat_type == "juvenile_score":
        heat_values = score_frame[["lat", "lon", "juvenile_score"]].values.tolist()
    elif heat_type == "availability_score":
        heat_values = score_frame[["lat", "lon", "availability_score"]].values.tolist()
    else:
        quantity_values = score_frame["quantity"].values
        normalized_quantity = (quantity_values - quantity_values.min()) / (quantity_values.max() - quantity_values.min() + 1e-6)
        heat_values = np.column_stack([score_frame["lat"].values, score_frame["lon"].values, normalized_quantity]).tolist()

    heatmap = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")
    HeatMap(heat_values, radius=15, blur=10, max_zoom=13).add_to(heatmap)

    candidates = score_frame[
        (score_frame["juvenile_score"] < 0.4) & (score_frame["availability_score"] > 0.55)
    ].sort_values(by=["availability_score", "quantity"], ascending=False).head(8)

    for _, row in candidates.iterrows():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=5,
            color="green",
            fill=True,
            fill_opacity=0.8,
            tooltip=(
                f"Availability {row.availability_score:.2f} | "
                f"Juvenile {row.juvenile_score:.2f} | "
                f"Qty {row.quantity:.0f}"
            ),
        ).add_to(heatmap)

    st_folium(heatmap, width=900, height=600)
    st.dataframe(score_frame.head(20), use_container_width=True)
