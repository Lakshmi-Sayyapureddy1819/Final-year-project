import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from prediction_engine import PredictionResult, predict_fishing_zone


PIPELINE_OPTIONS = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Hybrid (PCA + RF + Boosting)": "hybrid",
}

REGIONS = {
    "Vizag": {"lat": 17.6868, "lon": 83.2185, "sst": 29.0, "salinity": 33.0, "do": 6.2, "history": 300.0},
    "Kakinada": {"lat": 16.9891, "lon": 82.2475, "sst": 28.0, "salinity": 34.0, "do": 6.5, "history": 260.0},
    "Machilipatnam": {"lat": 16.1875, "lon": 81.1381, "sst": 27.0, "salinity": 32.0, "do": 6.8, "history": 210.0},
    "Goa": {"lat": 15.2993, "lon": 74.1240, "sst": 30.0, "salinity": 35.0, "do": 5.7, "history": 280.0},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "sst": 29.0, "salinity": 36.0, "do": 6.0, "history": 330.0},
}


st.set_page_config(page_title="AI-Driven Fish Catch Prediction System", layout="wide")

st.markdown(
    """
<style>
body {background-color: #e6f4ff;}
.card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 6px solid #0077b6;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center; color:#005f99;'>AI-Driven Fish Catch Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align:center; color:#0077b6;'>Fish availability, juvenile-risk screening, and safe-zone recommendation</h3>",
    unsafe_allow_html=True,
)
st.write("---")

with st.sidebar:
    st.header("Prediction Setup")
    selected_pipeline = st.selectbox("ML pipeline", list(PIPELINE_OPTIONS.keys()), index=0)
    st.caption("The app now uses one shared decision engine across manual, region, and map workflows.")

menu = st.radio("Choose prediction method", ["Manual Input", "Select Region", "Map Based GPS Input"], horizontal=True)
st.write("---")


def maturity_inputs(prefix: str) -> tuple[float | None, float | None]:
    enabled = st.checkbox("Use maturity-based juvenile risk", key=f"{prefix}_maturity")
    if not enabled:
        return None, None

    with st.expander("Maturity-based juvenile-risk inputs", expanded=True):
        observed_length = st.number_input(
            "Observed fish length (cm)",
            min_value=0.0,
            value=12.0,
            step=0.5,
            key=f"{prefix}_observed_length",
        )
        maturity_length = st.number_input(
            "Species maturity length (cm)",
            min_value=0.0,
            value=18.0,
            step=0.5,
            key=f"{prefix}_maturity_length",
        )

    return observed_length or None, maturity_length or None


def display_output(result: PredictionResult) -> None:
    st.markdown("### Prediction Summary")
    st.markdown(f"<div class='card'><h3>Location: {result.location}</h3></div>", unsafe_allow_html=True)

    column_1, column_2, column_3 = st.columns(3)
    column_1.metric("Fish Availability", "YES" if result.availability else "NO")
    column_2.metric("Catch Quantity", f"{result.quantity:.2f} kg" if result.availability else "Blocked")
    column_3.metric("Juvenile Risk", result.juvenile_risk)

    st.info(result.advisory)
    st.caption(
        f"Pipeline: {result.model_pipeline} | Availability score: {result.availability_score:.2f} | "
        f"Juvenile score: {result.juvenile_score:.2f}"
    )

    detail_columns = st.columns(2)
    detail_columns[0].metric("Base Juvenile Layer", result.base_juvenile_risk)
    detail_columns[1].metric(
        "Maturity Score",
        f"{result.maturity_score:.2f}" if result.maturity_score is not None else "Not used",
    )

    if result.safe_zone_suggestions:
        st.markdown("#### Suggested Safer Zones")
        safe_zone_frame = pd.DataFrame(result.safe_zone_suggestions)
        st.dataframe(
            safe_zone_frame.rename(
                columns={
                    "zone": "Zone",
                    "distance_km": "Distance (km)",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "expected_juvenile_risk": "Expected Risk",
                    "expected_quantity_kg": "Expected Quantity (kg)",
                }
            ),
            use_container_width=True,
        )

        if result.latitude is not None and result.longitude is not None:
            suggestion_map = folium.Map(location=[result.latitude, result.longitude], zoom_start=8)
            folium.Marker(
                [result.latitude, result.longitude],
                tooltip="Selected Zone",
                icon=folium.Icon(color="red"),
            ).add_to(suggestion_map)

            for zone in result.safe_zone_suggestions:
                folium.Marker(
                    [zone["latitude"], zone["longitude"]],
                    tooltip=f'{zone["zone"]}: {zone["expected_juvenile_risk"]}',
                    icon=folium.Icon(color="green"),
                ).add_to(suggestion_map)

            st_folium(suggestion_map, width=900, height=380, key=f"safe_zone_map_{result.location}")


def run_prediction(
    *,
    location: str,
    sst: float,
    salinity: float,
    dissolved_oxygen: float,
    historical_catch: float,
    latitude: float | None,
    longitude: float | None,
    observed_length: float | None,
    maturity_length: float | None,
) -> None:
    result = predict_fishing_zone(
        location=location,
        sst=sst,
        salinity=salinity,
        dissolved_oxygen=dissolved_oxygen,
        historical_catch=historical_catch,
        latitude=latitude,
        longitude=longitude,
        model_choice=PIPELINE_OPTIONS[selected_pipeline],
        observed_length_cm=observed_length,
        maturity_length_cm=maturity_length,
    )
    display_output(result)


if menu == "Manual Input":
    st.header("Manual Parameter Entry")

    left_column, right_column = st.columns(2)
    with left_column:
        location = st.text_input("Location name", "Vizag")
        latitude = st.number_input("Latitude (optional for safe-zone mapping)", value=17.6868, format="%.4f")
        sst = st.number_input("Sea Surface Temperature (C)", min_value=20.0, max_value=35.0, value=28.0)
    with right_column:
        longitude = st.number_input("Longitude (optional for safe-zone mapping)", value=83.2185, format="%.4f")
        salinity = st.number_input("Salinity (PSU)", min_value=20.0, max_value=40.0, value=33.0)
        dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/l)", min_value=1.0, max_value=10.0, value=6.4)

    historical_catch = st.number_input("Previous Average Catch (kg)", min_value=10.0, max_value=2000.0, value=200.0)
    observed_length, maturity_length = maturity_inputs("manual")

    if st.button("Predict (Manual)"):
        run_prediction(
            location=location,
            sst=sst,
            salinity=salinity,
            dissolved_oxygen=dissolved_oxygen,
            historical_catch=historical_catch,
            latitude=latitude,
            longitude=longitude,
            observed_length=observed_length,
            maturity_length=maturity_length,
        )

elif menu == "Select Region":
    st.header("Region-Based Prediction")
    region = st.selectbox("Select coastal zone", list(REGIONS.keys()))
    config = REGIONS[region]
    observed_length, maturity_length = maturity_inputs("region")

    st.write(
        f"Preloaded conditions: SST {config['sst']} C, Salinity {config['salinity']} PSU, "
        f"DO {config['do']} mg/l, Historical Catch {config['history']} kg"
    )

    if st.button("Predict (Region Based)"):
        run_prediction(
            location=region,
            sst=config["sst"],
            salinity=config["salinity"],
            dissolved_oxygen=config["do"],
            historical_catch=config["history"],
            latitude=config["lat"],
            longitude=config["lon"],
            observed_length=observed_length,
            maturity_length=maturity_length,
        )

else:
    st.header("Map Based GPS Input")
    st.caption("Click a point on the map, then adjust the environmental inputs before running prediction.")

    map_center = [16.9891, 82.2475]
    base_map = folium.Map(location=map_center, zoom_start=6)
    map_output = st_folium(base_map, width=900, height=500, key="selection_map")

    if map_output and map_output.get("last_clicked"):
        st.session_state["selected_point"] = map_output["last_clicked"]

    selected_point = st.session_state.get("selected_point")

    if selected_point:
        latitude = float(selected_point["lat"])
        longitude = float(selected_point["lng"])
        st.success(f"Selected location: lat {latitude:.3f}, lon {longitude:.3f}")

        left_column, right_column = st.columns(2)
        with left_column:
            sst = st.number_input("Sea Surface Temperature (C)", min_value=20.0, max_value=35.0, value=28.0, key="map_sst")
            salinity = st.number_input("Salinity (PSU)", min_value=20.0, max_value=40.0, value=33.0, key="map_salinity")
        with right_column:
            dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/l)", min_value=1.0, max_value=10.0, value=6.2, key="map_do")
            historical_catch = st.number_input(
                "Previous Average Catch (kg)",
                min_value=10.0,
                max_value=2000.0,
                value=250.0,
                key="map_history",
            )

        observed_length, maturity_length = maturity_inputs("map")
        if st.button("Predict From Map"):
            run_prediction(
                location=f"Lat {latitude:.3f}, Lon {longitude:.3f}",
                sst=sst,
                salinity=salinity,
                dissolved_oxygen=dissolved_oxygen,
                historical_catch=historical_catch,
                latitude=latitude,
                longitude=longitude,
                observed_length=observed_length,
                maturity_length=maturity_length,
            )
