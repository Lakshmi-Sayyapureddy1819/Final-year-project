import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from demo_algorithms import (
    get_balancing_comparison,
    get_code_sections,
    get_runtime_summary,
    run_algorithm_demos,
    write_algorithm_execution_report,
)
from field_data_store import (
    COASTAL_STATES,
    append_pfz_observation,
    data_quality_issues,
    observation_summary,
    recent_observations,
)
from prediction_engine import PredictionResult, known_species, lookup_maturity_length, predict_fishing_zone


PIPELINE_OPTIONS = {
    "Random Forest": "random_forest",
    "Boosting": "xgboost",
    "Hybrid (PCA + RF + ET + Boosting)": "hybrid",
}

REGIONS = {
    "Vizag": {"lat": 17.6868, "lon": 83.2185, "sst": 29.0, "salinity": 33.0, "do": 6.2, "history": 300.0},
    "Kakinada": {"lat": 16.9891, "lon": 82.2475, "sst": 28.0, "salinity": 34.0, "do": 6.5, "history": 260.0},
    "Machilipatnam": {"lat": 16.1875, "lon": 81.1381, "sst": 27.0, "salinity": 32.0, "do": 6.8, "history": 210.0},
    "Goa": {"lat": 15.2993, "lon": 74.1240, "sst": 30.0, "salinity": 35.0, "do": 5.7, "history": 280.0},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "sst": 29.0, "salinity": 36.0, "do": 6.0, "history": 330.0},
}

RESULT_STATE_KEY = "prediction_result"
RESULT_SOURCE_KEY = "prediction_result_source"
DEMO_RESULTS_KEY = "algorithm_demo_results"
DEMO_REPORT_KEY = "algorithm_demo_report_path"
INPUT_STATE_KEY = "prediction_input_payload"
COMPARISON_STATE_KEY = "prediction_pipeline_comparison"
METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "latest_metrics.json"


st.set_page_config(page_title="AI-Driven Fish Catch Prediction System", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
:root {
    --sea-ink: #0f2742;
    --sea-blue: #1f4e79;
    --sea-teal: #1f7a8c;
    --reef-gold: #f4b860;
    --surf: #eef6fb;
    --card: rgba(255, 255, 255, 0.9);
    --line: rgba(18, 52, 86, 0.12);
    --sand: #f3ede1;
}

@keyframes tideShift {
    0% { background-position: 0% 0%, 100% 0%, 0% 50%; }
    50% { background-position: 10% 12%, 88% 15%, 100% 50%; }
    100% { background-position: 0% 0%, 100% 0%, 0% 50%; }
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes glowPulse {
    0%, 100% { box-shadow: 0 18px 40px rgba(7, 40, 71, 0.18); }
    50% { box-shadow: 0 22px 50px rgba(31, 122, 140, 0.22); }
}

@keyframes boatDrift {
    0%, 100% { transform: translateX(0px) translateY(0px); }
    50% { transform: translateX(10px) translateY(-5px); }
}

@keyframes floatPulse {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(55, 112, 148, 0.14), transparent 30%),
        radial-gradient(circle at 85% 12%, rgba(244, 184, 96, 0.14), transparent 24%),
        linear-gradient(180deg, rgba(255,255,255,0.0), rgba(255,255,255,0.0)),
        linear-gradient(180deg, #f6efe3 0%, #eef4f8 24%, #edf6fb 60%, #f9fcff 100%);
    background-size: 165% 165%, 150% 150%, 28px 28px, 100% 100%;
    background-image:
        radial-gradient(circle at top left, rgba(55, 112, 148, 0.14), transparent 30%),
        radial-gradient(circle at 85% 12%, rgba(244, 184, 96, 0.14), transparent 24%),
        linear-gradient(135deg, rgba(31,122,140,0.035) 25%, transparent 25%, transparent 50%, rgba(31,122,140,0.035) 50%, rgba(31,122,140,0.035) 75%, transparent 75%, transparent),
        linear-gradient(180deg, #f6efe3 0%, #eef4f8 24%, #edf6fb 60%, #f9fcff 100%);
    animation: tideShift 22s ease-in-out infinite;
}

.block-container {
    max-width: 1320px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

.site-nav {
    position: relative;
    top: 0;
    z-index: 30;
    margin-bottom: 1.1rem;
}

.site-nav-shell {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.9rem 1.1rem;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(15, 39, 66, 0.12);
    backdrop-filter: blur(14px);
    box-shadow: 0 18px 40px rgba(15, 39, 66, 0.12);
}

h1, h2, h3, h4, h5, h6 {
    color: #102c49 !important;
}

.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
.stMarkdown h5,
.stMarkdown h6 {
    color: #102c49 !important;
}

label,
.stRadio label,
.stCheckbox label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stTextArea label,
.stDateInput label {
    color: #163754 !important;
    font-weight: 600 !important;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.nav-mark {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #0f2742 0%, #1f7a8c 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 1rem;
    box-shadow: 0 14px 28px rgba(15, 39, 66, 0.2);
}

.nav-title {
    color: #0b2540;
    font-weight: 700;
    font-size: 1.22rem;
    line-height: 1.15;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    flex-wrap: wrap;
}

.nav-links a {
    text-decoration: none;
    color: #1f4e79;
    font-weight: 600;
    padding: 0.55rem 0.88rem;
    border-radius: 999px;
    background: rgba(31, 78, 121, 0.06);
    border: 1px solid rgba(31, 78, 121, 0.08);
    transition: transform 0.18s ease, background 0.18s ease;
}

.nav-links a:hover {
    transform: translateY(-1px);
    background: rgba(31, 122, 140, 0.1);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 39, 66, 0.97) 0%, rgba(26, 65, 99, 0.95) 54%, rgba(31, 122, 140, 0.94) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

[data-testid="stSidebar"] * {
    color: #f4f8fb !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(244, 248, 251, 0.86) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] textarea {
    background: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid rgba(21, 59, 97, 0.16) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="input"] div,
[data-testid="stSidebar"] [data-baseweb="input"] span,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] svg {
    color: #153b61 !important;
    fill: #153b61 !important;
}

[data-testid="stSidebar"] .stAlert {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.14);
}

div[data-testid="stForm"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1rem 1.05rem 1.25rem;
    box-shadow: 0 14px 34px rgba(12, 43, 67, 0.1);
    backdrop-filter: blur(12px);
    animation: riseIn 0.55s ease;
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(244, 249, 252, 0.96) 100%);
    border: 1px solid rgba(31, 78, 121, 0.12);
    border-radius: 20px;
    padding: 0.7rem 0.4rem;
    box-shadow: 0 10px 26px rgba(15, 39, 66, 0.08);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #153b61 !important;
}

div[data-testid="stDataFrame"],
div[data-testid="stTabs"],
div[data-testid="stExpander"] {
    animation: riseIn 0.55s ease;
}

[data-testid="stDataFrame"] div,
[data-testid="stDataFrame"] span,
[data-testid="stDataFrame"] p,
[data-testid="stTable"] div,
[data-testid="stTable"] span {
    color: #153b61 !important;
}

[data-testid="stAlert"] {
    border-radius: 18px;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #153b61 !important;
}

[data-testid="stCodeBlock"] pre,
[data-testid="stCode"] pre {
    background: #0f2742 !important;
    color: #f7fbff !important;
    border-radius: 18px !important;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #4f6980 !important;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown label,
.stMarkdown span {
    color: #244760;
}

.stMarkdown strong,
.stMarkdown b {
    color: #102c49;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #244760 !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {
    color: #102c49 !important;
}

[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #163754 !important;
}

[data-baseweb="radio"] label,
[data-baseweb="checkbox"] label {
    color: #153b61 !important;
}

[data-baseweb="select"] input,
[data-baseweb="input"] input,
textarea {
    color: #153b61 !important;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
textarea {
    background: rgba(255,255,255,0.94) !important;
    border: 1px solid rgba(31, 78, 121, 0.16) !important;
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    border: none;
    border-radius: 999px;
    color: #ffffff;
    font-weight: 700;
    letter-spacing: 0.01em;
    background: linear-gradient(135deg, #0f2742 0%, #1f4e79 52%, #1f7a8c 100%);
    box-shadow: 0 12px 22px rgba(15, 39, 66, 0.22);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.02);
    box-shadow: 0 18px 32px rgba(15, 39, 66, 0.28);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    border-radius: 16px !important;
}

button[role="tab"] {
    border-radius: 999px !important;
    margin-right: 0.45rem;
    padding: 0.5rem 1rem !important;
    border: 1px solid rgba(31, 78, 121, 0.12) !important;
    background: rgba(255, 255, 255, 0.62) !important;
}

button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(15, 39, 66, 0.96), rgba(31, 122, 140, 0.96)) !important;
    color: white !important;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    border-radius: 30px;
    padding: 2.2rem 2.3rem;
    margin-bottom: 1.1rem;
    background: linear-gradient(135deg, #0d2742 0%, #153b61 48%, #1f7a8c 100%);
    color: #f7fbff;
    box-shadow: 0 24px 54px rgba(10, 39, 63, 0.26);
    animation: riseIn 0.7s ease, glowPulse 8s ease-in-out infinite;
}

.hero-shell::before,
.hero-shell::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
}

.hero-shell::before {
    width: 280px;
    height: 280px;
    top: -110px;
    right: -60px;
}

.hero-shell::after {
    width: 220px;
    height: 220px;
    bottom: -120px;
    left: -70px;
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 1.5rem;
    align-items: end;
}

.eyebrow {
    display: inline-flex;
    padding: 0.38rem 0.72rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.18);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.9rem;
}

.hero-title {
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
    font-size: 2.65rem;
    line-height: 1.02;
    margin: 0 0 0.8rem;
    color: #f7fbff !important;
}

.hero-copy p {
    max-width: 760px;
    margin: 0;
    font-size: 1rem;
    line-height: 1.65;
    color: rgba(247, 251, 255, 0.94) !important;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.hero-chip {
    padding: 0.45rem 0.78rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.16);
    font-size: 0.86rem;
    color: #f7fbff !important;
}

.hero-kpis {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.8rem;
}

.hero-visual {
    position: relative;
    min-height: 320px;
    display: flex;
    align-items: end;
    justify-content: center;
}

.hero-visual::before {
    content: "";
    position: absolute;
    inset: 1rem 0.5rem 0.3rem;
    border-radius: 28px;
    background: linear-gradient(180deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.04) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(8px);
}

.hero-scene {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 430px;
    animation: boatDrift 8s ease-in-out infinite;
}

.hero-scene svg {
    width: 100%;
    height: auto;
    display: block;
    filter: drop-shadow(0 18px 30px rgba(0,0,0,0.18));
}

.hero-float {
    position: absolute;
    z-index: 1;
    color: rgba(255,255,255,0.84);
    font-size: 1.5rem;
    animation: floatPulse 4.8s ease-in-out infinite;
}

.hero-float.left {
    left: 0.8rem;
    top: 2rem;
}

.hero-float.right {
    right: 1.2rem;
    top: 3.4rem;
    animation-delay: 1.2s;
}

.hero-ribbon {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 1rem;
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(247,251,255,0.96) !important;
    font-size: 0.9rem;
}

.hero-shell .eyebrow,
.hero-shell .hero-title,
.hero-shell .hero-copy p,
.hero-shell .hero-chip,
.hero-shell .hero-ribbon,
.hero-shell .hero-tags span,
.hero-shell .hero-copy span {
    color: #f7fbff !important;
}

.hero-shell .hero-copy p {
    opacity: 1 !important;
}

.hero-stat,
.insight-card,
.result-shell {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(244, 249, 252, 0.98) 100%);
    border: 1px solid rgba(31, 78, 121, 0.1);
    border-radius: 22px;
    box-shadow: 0 14px 32px rgba(15, 39, 66, 0.09);
}

.hero-stat {
    padding: 1rem 1rem 0.9rem;
    color: var(--sea-ink);
}

.hero-stat-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #53718d;
}

.hero-stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.2rem;
    color: #0f2742;
}

.hero-stat-note {
    margin-top: 0.2rem;
    font-size: 0.88rem;
    color: #526d87;
}

.section-banner {
    border-left: 6px solid var(--sea-teal);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.86);
    padding: 1.05rem 1.15rem;
    margin: 0.45rem 0 1rem;
    box-shadow: 0 14px 30px rgba(15, 39, 66, 0.07);
    animation: riseIn 0.55s ease;
}

.section-banner h3 {
    margin: 0;
    color: var(--sea-ink);
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
}

.section-banner p {
    margin: 0.35rem 0 0;
    color: #526d87;
}

.insight-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.95rem;
    margin-bottom: 1rem;
}

.insight-card {
    padding: 1rem 1rem 0.9rem;
    animation: riseIn 0.55s ease;
}

.insight-card h4 {
    margin: 0 0 0.45rem;
    color: #153b61;
    font-size: 1rem;
}

.insight-card p {
    margin: 0;
    color: #58748e;
    line-height: 1.55;
}

.result-shell {
    padding: 1.15rem 1.2rem;
    margin: 0.4rem 0 1rem;
    animation: riseIn 0.55s ease;
}

.result-top {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: start;
}

.result-top h3 {
    margin: 0.1rem 0 0.2rem;
    color: var(--sea-ink);
}

.result-top p {
    margin: 0;
    color: #60798f;
}

.status-pill {
    padding: 0.48rem 0.86rem;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.03em;
    white-space: nowrap;
}

.status-positive {
    background: rgba(37, 152, 95, 0.12);
    color: #146c43;
}

.status-negative {
    background: rgba(199, 62, 67, 0.12);
    color: #a3212a;
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin-top: 1rem;
}

.result-card {
    border-radius: 18px;
    padding: 0.95rem 1rem;
    background: #f7fbff;
    border: 1px solid rgba(31, 78, 121, 0.1);
}

.result-card span {
    display: block;
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6a8498;
}

.result-card strong {
    display: block;
    margin-top: 0.35rem;
    font-size: 1.35rem;
    color: #153b61;
}

.mode-points {
    margin: 0.7rem 0 0;
    padding-left: 1rem;
    color: #4f6980;
}

.analysis-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.95fr;
    gap: 0.95rem;
    margin: 1rem 0;
}

.analysis-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(244,249,252,0.98) 100%);
    border: 1px solid rgba(31, 78, 121, 0.1);
    border-radius: 22px;
    padding: 1.05rem 1.1rem;
    box-shadow: 0 14px 32px rgba(15, 39, 66, 0.09);
}

.analysis-card h4 {
    margin: 0 0 0.7rem;
    color: #153b61;
    font-size: 1.05rem;
}

.analysis-card ul {
    margin: 0;
    padding-left: 1rem;
    color: #566f85;
}

.analysis-card li {
    margin: 0 0 0.55rem;
    line-height: 1.5;
}

.meter-stack {
    display: grid;
    gap: 0.75rem;
}

.meter-row strong {
    display: block;
    color: #153b61;
    margin-bottom: 0.2rem;
}

.meter-row span {
    display: block;
    color: #60798f;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

.meter-track {
    width: 100%;
    height: 12px;
    border-radius: 999px;
    background: rgba(31, 78, 121, 0.1);
    overflow: hidden;
}

.meter-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #1f7a8c 0%, #1f4e79 100%);
}

.comparison-shell {
    margin-top: 0.6rem;
    margin-bottom: 0.8rem;
}

.download-note {
    margin-top: 0.45rem;
    color: #60798f;
    font-size: 0.92rem;
}

.section-anchor {
    display: block;
    position: relative;
    top: -18px;
    visibility: hidden;
}

.website-section {
    margin-bottom: 1rem;
}

[data-testid="stHorizontalBlock"] > div:has(> div[role="radiogroup"]) {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(31, 78, 121, 0.1);
    border-radius: 22px;
    padding: 0.35rem;
    box-shadow: 0 14px 28px rgba(15, 39, 66, 0.06);
}

div[role="radiogroup"] {
    gap: 0.45rem !important;
}

div[role="radiogroup"] label {
    border-radius: 999px !important;
    padding: 0.5rem 0.9rem !important;
    background: transparent;
    transition: background 0.18s ease, transform 0.18s ease;
}

div[role="radiogroup"] label * {
    color: #163754 !important;
}

div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(15, 39, 66, 0.96), rgba(31, 122, 140, 0.96));
}

div[role="radiogroup"] label:has(input:checked) * {
    color: #ffffff !important;
}

@media (max-width: 1100px) {
    .site-nav-shell,
    .hero-grid,
    .insight-grid,
    .result-grid,
    .analysis-grid {
        grid-template-columns: 1fr;
    }

    .site-nav-shell {
        flex-direction: column;
        align-items: stretch;
    }

    .nav-links {
        justify-content: flex-start;
    }

    .hero-visual {
        min-height: 260px;
        margin-top: 0.6rem;
    }

    .site-nav {
        margin-bottom: 0.85rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

if RESULT_STATE_KEY not in st.session_state:
    st.session_state[RESULT_STATE_KEY] = None
if RESULT_SOURCE_KEY not in st.session_state:
    st.session_state[RESULT_SOURCE_KEY] = None
if DEMO_RESULTS_KEY not in st.session_state:
    st.session_state[DEMO_RESULTS_KEY] = None
if DEMO_REPORT_KEY not in st.session_state:
    st.session_state[DEMO_REPORT_KEY] = None
if INPUT_STATE_KEY not in st.session_state:
    st.session_state[INPUT_STATE_KEY] = None
if COMPARISON_STATE_KEY not in st.session_state:
    st.session_state[COMPARISON_STATE_KEY] = None


def load_validation_snapshot() -> dict[str, object]:
    if not METRICS_PATH.exists():
        return {
            "dataset_rows": 0,
            "best_availability_model": "Random Forest",
            "best_availability_accuracy": 0.0,
            "best_quantity_model": "Random Forest",
            "best_quantity_rmse": 0.0,
            "juvenile_accuracy": 0.0,
            "exact_ready_rows": 0,
            "field_rows": 0,
            "tests_ready": 0,
        }

    try:
        data = json.loads(METRICS_PATH.read_text())
    except json.JSONDecodeError:
        return {
            "dataset_rows": 0,
            "best_availability_model": "Random Forest",
            "best_availability_accuracy": 0.0,
            "best_quantity_model": "Random Forest",
            "best_quantity_rmse": 0.0,
            "juvenile_accuracy": 0.0,
            "exact_ready_rows": 0,
            "field_rows": 0,
            "tests_ready": 0,
        }

    main_models = data.get("main_models", {})
    juvenile_model = data.get("juvenile_model", {})
    field_data = data.get("field_data", {})

    availability_scores = {
        "Random Forest": main_models.get("random_forest", {}).get("availability", {}).get("accuracy", 0.0),
        "Boosting": main_models.get("boosting", {}).get("availability", {}).get("accuracy", 0.0),
        "Hybrid": main_models.get("hybrid", {}).get("availability", {}).get("accuracy", 0.0),
    }
    quantity_scores = {
        "Random Forest": main_models.get("random_forest", {}).get("quantity", {}).get("rmse", float("inf")),
        "Boosting": main_models.get("boosting", {}).get("quantity", {}).get("rmse", float("inf")),
        "Hybrid": main_models.get("hybrid", {}).get("quantity", {}).get("rmse", float("inf")),
    }
    demo_checks = data.get("demo_checks", [])

    return {
        "dataset_rows": int(main_models.get("dataset_rows", 0)),
        "best_availability_model": max(availability_scores, key=availability_scores.get),
        "best_availability_accuracy": float(max(availability_scores.values(), default=0.0)),
        "best_quantity_model": min(quantity_scores, key=quantity_scores.get),
        "best_quantity_rmse": float(min(quantity_scores.values(), default=0.0)),
        "juvenile_accuracy": float(juvenile_model.get("metrics", {}).get("accuracy", 0.0)),
        "exact_ready_rows": int(field_data.get("exact_ready_rows", 0)),
        "field_rows": int(field_data.get("rows", 0)),
        "tests_ready": sum(1 for check in demo_checks if check.get("passed")),
    }


def render_hero(snapshot: dict[str, object], active_pipeline: str) -> None:
    st.markdown(
        f"""
<div id="overview" class="section-anchor"></div>
<section class="hero-shell">
  <div class="hero-grid">
    <div class="hero-copy">
      <span class="eyebrow">Sustainable Marine Intelligence</span>
      <h1 class="hero-title">AI-Driven Fish Catch Prediction System</h1>
      <p>
        A presentation-ready fisheries interface that combines fish availability prediction, catch quantity
        estimation, juvenile-risk screening, and safe-zone recommendation from real project data.
      </p>
      <div class="hero-tags">
        <span class="hero-chip">Active pipeline: {active_pipeline}</span>
        <span class="hero-chip">Best classifier: {snapshot['best_availability_model']}</span>
        <span class="hero-chip">Best regressor: {snapshot['best_quantity_model']}</span>
        <span class="hero-chip">Exact juvenile rule enabled</span>
      </div>
      <div class="hero-ribbon">Coastal decision support for fishers, field teams, and sustainable fisheries review</div>
    </div>
    <div class="hero-visual">
      <div class="hero-float left">~</div>
      <div class="hero-float right">~</div>
      <div class="hero-scene">
        <svg viewBox="0 0 520 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Fishing boat scene">
          <defs>
            <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#f9d18b"/>
              <stop offset="48%" stop-color="#79c9dd"/>
              <stop offset="100%" stop-color="#1b5d8f"/>
            </linearGradient>
            <linearGradient id="sea" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#1f7a8c"/>
              <stop offset="100%" stop-color="#163f6a"/>
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="520" height="360" rx="28" fill="url(#sky)"/>
          <circle cx="404" cy="82" r="36" fill="#ffd77a" opacity="0.92"/>
          <path d="M0 214 C55 194, 116 188, 182 206 C245 224, 305 229, 363 210 C420 191, 473 191, 520 212 L520 360 L0 360 Z" fill="url(#sea)"/>
          <path d="M0 246 C62 228, 124 225, 190 242 C253 258, 326 263, 392 244 C450 228, 487 229, 520 239" stroke="rgba(255,255,255,0.42)" stroke-width="5" fill="none"/>
          <path d="M0 280 C62 262, 124 259, 190 276 C253 292, 326 297, 392 278 C450 262, 487 263, 520 273" stroke="rgba(255,255,255,0.34)" stroke-width="4" fill="none"/>
          <path d="M145 215 L307 215 L336 253 L184 253 Z" fill="#673f2d"/>
          <path d="M170 198 L276 198 L307 215 L145 215 Z" fill="#8b5a3d"/>
          <rect x="224" y="122" width="8" height="78" fill="#2a2a2a"/>
          <path d="M232 128 L285 155 L232 183 Z" fill="#f7fbff" opacity="0.96"/>
          <path d="M215 152 L170 180" stroke="#2a2a2a" stroke-width="4"/>
          <circle cx="205" cy="145" r="10" fill="#14243a"/>
          <path d="M205 155 L205 188" stroke="#14243a" stroke-width="7" stroke-linecap="round"/>
          <path d="M205 170 L187 188" stroke="#14243a" stroke-width="6" stroke-linecap="round"/>
          <path d="M205 169 L223 187" stroke="#14243a" stroke-width="6" stroke-linecap="round"/>
          <path d="M205 188 L190 208" stroke="#14243a" stroke-width="6" stroke-linecap="round"/>
          <path d="M205 188 L220 208" stroke="#14243a" stroke-width="6" stroke-linecap="round"/>
          <path d="M170 180 C155 212, 132 235, 111 246" stroke="#f1f5f9" stroke-width="3" fill="none" opacity="0.85"/>
          <ellipse cx="109" cy="248" rx="18" ry="8" fill="#d6eef2" opacity="0.9"/>
          <path d="M100 248 L84 241" stroke="#d6eef2" stroke-width="4" stroke-linecap="round"/>
          <path d="M100 248 L84 255" stroke="#d6eef2" stroke-width="4" stroke-linecap="round"/>
          <path d="M342 92 Q356 80 369 93" stroke="#ffffff" stroke-width="3" fill="none" opacity="0.9"/>
          <path d="M367 96 Q381 84 394 97" stroke="#ffffff" stroke-width="3" fill="none" opacity="0.88"/>
        </svg>
      </div>
      <div class="hero-kpis" style="position:absolute; left: 0.9rem; right: 0.9rem; bottom: 0.9rem; z-index:2;">
        <div class="hero-stat">
          <div class="hero-stat-label">Availability Accuracy</div>
          <div class="hero-stat-value">{float(snapshot['best_availability_accuracy']) * 100:.2f}%</div>
          <div class="hero-stat-note">{snapshot['best_availability_model']}</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-label">Juvenile Accuracy</div>
          <div class="hero-stat-value">{float(snapshot['juvenile_accuracy']) * 100:.2f}%</div>
          <div class="hero-stat-note">Balanced juvenile model</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-label">Training Rows</div>
          <div class="hero-stat-value">{int(snapshot['dataset_rows'])}</div>
          <div class="hero-stat-note">{int(snapshot['field_rows'])} field observations</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-label">Exact-Ready Rows</div>
          <div class="hero-stat-value">{int(snapshot['exact_ready_rows'])}</div>
          <div class="hero-stat-note">{int(snapshot['tests_ready'])} demo checks passing</div>
        </div>
      </div>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_top_navigation() -> None:
    st.markdown(
        """
<nav class="site-nav">
  <div class="site-nav-shell">
    <div class="nav-brand">
      <div class="nav-mark">FC</div>
      <div>
        <div class="nav-title">Fish Catch Prediction System</div>
      </div>
    </div>
    <div class="nav-links">
      <a href="#overview">Overview</a>
      <a href="#field-intel">Field Intel</a>
      <a href="#prediction-lab">Prediction Lab</a>
      <a href="#viva-demo">Viva Demo</a>
    </div>
  </div>
</nav>
""",
        unsafe_allow_html=True,
    )


def section_banner(title: str, description: str) -> str:
    return f"""
<div class="section-banner">
  <h3>{title}</h3>
  <p>{description}</p>
</div>
"""


def render_insight_cards(cards: list[tuple[str, str]]) -> None:
    html = ["<div class='insight-grid'>"]
    for title, description in cards:
        html.append(
            f"""
  <div class="insight-card">
    <h4>{title}</h4>
    <p>{description}</p>
  </div>
"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_mode_intro(selected_mode: str) -> None:
    mode_cards = {
        "Manual Input": [
            ("Precision Control", "Tune SST, salinity, dissolved oxygen, and catch history to compare outputs under custom marine conditions."),
            ("Best for Viva", "Use this mode when you want to explain how each environmental factor influences the final decision."),
            ("Exact Juvenile Logic", "Add species, observed length, and maturity length to trigger the biological risk path."),
        ],
        "Select Region": [
            ("Quick Demonstration", "Pick a predefined coastal region and show a complete prediction without entering every parameter manually."),
            ("Scenario Ready", "Useful when teachers ask for fast comparisons between project locations such as Vizag, Kakinada, and Kochi."),
            ("Consistent Inputs", "Each region includes preloaded SST, salinity, oxygen, and historical catch values."),
        ],
        "Map Based GPS Input": [
            ("Spatial Exploration", "Click directly on the map to simulate GPS-based advisory behavior for field-style usage."),
            ("Safe-Zone Story", "This mode is great for demonstrating zone rejection and alternate safer location suggestions."),
            ("Location First", "Choose the point visually, then layer environmental conditions and maturity information on top."),
        ],
    }
    descriptions = {
        "Manual Input": "Design a custom ocean-condition scenario and compare how the project responds under different fisheries parameters.",
        "Select Region": "Use curated coastal presets for a fast, stable demonstration of the full prediction pipeline.",
        "Map Based GPS Input": "Explore the project spatially by selecting coordinates first and then evaluating risk and advisory output.",
    }
    st.markdown(section_banner(selected_mode, descriptions[selected_mode]), unsafe_allow_html=True)
    render_insight_cards(mode_cards[selected_mode])


def render_field_intelligence(summary: dict[str, int], snapshot: dict[str, object]) -> None:
    st.markdown('<div id="field-intel" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        section_banner(
            "Field Intelligence",
            "A quick project-status section that shows the biological readiness of the dataset and the current validation posture.",
        ),
        unsafe_allow_html=True,
    )
    render_insight_cards(
        [
            (
                "Observation Coverage",
                f"{summary['rows']} field records are stored right now, covering {summary['states']} coastal states and {summary['species']} species entries.",
            ),
            (
                "Exact-Ready Biology",
                f"{summary['exact_ready_rows']} rows currently support exact juvenile reasoning through observed and maturity lengths.",
            ),
            (
                "Validation Snapshot",
                f"{int(snapshot['tests_ready'])} demo checks are passing and the latest dataset includes {int(snapshot['dataset_rows'])} training rows.",
            ),
        ]
    )


def render_result_header(result: PredictionResult) -> None:
    status_class = "status-positive" if result.availability else "status-negative"
    status_text = "Fishing Window Open" if result.availability else "Risk-Controlled Zone"
    species_line = f"Species context: {result.species}" if result.species else "Species context: environmental and catch features only"
    st.markdown(
        f"""
<div class="result-shell">
  <div class="result-top">
    <div>
      <span class="eyebrow" style="background: rgba(31, 78, 121, 0.08); color: #1f4e79; border-color: rgba(31, 78, 121, 0.12);">Prediction Summary</span>
      <h3>{result.location}</h3>
      <p>{species_line}</p>
    </div>
    <div class="status-pill {status_class}">{status_text}</div>
  </div>
  <div class="result-grid">
    <div class="result-card">
      <span>Fish Availability</span>
      <strong>{"YES" if result.availability else "NO"}</strong>
    </div>
    <div class="result-card">
      <span>Catch Quantity</span>
      <strong>{f"{result.quantity:.2f} kg" if result.availability else "Blocked"}</strong>
    </div>
    <div class="result-card">
      <span>Juvenile Risk</span>
      <strong>{result.juvenile_risk}</strong>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _clamp_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def build_reason_cards(payload: dict[str, object], result: PredictionResult) -> list[tuple[str, str]]:
    sst = float(payload["sst"])
    salinity = float(payload["salinity"])
    dissolved_oxygen = float(payload["dissolved_oxygen"])
    historical_catch = float(payload["historical_catch"])

    if 22 <= sst <= 30:
        thermal_note = f"SST at {sst:.1f} C sits inside the working thermal band, which supports the availability decision."
    else:
        thermal_note = f"SST at {sst:.1f} C is outside the preferred thermal band, so the system treats the location as thermally stressed."

    if dissolved_oxygen >= 5.5:
        oxygen_note = f"Dissolved oxygen at {dissolved_oxygen:.1f} mg/l indicates healthier water conditions for fishing activity."
    else:
        oxygen_note = f"Dissolved oxygen at {dissolved_oxygen:.1f} mg/l is low, which adds ecological caution to the recommendation."

    if salinity < 30 or salinity > 36:
        salinity_note = f"Salinity at {salinity:.1f} PSU is outside the stronger coastal operating range, reducing confidence."
    else:
        salinity_note = f"Salinity at {salinity:.1f} PSU aligns with the trained coastal range used by the models."

    if historical_catch >= 200:
        catch_note = f"Historical catch of {historical_catch:.1f} kg acts as a positive prior signal for expected availability and quantity."
    else:
        catch_note = f"Historical catch of {historical_catch:.1f} kg is modest, so the system avoids overestimating the catch window."

    biology_note = (
        f"The juvenile decision uses {result.juvenile_method.lower()} and currently classifies this zone as {result.juvenile_risk.lower()} risk."
    )
    advisory_note = (
        "Because the zone is risky or weak, the app generates safer nearby alternatives."
        if result.safe_zone_suggestions
        else "The current zone is accepted without needing an alternate safe-zone shift."
    )

    return [
        ("Environmental reasoning", thermal_note + " " + oxygen_note),
        ("Catch and habitat signal", salinity_note + " " + catch_note),
        ("Biological sustainability", biology_note + " " + advisory_note),
    ]


def readiness_metrics(payload: dict[str, object], result: PredictionResult) -> list[tuple[str, float, str]]:
    sst = float(payload["sst"])
    dissolved_oxygen = float(payload["dissolved_oxygen"])
    historical_catch = float(payload["historical_catch"])

    thermal_fit = _clamp_score(1.0 - abs(sst - 27.0) / 8.0)
    oxygen_fit = _clamp_score(dissolved_oxygen / 8.0)
    catch_strength = _clamp_score(historical_catch / 400.0)
    sustainability = _clamp_score(1.0 - result.juvenile_score)

    return [
        ("Thermal fit", thermal_fit, f"SST alignment score based on how close {sst:.1f} C is to the preferred zone."),
        ("Oxygen readiness", oxygen_fit, f"Water-column support estimated from dissolved oxygen at {dissolved_oxygen:.1f} mg/l."),
        ("Catch momentum", catch_strength, f"Historical catch strength estimated from {historical_catch:.1f} kg of prior catch."),
        ("Sustainability margin", sustainability, "Higher values indicate lower juvenile pressure and a safer fishing recommendation."),
    ]


def render_readiness_panel(payload: dict[str, object], result: PredictionResult) -> None:
    rows = []
    for label, score, note in readiness_metrics(payload, result):
        rows.append(
            f"""
<div class="meter-row">
  <strong>{label}: {score * 100:.1f}%</strong>
  <span>{note}</span>
  <div class="meter-track"><div class="meter-fill" style="width: {score * 100:.1f}%;"></div></div>
</div>
"""
        )
    st.markdown(
        "<div class='analysis-card'><h4>Marine Condition Readiness</h4><div class='meter-stack'>"
        + "".join(rows)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def compare_pipelines(payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pipeline_label, model_choice in PIPELINE_OPTIONS.items():
        compared = predict_fishing_zone(
            location=str(payload["location"]),
            sst=float(payload["sst"]),
            salinity=float(payload["salinity"]),
            dissolved_oxygen=float(payload["dissolved_oxygen"]),
            historical_catch=float(payload["historical_catch"]),
            species=payload["species"],
            latitude=payload["latitude"],
            longitude=payload["longitude"],
            model_choice=model_choice,
            observed_length_cm=payload["observed_length"],
            maturity_length_cm=payload["maturity_length"],
        )
        rows.append(
            {
                "Requested Pipeline": pipeline_label,
                "Resolved Pipeline": compared.model_pipeline,
                "Availability": "YES" if compared.availability else "NO",
                "Availability Score": round(compared.availability_score, 3),
                "Quantity (kg)": round(compared.quantity, 2),
                "Juvenile Risk": compared.juvenile_risk,
                "Juvenile Method": compared.juvenile_method,
                "Safe Zones": len(compared.safe_zone_suggestions),
            }
        )
    return rows


def advisory_export_text(payload: dict[str, object], result: PredictionResult, comparison_rows: list[dict[str, object]]) -> str:
    lines = [
        "AI-Driven Fish Catch Prediction System",
        "Prediction Advisory Summary",
        "",
        f"Mode: {payload['source_mode']}",
        f"Location: {payload['location']}",
        f"Latitude: {payload['latitude']}",
        f"Longitude: {payload['longitude']}",
        f"Pipeline used: {result.model_pipeline}",
        "",
        "Input Conditions",
        f"- SST: {float(payload['sst']):.2f} C",
        f"- Salinity: {float(payload['salinity']):.2f} PSU",
        f"- Dissolved Oxygen: {float(payload['dissolved_oxygen']):.2f} mg/l",
        f"- Historical Catch: {float(payload['historical_catch']):.2f} kg",
        f"- Species: {payload['species'] or 'Not provided'}",
        f"- Observed Length: {payload['observed_length'] if payload['observed_length'] is not None else 'Not provided'}",
        f"- Maturity Length: {payload['maturity_length'] if payload['maturity_length'] is not None else 'Not provided'}",
        "",
        "Prediction Output",
        f"- Fish Availability: {'YES' if result.availability else 'NO'}",
        f"- Catch Quantity: {result.quantity:.2f} kg",
        f"- Juvenile Risk: {result.juvenile_risk}",
        f"- Juvenile Method: {result.juvenile_method}",
        f"- Advisory: {result.advisory}",
        "",
        "Pipeline Comparison",
    ]
    for row in comparison_rows:
        lines.append(
            f"- {row['Requested Pipeline']}: availability {row['Availability']}, "
            f"score {row['Availability Score']}, quantity {row['Quantity (kg)']} kg, "
            f"juvenile risk {row['Juvenile Risk']}"
        )
    if result.safe_zone_suggestions:
        lines.extend(["", "Suggested Safer Zones"])
        for zone in result.safe_zone_suggestions:
            lines.append(
                f"- {zone['zone']}: {zone['distance_km']} km, risk {zone['expected_juvenile_risk']}, "
                f"expected quantity {zone['expected_quantity_kg']} kg"
            )
    return "\n".join(lines)


validation_snapshot = load_validation_snapshot()
field_summary = observation_summary()

render_top_navigation()

with st.sidebar:
    st.header("Prediction Setup")
    selected_pipeline = st.selectbox("ML pipeline", list(PIPELINE_OPTIONS.keys()), index=0)
    st.caption("The app now uses one shared decision engine across manual, region, and map workflows.")

    st.write("---")
    st.subheader("Field Data Collection")
    summary = field_summary
    st.caption(
        f"Rows: {summary['rows']} | Exact-ready: {summary['exact_ready_rows']} | "
        f"States: {summary['states']} | Species: {summary['species']}"
    )
    quality_notes = data_quality_issues()
    for issue in quality_notes[:3]:
        st.warning(issue)

    with st.expander("Record real observation", expanded=False):
        with st.form("field_observation_form", clear_on_submit=True):
            obs_state = st.selectbox("State", COASTAL_STATES, index=0)
            obs_date = st.date_input("Observation date")
            obs_pfz_count = st.number_input("PFZ count", min_value=1, value=1, step=1)
            obs_distance = st.number_input("Distance from shore (km)", min_value=0.0, value=18.5, step=0.5)
            obs_depth = st.number_input("Water depth (m)", min_value=0.0, value=55.0, step=1.0)
            obs_species_options = ["Custom / Unknown species"] + known_species()
            obs_species_choice = st.selectbox("Species observed", obs_species_options, index=0)
            obs_species = None if obs_species_choice == "Custom / Unknown species" else obs_species_choice
            obs_length = st.number_input("Observed fish length (cm)", min_value=0.0, value=12.0, step=0.5)
            default_obs_maturity = 0.0
            if obs_species is not None:
                looked_up_length, _ = lookup_maturity_length(obs_species)
                if looked_up_length is not None:
                    default_obs_maturity = float(looked_up_length)
            obs_maturity = st.number_input("Maturity length (cm)", min_value=0.0, value=default_obs_maturity, step=0.5)
            obs_source = st.text_input("Data source", value="INCOIS PFZ + field landing observation")
            obs_recorder = st.text_input("Recorded by", value="")
            obs_notes = st.text_area("Notes", value="")
            save_obs = st.form_submit_button("Save observation")

        if save_obs:
            append_pfz_observation(
                state=obs_state,
                observed_date=obs_date,
                pfz_count=int(obs_pfz_count),
                distance_km=float(obs_distance),
                depth_m=float(obs_depth),
                species=obs_species,
                observed_length_cm=float(obs_length) if obs_length > 0 else None,
                maturity_length_cm=float(obs_maturity) if obs_maturity > 0 else None,
                data_source=obs_source,
                recorded_by=obs_recorder,
                notes=obs_notes,
            )
            st.success("Observation saved to data/external/incois_pfz.csv. Rerun the full pipeline to train on it.")

    st.caption("Batch import command: .venv/bin/python src/import_pfz_observations.py your_file.csv")

    recent_frame = recent_observations(limit=5)
    if not recent_frame.empty:
        st.caption("Recent observations")
        st.dataframe(recent_frame, use_container_width=True, hide_index=True)

render_hero(validation_snapshot, selected_pipeline)
render_field_intelligence(field_summary, validation_snapshot)
st.markdown('<div id="prediction-lab" class="section-anchor"></div>', unsafe_allow_html=True)
menu = st.radio("Choose prediction method", ["Manual Input", "Select Region", "Map Based GPS Input"], horizontal=True)
render_mode_intro(menu)


def maturity_inputs(prefix: str) -> tuple[str | None, float | None, float | None]:
    enabled = st.checkbox("Use maturity-based juvenile risk", key=f"{prefix}_maturity")
    if not enabled:
        return None, None, None

    species_options = ["Custom / Unknown species"] + known_species()

    with st.expander("Maturity-based juvenile-risk inputs", expanded=True):
        species = st.selectbox("Species", species_options, key=f"{prefix}_species")
        observed_length = st.number_input(
            "Observed fish length (cm)",
            min_value=0.0,
            value=12.0,
            step=0.5,
            key=f"{prefix}_observed_length",
        )
        st.caption("Exact juvenile formula: JR = 1 - observed_length / maturity_length")
        default_maturity = 18.0
        if species != "Custom / Unknown species":
            looked_up_length, _ = lookup_maturity_length(species)
            if looked_up_length is not None:
                default_maturity = looked_up_length
        maturity_length = st.number_input(
            "Species maturity length (cm)",
            min_value=0.0,
            value=default_maturity,
            step=0.5,
            key=f"{prefix}_maturity_length",
        )

    selected_species = None if species == "Custom / Unknown species" else species
    return selected_species, observed_length or None, maturity_length or None


def display_output(result: PredictionResult, payload: dict[str, object] | None = None) -> None:
    render_result_header(result)

    column_1, column_2, column_3 = st.columns(3)
    column_1.metric("Fish Availability", "YES" if result.availability else "NO")
    column_2.metric("Catch Quantity", f"{result.quantity:.2f} kg" if result.availability else "Blocked")
    column_3.metric("Juvenile Risk", result.juvenile_risk)

    st.info(result.advisory)
    st.caption(
        f"Pipeline: {result.model_pipeline} | Availability score: {result.availability_score:.2f} | "
        f"Juvenile score: {result.juvenile_score:.2f}"
    )
    st.caption(f"Juvenile method: {result.juvenile_method}")

    detail_columns = st.columns(2)
    detail_columns[0].metric("Base Juvenile Layer", result.base_juvenile_risk)
    detail_columns[1].metric(
        "Maturity Score",
        f"{result.maturity_score:.2f}" if result.maturity_score is not None else "Not used",
    )
    if result.maturity_length_cm is not None:
        st.caption(f"Applied maturity length: {result.maturity_length_cm:.2f} cm")

    if payload is not None:
        reason_cards = build_reason_cards(payload, result)
        insight_left, insight_right = st.columns([1.15, 0.95])
        with insight_left:
            st.markdown(
                "<div class='analysis-card'><h4>Why This Result?</h4><ul>"
                + "".join(f"<li><strong>{title}:</strong> {description}</li>" for title, description in reason_cards)
                + "</ul></div>",
                unsafe_allow_html=True,
            )
        with insight_right:
            render_readiness_panel(payload, result)

        comparison_rows = st.session_state.get(COMPARISON_STATE_KEY) or []
        if comparison_rows:
            st.markdown(
                section_banner(
                    "Scenario Comparator",
                    "Same input, three pipelines. This gives you a strong presentation feature for showing how the models behave under one marine scenario.",
                ),
                unsafe_allow_html=True,
            )
            comparison_frame = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
            st.bar_chart(
                comparison_frame.set_index("Requested Pipeline")[["Availability Score"]],
                use_container_width=True,
            )

            advisory_text = advisory_export_text(payload, result, comparison_rows)
            st.download_button(
                "Download Advisory Summary",
                data=advisory_text,
                file_name=f"{str(payload['source_mode']).lower().replace(' ', '_')}_advisory_summary.txt",
                mime="text/plain",
            )
            st.markdown(
                "<div class='download-note'>Export this summary to show inputs, final recommendation, and model comparison during review.</div>",
                unsafe_allow_html=True,
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
    source_mode: str,
    location: str,
    sst: float,
    salinity: float,
    dissolved_oxygen: float,
    historical_catch: float,
    species: str | None,
    latitude: float | None,
    longitude: float | None,
    observed_length: float | None,
    maturity_length: float | None,
) -> None:
    payload = {
        "source_mode": source_mode,
        "location": location,
        "sst": float(sst),
        "salinity": float(salinity),
        "dissolved_oxygen": float(dissolved_oxygen),
        "historical_catch": float(historical_catch),
        "species": species,
        "latitude": latitude,
        "longitude": longitude,
        "observed_length": observed_length,
        "maturity_length": maturity_length,
    }
    result = predict_fishing_zone(
        location=location,
        sst=sst,
        salinity=salinity,
        dissolved_oxygen=dissolved_oxygen,
        historical_catch=historical_catch,
        species=species,
        latitude=latitude,
        longitude=longitude,
        model_choice=PIPELINE_OPTIONS[selected_pipeline],
        observed_length_cm=observed_length,
        maturity_length_cm=maturity_length,
    )
    st.session_state[RESULT_STATE_KEY] = result
    st.session_state[RESULT_SOURCE_KEY] = source_mode
    st.session_state[INPUT_STATE_KEY] = payload
    st.session_state[COMPARISON_STATE_KEY] = compare_pipelines(payload)


def display_saved_result(source_mode: str) -> None:
    result = st.session_state.get(RESULT_STATE_KEY)
    result_source = st.session_state.get(RESULT_SOURCE_KEY)
    if result is not None and result_source == source_mode:
        display_output(result, st.session_state.get(INPUT_STATE_KEY))


if menu == "Manual Input":
    st.markdown(
        section_banner(
            "Manual Parameter Entry",
            "Build a tailored marine scenario and present how the model responds to direct environmental inputs.",
        ),
        unsafe_allow_html=True,
    )
    with st.form("manual_prediction_form"):
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
        species, observed_length, maturity_length = maturity_inputs("manual")
        predict_manual = st.form_submit_button("Predict (Manual)")

    if predict_manual:
        run_prediction(
            source_mode="Manual Input",
            location=location,
            sst=sst,
            salinity=salinity,
            dissolved_oxygen=dissolved_oxygen,
            historical_catch=historical_catch,
            species=species,
            latitude=latitude,
            longitude=longitude,
            observed_length=observed_length,
            maturity_length=maturity_length,
        )

    display_saved_result("Manual Input")

elif menu == "Select Region":
    st.markdown(
        section_banner(
            "Region-Based Prediction",
            "Use coastal presets to demonstrate the project quickly while keeping the underlying model logic intact.",
        ),
        unsafe_allow_html=True,
    )
    with st.form("region_prediction_form"):
        region = st.selectbox("Select coastal zone", list(REGIONS.keys()))
        config = REGIONS[region]
        species, observed_length, maturity_length = maturity_inputs("region")

        st.write(
            f"Preloaded conditions: SST {config['sst']} C, Salinity {config['salinity']} PSU, "
            f"DO {config['do']} mg/l, Historical Catch {config['history']} kg"
        )

        predict_region = st.form_submit_button("Predict (Region Based)")

    if predict_region:
        run_prediction(
            source_mode="Select Region",
            location=region,
            sst=config["sst"],
            salinity=config["salinity"],
            dissolved_oxygen=config["do"],
            historical_catch=config["history"],
            species=species,
            latitude=config["lat"],
            longitude=config["lon"],
            observed_length=observed_length,
            maturity_length=maturity_length,
        )

    display_saved_result("Select Region")

else:
    st.markdown(
        section_banner(
            "Map Based GPS Input",
            "Select a sea location visually, then layer environmental conditions and juvenile-risk logic on top of it.",
        ),
        unsafe_allow_html=True,
    )
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

        with st.form("map_prediction_form"):
            species, observed_length, maturity_length = maturity_inputs("map")
            predict_map = st.form_submit_button("Predict From Map")

        if predict_map:
            run_prediction(
                source_mode="Map Based GPS Input",
                location=f"Lat {latitude:.3f}, Lon {longitude:.3f}",
                sst=sst,
                salinity=salinity,
                dissolved_oxygen=dissolved_oxygen,
                historical_catch=historical_catch,
                species=species,
                latitude=latitude,
                longitude=longitude,
                observed_length=observed_length,
                maturity_length=maturity_length,
            )

        display_saved_result("Map Based GPS Input")

st.write("---")
st.markdown('<div id="viva-demo" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown(
    section_banner(
        "Viva Algorithm Demo",
        "Show the execution of every algorithm path, compare the improved metrics, and open the exact project code blocks during your presentation.",
    ),
    unsafe_allow_html=True,
)
render_insight_cards(
    [
        (
            "Best Availability Model",
            f"{validation_snapshot['best_availability_model']} currently leads with {float(validation_snapshot['best_availability_accuracy']) * 100:.2f}% accuracy.",
        ),
        (
            "Best Quantity Model",
            f"{validation_snapshot['best_quantity_model']} currently gives the lowest RMSE of {float(validation_snapshot['best_quantity_rmse']):.2f}.",
        ),
        (
            "Biological Readiness",
            f"{int(validation_snapshot['exact_ready_rows'])} exact-ready rows are available for maturity-driven juvenile assessment.",
        ),
    ]
)

runtime_frame = pd.DataFrame(get_runtime_summary())
st.dataframe(runtime_frame, use_container_width=True, hide_index=True)

st.subheader("Before vs After Improvements")
st.caption("This table compares the earlier baseline against the current improved models after feature engineering, balancing, and retraining.")
balancing_frame = pd.DataFrame(get_balancing_comparison())
st.dataframe(balancing_frame, use_container_width=True, hide_index=True)

demo_actions = st.columns([1, 2])
with demo_actions[0]:
    if st.button("Run All Algorithm Demos"):
        st.session_state[DEMO_RESULTS_KEY] = run_algorithm_demos()
        st.session_state[DEMO_REPORT_KEY] = str(write_algorithm_execution_report())
with demo_actions[1]:
    report_path = st.session_state.get(DEMO_REPORT_KEY)
    if report_path:
        st.success(f"Latest viva report: {report_path}")
    else:
        st.info("Run the demo once to generate reports/algorithm_execution_report.md")

execution_tab, code_tab = st.tabs(["Execution Results", "Code Blocks"])

with execution_tab:
    demo_results = st.session_state.get(DEMO_RESULTS_KEY)
    if demo_results:
        demo_frame = pd.DataFrame(demo_results)
        st.dataframe(
            demo_frame[
                [
                    "case",
                    "resolved_pipeline",
                    "availability",
                    "quantity_kg",
                    "juvenile_risk",
                    "juvenile_method",
                    "safe_zone_count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        for item in demo_results:
            with st.expander(item["case"], expanded=False):
                st.write(item["description"])
                st.caption(
                    f"Requested: {item['requested_pipeline']} | Resolved: {item['resolved_pipeline']} | "
                    f"Availability score: {item['availability_score']:.3f}"
                )
                st.write(f"Quantity: {item['quantity_kg']:.2f} kg")
                st.write(f"Juvenile risk: {item['juvenile_risk']}")
                st.write(f"Juvenile method: {item['juvenile_method']}")
                st.write(f"Advisory: {item['advisory']}")
    else:
        st.info("Click 'Run All Algorithm Demos' to execute Random Forest, Boosting, Hybrid, and juvenile-risk demo cases.")

with code_tab:
    for item in get_code_sections():
        with st.expander(item["title"], expanded=False):
            st.caption(item["description"])
            st.caption(f"Source: {item['path']}")
            st.code(item["code"], language="python")
