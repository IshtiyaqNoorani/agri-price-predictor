import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = Path("data/processed/weekly_data.csv")
MODEL_DIR = Path("models/weekly_models")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data = load_data()

# --------------------------------------------------
# Supported commodities
# --------------------------------------------------
SUPPORTED_COMMODITIES = sorted([
    "Tomato",
    "Pumpkin",
    "Brinjal",
    "Cauliflower",
    "Onion",
    "Cabbage",
    "Potato",
    "Cucumbar(Kheera)",
])

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Agri Price Predictor",
    page_icon="🌾",
    layout="centered"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🌾 Agri Price Predictor")
st.subheader("Weekly Expected Mandi Prices")

st.markdown(
    "This application estimates **weekly average mandi prices** for selected "
    "agricultural commodities using historical data and machine learning."
)

st.info(
    "ℹ️ Prices are shown **per quintal (100 kg)** with an approximate **per-kg** value. "
    "Predictions are **not live market prices**."
)

st.markdown("---")

# --------------------------------------------------
# Sidebar — Inputs
# --------------------------------------------------
with st.sidebar:
    st.header("🔧 Prediction Settings")

    commodity = st.selectbox(
        "Select Commodity",
        SUPPORTED_COMMODITIES
    )

    state = st.selectbox(
        "Select State",
        sorted(
            data[data["Commodity"] == commodity]["State"].unique()
        )
    )

    selected_date = st.date_input(
        "Select Date (seasonal context)",
        value=date.today()
    )

    st.caption(
        "The selected date is used only for seasonal context.\n"
        "Predictions represent a typical week around this time of year."
    )

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.header("📊 Price Prediction")

if st.button("🔍 Predict Weekly Price"):
    model_file = MODEL_DIR / f"{commodity.replace(' ', '_').lower()}.pkl"

    if not model_file.exists():
        st.error("❌ Prediction model not available for this commodity.")
    else:
        model = joblib.load(model_file)

        input_df = pd.DataFrame([{
            "State": state,
            "Year": selected_date.year,
            "Month": selected_date.month,
            "DayOfWeek": selected_date.weekday(),
        }])

        prediction = model.predict(input_df)[0]

        # --------------------------------------------------
        # HTML Card (RELIABLE RENDERING)
        # --------------------------------------------------
        html_card = f"""
        <div style="
            background-color:#f0f9f4;
            padding:26px;
            border-radius:16px;
            border-left:6px solid #2ecc71;
            max-width:650px;
            font-family: Arial, sans-serif;
        ">
            <h3>✅ Weekly Expected Price</h3>

            <p><strong>Commodity:</strong> {commodity}</p>
            <p><strong>State:</strong> {state}</p>

            <h1 style="color:#27ae60; margin:18px 0;">
                ₹ {prediction:,.2f}
            </h1>

            <p style="color:#555;">
                per quintal (≈ ₹ {prediction / 100:.2f} per kg)
            </p>

            <p style="font-size:14px; color:#666; margin-top:12px;">
                Expected weekly average based on historical mandi trends.
            </p>
        </div>
        """

        components.html(html_card, height=320)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Python, Machine Learning & Streamlit · "
    "Data source: Government of India (data.gov.in)"
)

