import streamlit as st
import pandas as pd
import pickle


# ==========================================
# Load trained HDB model
# ==========================================

with open("hdb_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# Extract categories from trained encoder
# ==========================================

preprocessor = model.named_steps["preprocessor"]

encoder = preprocessor.named_transformers_["cat"]

flat_type_options = list(encoder.categories_[0])
town_options = list(encoder.categories_[1])
storey_range_options = list(encoder.categories_[2])
flat_model_options = list(encoder.categories_[3])


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("🏠 HDB Resale Price Predictor")

st.write(
    "Estimate the resale price of an HDB flat "
    "using the trained Linear Regression model."
)

st.divider()


# ==========================================
# HDB Property Information
# ==========================================

st.subheader("🏢 Flat Information")


flat_type = st.selectbox(
    "Flat Type",
    flat_type_options
)


town = st.selectbox(
    "Town",
    town_options
)


storey_range = st.selectbox(
    "Storey Range",
    storey_range_options
)


flat_model = st.selectbox(
    "Flat Model",
    flat_model_options
)


# ==========================================
# Numerical Information
# ==========================================

st.subheader("📐 Property Details")


floor_area_sqm = st.number_input(
    "Floor Area (sqm)",
    min_value=20.0,
    max_value=300.0,
    value=90.0,
    step=1.0
)


remaining_lease = st.number_input(
    "Remaining Lease (years)",
    min_value=1.0,
    max_value=99.0,
    value=70.0,
    step=1.0
)


transact_year = st.number_input(
    "Transaction Year",
    min_value=2007,
    max_value=2016,
    value=2016,
    step=1
)


st.divider()


# ==========================================
# Prediction Button
# ==========================================

if st.button("🔮 Predict Resale Price", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "flat_type": [flat_type],
        "town": [town],
        "storey_range": [storey_range],
        "flat_model": [flat_model],
        "floor_area_sqm": [floor_area_sqm],
        "remaining_lease": [remaining_lease],
        "transact_year": [transact_year]
    })

    # Make prediction
    prediction = model.predict(input_data)

    predicted_price = prediction[0]


    # ======================================
    # Display prediction
    # ======================================

    st.success("Prediction completed successfully!")

    st.metric(
        label="Estimated HDB Resale Price",
        value=f"${predicted_price:,.0f}"
    )


    # ======================================
    # Show input summary
    # ======================================

    st.subheader("Prediction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Flat Type:** {flat_type}")
        st.write(f"**Town:** {town}")
        st.write(f"**Storey:** {storey_range}")
        st.write(f"**Flat Model:** {flat_model}")

    with col2:
        st.write(f"**Floor Area:** {floor_area_sqm:.0f} sqm")
        st.write(f"**Remaining Lease:** {remaining_lease:.0f} years")
        st.write(f"**Transaction Year:** {transact_year}")