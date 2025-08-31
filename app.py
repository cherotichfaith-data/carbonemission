# app.py
import streamlit as st
import pandas as pd
import joblib

# --- Load model & encoder ---
model = joblib.load("co2_model.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Nairobi Taxi Emissions", page_icon="🚖")

st.title("🚖 Nairobi Taxi CO₂ Emissions & Fees")
st.markdown("Enter your trip details to predict CO₂ emissions and calculate fees.")

# --- Input form ---
with st.form("trip_form"):
    trip_id = st.number_input("Trip ID", min_value=1, value=1)
    distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1)
    vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "MiniBus", "Motorbike"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "EV"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input dataframe
    df_input = pd.DataFrame([{
        "trip_id": trip_id,
        "distance_km": distance_km,
        "vehicle_type": vehicle_type,
        "fuel_type": fuel_type
    }])

    # Encode categorical columns
    categorical_cols = ["vehicle_type", "fuel_type"]
    df_encoded = encoder.transform(df_input[categorical_cols])
    df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine numeric + encoded categorical columns
    df_model_input = df_input.copy()
    for col in df_encoded.columns:
        df_model_input[col] = df_encoded[col]

    # Predict emissions
    predicted_emissions = model.predict(df_model_input)[0]

    # Calculate fee
    base_fee = 5
    emission_cost_per_kg = 2
    calculated_fee = base_fee + predicted_emissions * emission_cost_per_kg

    # Display results
    st.success(f"Predicted CO₂ Emissions: {predicted_emissions:.2f} kg")
    st.info(f"Calculated Trip Fee: ${calculated_fee:.2f}")
