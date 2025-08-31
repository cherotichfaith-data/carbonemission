import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
@st.cache_resource
def load_model():
    model = joblib.load("co2_model.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, encoder

model, encoder = load_model()

st.set_page_config(page_title="Nairobi Taxi CO2 & Fees", page_icon="🚖")
st.title("Nairobi Taxi CO₂ Emissions & Fees Calculator 🚖🌍")

# Input fields
trip_id = st.number_input("Trip ID:", min_value=1, step=1)
distance_km = st.number_input("Distance (km):", min_value=0.0, step=0.1)
vehicle_type = st.selectbox("Vehicle Type:", ["Taxi", "Van", "Bus"])
fuel_type = st.selectbox("Fuel Type:", ["Gasoline", "Diesel", "EV"])

# Predict button
if st.button("Predict Emissions & Fee"):
    # Prepare input dataframe
    df_input = pd.DataFrame([{
        "trip_id": trip_id,
        "distance_km": distance_km,
        "vehicle_type": vehicle_type,
        "fuel_type": fuel_type
    }])

    # Encode categorical columns
    categorical_cols = ["vehicle_type", "fuel_type"]
    df_encoded = pd.DataFrame(
        encoder.transform(df_input[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Merge numeric columns
    df_final = df_encoded.copy()
    df_final["distance_km"] = distance_km

    # Predict emissions
    predicted_emissions = model.predict(df_final)[0]

    # Calculate fee
    base_fee = 5
    emission_cost_per_kg = 2
    calculated_fee = base_fee + emission_cost_per_kg * predicted_emissions

    # Recommendation
    recommendation = None
    if fuel_type != "EV":
        recommendation = f"Choosing an EV could save ~{predicted_emissions:.2f} kg CO₂."

    # Display results
    st.success(f"Predicted Emissions: {predicted_emissions:.2f} kg CO₂")
    st.info(f"Calculated Fee: ${calculated_fee:.2f}")
    if recommendation:
        st.warning(recommendation)
