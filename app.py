import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Load the trained model
model = joblib.load("co2_model.pkl")

# Define function to calculate fees
def calculate_fees(emissions, base_fee=5, emission_cost_per_kg=2):
    return base_fee + emissions * emission_cost_per_kg

# Optional: average gasoline emissions for recommendation
average_gasoline_emissions = 15.0  # replace with your actual average

st.title("🚖 Nairobi Taxi CO₂ Emissions & Fee Predictor")

# User input
trip_id = st.number_input("Trip ID", min_value=1, value=1)
distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0)
vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "MiniBus", "Van"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "EV"])

# Prepare dataframe
df_input = pd.DataFrame([{
    "distance_km": distance_km,
    "vehicle_type": vehicle_type,
    "fuel_type": fuel_type
}])

# One-hot encode input on the fly
df_encoded = pd.get_dummies(df_input)

# Align with training columns
# Replace these with the actual columns used in your model training
expected_columns = [
    'distance_km', 'vehicle_type_Sedan', 'vehicle_type_SUV', 'vehicle_type_MiniBus', 
    'vehicle_type_Van', 'fuel_type_Petrol', 'fuel_type_Diesel', 
    'fuel_type_Hybrid', 'fuel_type_EV'
]

for col in expected_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[expected_columns]

# Predict button
if st.button("Predict Emissions & Fee"):
    predicted_emissions = model.predict(df_encoded)[0]
    calculated_fee = calculate_fees(predicted_emissions)
    
    recommendation = None
    if fuel_type != "EV":
        estimated_savings = average_gasoline_emissions - predicted_emissions
        if estimated_savings > 0.1:
            recommendation = f"Choosing an EV could save ~{estimated_savings:.2f} kg CO₂."

    st.success(f"Predicted CO₂ Emissions: {predicted_emissions:.2f} kg")
    st.info(f"Calculated Fee: ${calculated_fee:.2f}")
    if recommendation:
        st.warning(recommendation)
