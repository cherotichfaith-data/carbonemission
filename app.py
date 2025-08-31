# app.py
import streamlit as st
import joblib
import pandas as pd

# --- Load Model and Encoder ---
@st.cache_resource
def load_model():
    model = joblib.load("co2_model.pkl")
    encoder = joblib.load("encoder.pkl")
    expected_column_order = joblib.load("columns.pkl")
    average_gasoline_emissions = joblib.load("avg_gas.pkl")
    return model, encoder, expected_column_order, average_gasoline_emissions

model, encoder, expected_column_order, average_gasoline_emissions = load_model()

# --- Streamlit UI ---
st.title("CO2Taxi Emissions & Fee Predictor 🚖🌍")
st.write("Predict CO₂ emissions for Nairobi taxi trips and calculate the corresponding fees.")

trip_id = st.number_input("Trip ID", min_value=1)
distance = st.number_input("Distance (km)", min_value=0.1, step=0.1)
vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Minivan"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "EV"])

# --- Prediction Logic ---
def predict_emissions_and_fee(trip_id, distance, vehicle_type, fuel_type):
    df_input = pd.DataFrame([{
        "trip_id": trip_id,
        "distance_km": distance,
        "vehicle_type": vehicle_type,
        "fuel_type": fuel_type
    }])

    # Encode categorical variables
    df_encoded = encoder.transform(df_input[['vehicle_type', 'fuel_type']])
    df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(['vehicle_type', 'fuel_type']))

    # Prepare processed dataframe
    df_processed = pd.DataFrame(0, index=[0], columns=expected_column_order)
    df_processed['trip_id'] = trip_id
    df_processed['distance_km'] = distance
    for col in df_encoded.columns:
        if col in df_processed.columns:
            df_processed[col] = df_encoded[col]

    # Predict emissions
    predicted_emissions = model.predict(df_processed)[0]
    calculated_fee = 5 + predicted_emissions * 2  # base fee + per kg emission cost

    # Recommendation
    recommendation = None
    if fuel_type != "EV":
        savings = average_gasoline_emissions - predicted_emissions
        if savings > 0.1:
            recommendation = f"Choosing an EV could save ~{savings:.2f} kg CO₂."

    return round(predicted_emissions, 4), round(calculated_fee, 2), recommendation

# --- Predict Button ---
if st.button("Predict Emissions & Fee"):
    emissions, fee, recommendation = predict_emissions_and_fee(trip_id, distance, vehicle_type, fuel_type)
    st.success(f"Predicted Emissions: {emissions} kg CO₂")
    st.info(f"Calculated Fee: ${fee}")
    if recommendation:
        st.warning(f"Recommendation: {recommendation}")
