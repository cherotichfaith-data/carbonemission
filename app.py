from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

# Load trained model + encoder
model = joblib.load("co2_model.pkl")
encoder = joblib.load("encoder.pkl")

# (Temporary) Load saved training columns and avg emissions
try:
    expected_column_order = joblib.load("columns.pkl")  # Save this during training
    average_gasoline_emissions = joblib.load("avg_gas.pkl")  # Save this during training
except:
    expected_column_order = None
    average_gasoline_emissions = None

app = FastAPI(title="Nairobi Taxi Emissions and Fees API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to CO2Taxi API 🚖🌍"}

class TripInput(BaseModel):
    trip_id: int
    distance_km: float
    vehicle_type: str
    fuel_type: str

def calculate_fees(emissions, base_fee=5, emission_cost_per_kg=2):
    return base_fee + emissions * emission_cost_per_kg

@app.post("/predict_emissions_and_fees")
def predict_emissions_and_fees(trip: TripInput):
    df_input = pd.DataFrame([trip.dict()])

    # Encode categoricals
    categorical_cols = ['vehicle_type', 'fuel_type']
    df_encoded_input = encoder.transform(df_input[categorical_cols])
    df_encoded_input = pd.DataFrame(
        df_encoded_input,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Build processed dataframe
    df_processed_input = pd.DataFrame(0, index=[0], columns=expected_column_order or [])
    df_processed_input['trip_id'] = trip.trip_id
    df_processed_input['distance_km'] = trip.distance_km
    for col in df_encoded_input.columns:
        if col in df_processed_input.columns:
            df_processed_input[col] = df_encoded_input[col]

    # Prediction
    predicted_emissions = model.predict(df_processed_input)[0]
    calculated_fee = calculate_fees(predicted_emissions)

    recommendation = None
    if trip.fuel_type != "EV" and average_gasoline_emissions:
        estimated_savings = average_gasoline_emissions - predicted_emissions
        if estimated_savings > 0.1:
            recommendation = f"Choosing an EV could save ~{estimated_savings:.2f} kg CO₂."

    return {
        "predicted_emissions_kg": round(float(predicted_emissions), 4),
        "calculated_fee": round(float(calculated_fee), 2),
        "recommendation": recommendation
    }
