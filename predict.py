from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import io
import numpy as np

app = FastAPI()

# Paths to the saved model and scaler
MODEL_PATH = r"D:\hse_hw\ridge_model.pkl"
SCALER_PATH = r"D:\hse_hw\scaler.pkl"

# Load the pre-trained model and scaler
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the base features used for training
FEATURE_COLUMNS = [
    "year", "km_driven", "fuel", "seller_type", "transmission",
    "owner", "mileage", "engine", "max_power", "seats",
    "torque_value", "max_torque_rpm", "brand",
    "power_per_km", "power_to_mileage", "torque_per_km"
]

# Pydantic models
class Item(BaseModel):
    year: int
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: int
    torque_value: float
    max_torque_rpm: float
    brand: str

class Items(BaseModel):
    objects: List[Item]

def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and transform data."""
    # Log transformation for specific columns
    for column in ["km_driven", "mileage", "max_power"]:
        df[column] = np.log1p(df[column])
    
    # Create derived features
    df["power_per_km"] = df["max_power"] / df["km_driven"]
    df["power_to_mileage"] = df["max_power"] * df["mileage"] / df["km_driven"]
    df["torque_per_km"] = df["torque_value"] / df["km_driven"]
    
    # Convert 'year' and 'seats' to categorical
    df["year"] = df["year"].astype("category")
    df["seats"] = df["seats"].astype("category")
    
    return df

def preprocess_item(item: Item) -> pd.DataFrame:
    """Preprocess a single item into the correct DataFrame format."""
    data = {
        "year": [item.year],
        "km_driven": [item.km_driven],
        "fuel": [item.fuel],
        "seller_type": [item.seller_type],
        "transmission": [item.transmission],
        "owner": [item.owner],
        "mileage": [item.mileage],
        "engine": [item.engine],
        "max_power": [item.max_power],
        "seats": [item.seats],
        "torque_value": [item.torque_value],
        "max_torque_rpm": [item.max_torque_rpm],
        "brand": [item.brand],
    }
    df = pd.DataFrame(data)
    
    # Generate new features (power_per_km, power_to_mileage, torque_per_km)
    df = create_new_features(df)
    
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """Predict the price for a single car item."""
    df = preprocess_item(item)
    df_dummies = pd.get_dummies(df).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df_scaled = scaler.transform(df_dummies)
    prediction = model.predict(df_scaled)
    return prediction[0]

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    """Predict prices for multiple car items."""
    dfs = [preprocess_item(item) for item in items.objects]
    df = pd.concat(dfs, ignore_index=True)
    df_dummies = pd.get_dummies(df).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df_scaled = scaler.transform(df_dummies)
    predictions = model.predict(df_scaled)
    return predictions.tolist()

@app.post("/predict_csv")
async def predict_csv(file: UploadFile):
    """Predict prices for cars from a CSV file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    if not set(FEATURE_COLUMNS).issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {FEATURE_COLUMNS}")
    
    # Generate new features for the CSV data
    df = create_new_features(df)
    
    # Generate dummy variables for categorical columns and align with FEATURE_COLUMNS
    df_dummies = pd.get_dummies(df).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df_scaled = scaler.transform(df_dummies)
    predictions = model.predict(df_scaled)
    df["predicted_price"] = predictions

    output = io.StringIO()
    df.to_csv(output, index=False)
    return {"file": output.getvalue()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8001)
