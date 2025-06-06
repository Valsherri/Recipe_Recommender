efrom fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

# Load model and encoders
model = tf.keras.models.load_model("hybrid_model.h5")
scaler = joblib.load("scaler.pkl")
user_encoder = joblib.load("user_encoder.pkl")
item_encoder = joblib.load("item_encoder.pkl")

# Initialize FastAPI
app = FastAPI()

# Input schema
class RatingRequest(BaseModel):
    user_id: str
    item_id: str
    prep_feature: float  # your continuous feature like avg rating etc

# API endpoint
@app.post("/predict/")
def predict_rating(data: RatingRequest):
    try:
        user_encoded = user_encoder.transform([data.user_id])[0]
        item_encoded = item_encoder.transform([data.item_id])[0]
        prep_scaled = scaler.transform([[data.prep_feature]])[0][0]

        # Model expects inputs as 2D arrays
        prediction = model.predict([
            np.array([[user_encoded]]),
            np.array([[item_encoded]]),
            np.array([[prep_scaled]])
        ], verbose=0)

        return {"predicted_rating": float(prediction[0][0])}
    except Exception as e:
        return {"error": str(e)}
