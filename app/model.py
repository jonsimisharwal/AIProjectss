import joblib
import numpy as np

# 🔁 Model load (yeh ek hi baar chalega FastAPI start hote hi)
model = joblib.load("models/behavior_model.pkl")

# 📦 Prediction function
def make_prediction(data):
    input_data = np.array([[
        data.pages_visited,
        data.time_spent,
        data.days_since_signup,
        int(data.is_logged_in)  # boolean to int
    ]])

    prediction = model.predict(input_data)[0]
    confidence = round(model.predict_proba(input_data)[0][prediction], 2)

    return {
        "prediction": str(prediction),  # "0" or "1"
        "confidence": confidence        # 0.85, 0.60 etc.
    }
