
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List

app = FastAPI(title="Lung Cancer Prediction API")

# Load the trained model
model = joblib.load("lung-cancer-gradint.joblib")

# Define input data model
class LungCancerInput(BaseModel):
    GENDER: str
    AGE: int
    SMOKING: int
    YELLOW_FINGERS: int
    ANXIETY: int
    PEER_PRESSURE: int
    CHRONIC_DISEASE: int
    FATIGUE: int
    ALLERGY: int
    WHEEZING: int
    ALCOHOL_CONSUMING: int
    COUGHING: int
    SHORTNESS_OF_BREATH: int
    SWALLOWING_DIFFICULTY: int
    CHEST_PAIN: int

# Define response model
class PredictionResponse(BaseModel):
    prediction: str
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: LungCancerInput):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Preprocess the input data
        # Convert GENDER to match model's encoding (M=1, anything else=0)
        input_data['M'] = input_data['GENDER'].map({'M': 1, 'F': 0})
        input_data = input_data.drop(columns=['GENDER'])  # Drop original GENDER column
        
        # Define feature names as expected by the model
        model_feature_names = [
            'M', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
            'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 
            'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
            'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
        
        # Rename input columns to match model feature names
        input_feature_names = [
            'M', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
            'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 
            'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 
            'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
        ]
        rename_dict = dict(zip(input_feature_names, model_feature_names))
        input_data = input_data.rename(columns=rename_dict)
        
        # Reorder columns to match training data
        input_data = input_data[model_feature_names]
        
        # Validate binary features (must be 1 or 2)
        binary_features = model_feature_names[2:]  # Exclude M and AGE
        for feature in binary_features:
            if input_data[feature].iloc[0] not in [1, 2]:
                raise HTTPException(status_code=400, detail=f"Invalid value for {feature}. Must be 1 or 2.")
        
        # Debugging: Print input columns
        print("Input data columns:", input_data.columns.tolist())
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of positive class
        
        # Convert prediction to string
        prediction_str = "YES" if prediction == 1 else "NO"
        
        return PredictionResponse(
            prediction=prediction_str,
            probability=float(probability)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Lung Cancer Prediction API. Use /predict endpoint for predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)