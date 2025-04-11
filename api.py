from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Lung Disease Prediction API", description="API to predict lung disease level based on input features")

# Load the trained model
try:
    model = joblib.load("KNN_model.pkl")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Define the input data schema using Pydantic with corrected field names
class PredictionInput(BaseModel):
    Age: int
    Gender: int
    Air_Pollution: int
    Alcohol_use: int
    Dust_Allergy: int
    OccuPational_Hazards: int
    Genetic_Risk: int
    chronic_Lung_Disease: int
    Balanced_Diet: int
    Obesity: int
    Smoking: int
    Passive_Smoker: int
    Chest_Pain: int
    Coughing_of_Blood: int
    Fatigue: int
    Weight_Loss: int
    Shortness_of_Breath: int
    Wheezing: int
    Swallowing_Difficulty: int
    Clubbing_of_Finger_Nails: int
    Frequent_Cold: int
    Dry_Cough: int
    Snoring: int

    class Config:
        schema_extra = {
            "example": {
                "Age": 33,
                "Gender": 1,
                "Air_Pollution": 2,
                "Alcohol_use": 4,
                "Dust_Allergy": 5,
                "OccuPational_Hazards": 4,
                "Genetic_Risk": 3,
                "chronic_Lung_Disease": 2,
                "Balanced_Diet": 2,
                "Obesity": 4,
                "Smoking": 3,
                "Passive_Smoker": 2,
                "Chest_Pain": 2,
                "Coughing_of_Blood": 4,
                "Fatigue": 3,
                "Weight_Loss": 4,
                "Shortness_of_Breath": 2,
                "Wheezing": 2,
                "Swallowing_Difficulty": 3,
                "Clubbing_of_Finger_Nails": 1,
                "Frequent_Cold": 2,
                "Dry_Cough": 3,
                "Snoring": 4
            }
        }


# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to a dictionary and then to a DataFrame
        data_dict = input_data.dict()
        input_df = pd.DataFrame([data_dict])

        # Define the feature order to match the model's expectations
        feature_order = [
            'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
            'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
            'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
            'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
            'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
            'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
        ]

        # Rename the DataFrame columns to match the model's expected names
        input_df = input_df.rename(columns={
            'Air_Pollution': 'Air Pollution',
            'Alcohol_use': 'Alcohol use',
            'Dust_Allergy': 'Dust Allergy',
            'OccuPational_Hazards': 'OccuPational Hazards',
            'Genetic_Risk': 'Genetic Risk',
            'chronic_Lung_Disease': 'chronic Lung Disease',
            'Balanced_Diet': 'Balanced Diet',
            'Passive_Smoker': 'Passive Smoker',
            'Chest_Pain': 'Chest Pain',
            'Coughing_of_Blood': 'Coughing of Blood',
            'Weight_Loss': 'Weight Loss',
            'Shortness_of_Breath': 'Shortness of Breath',
            'Swallowing_Difficulty': 'Swallowing Difficulty',
            'Clubbing_of_Finger_Nails': 'Clubbing of Finger Nails',
            'Frequent_Cold': 'Frequent Cold',
            'Dry_Cough': 'Dry Cough'
        })

        # Reorder the DataFrame columns to match the model's feature order
        input_df = input_df[feature_order]

        # Make prediction
        prediction = model.predict(input_df)[0]

        if prediction == 0:
            prediction = "Low Risk"
        elif prediction == 1:
            prediction = "Medium Risk"
        elif prediction == 2:
            prediction = "High Risk"

        try:
            probabilities = model.predict_proba(input_df)[0]
            probabilities = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        except AttributeError:
            probabilities = None  

        # Return the prediction
        return {
            "prediction": str(prediction),
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")