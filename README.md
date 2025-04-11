# Lung Disease Prediction API
This FastAPI application deploys a machine learning model to predict lung disease risk levels ("Low", "Medium", or "High") based on 23 patient features, such as age, gender, air pollution exposure, and symptoms. The model is trained on a dataset of patient records and exposed via a RESTful API.

## Features
- Predicts lung disease risk using a pre-trained scikit-learn model.
- Accepts JSON input with 23 integer features (e.g., Age, Air_Pollution).
- Returns the predicted risk level and class probabilities (if supported).
- Provides interactive API documentation via Swagger UI.
- Supports testing with Postman, curl, or other HTTP clients.
## Prerequisites
- Python 3.8+: Install from `python.org`.
- pip: Python package manager (included with Python).
- Trained Model: A `KNN_model.pkl` file (scikit-learn model saved with `joblib`) in the project root.
- Postman (optional): For testing (postman.com).
- Git (optional): For cloning the repository.
## Installation
1. **Clone the Repository** (if applicable):
```bash
git clone <repository-url>
cd lung-disease-prediction-api
```
2. **Create a Virtual Environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install Dependencies**: Install required packages:
```bash

pip install -r requirements.txt
```

## Running the API
1. **Start the FastAPI Server**: Run the application using Uvicorn:
```bash
uvicorn api:app --reload
```
- `api:app`: Refers to the FastAPI app in main.py.
- `--reload`: Enables auto-reload for development (omit in production).
2. **Verify the API**:
- The API runs at `http://127.0.0.1:8000`.
- Open `http://127.0.0.1:8000/docs` to access Swagger UI.
## API Endpoints

- **POST /predict**: Predicts lung disease risk.
    - Request Body: JSON with 23 integer fields.
    - Response: JSON with `prediction` and `probabilities`.

## Testing with Postman
To test the `/predict` endpoint:

1. **Open Postman**: Create a new HTTP request.
2. **Configure the Request**:
- **Method**: `POST`
- **URL**: `http://127.0.0.1:8000/predict`
- **Headers**:
    - Key: `Content-Type`
    - Value: `application/json`
- **Body**:
    - Select **raw** and set to **JSON**.
    - Paste the example JSON (see ).
3. **Send the Request**: Click **Send** to view the response.

## Testing with Swagger UI
1. **Access Swagger UI**: Open `http://127.0.0.1:8000/docs`.
2. **Test /predict**:
    - Expand **POST /predict**.
    - Click **Try it out**.
    - Enter the example JSON in the request body.
    - Click **Execute**.
## Example Input and Output
### Example Input
This JSON represents a high-risk patient (Patient P1000):

```json

{
    "Age": 37,
    "Gender": 1,
    "Air_Pollution": 7,
    "Alcohol_use": 7,
    "Dust_Allergy": 7,
    "OccuPational_Hazards": 7,
    "Genetic_Risk": 6,
    "chronic_Lung_Disease": 7,
    "Balanced_Diet": 7,
    "Obesity": 7,
    "Smoking": 7,
    "Passive_Smoker": 7,
    "Chest_Pain": 7,
    "Coughing_of_Blood": 8,
    "Fatigue": 4,
    "Weight_Loss": 2,
    "Shortness_of_Breath": 3,
    "Wheezing": 1,
    "Swallowing_Difficulty": 4,
    "Clubbing_of_Finger_Nails": 5,
    "Frequent_Cold": 6,
    "Dry_Cough": 7,
    "Snoring": 5
}
```
### Example Output
For a classification model:

```json

{
    "prediction": "High",
    "probabilities": {
        "0": 0.05,
        "1": 0.15,
        "2": 0.80
    }
}
```
- `prediction`: Risk level ("Low", "Medium", "High").
- `probabilities`: Probabilities for each class.
### Input Notes
- **Required Fields**: All 23 fields must be integers.
- **Ranges:**
    - `Age`: Positive integers (e.g., 0–100).
    - `Gender`: 1 or 2.
    - Others: Typically 1–9 (check dataset).
- **Case Sensitivity**: Use exact field names (e.g., Age, not age).
