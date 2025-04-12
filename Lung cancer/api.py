from fastapi import FastAPI, File, UploadFile, HTTPException
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lung Cancer Classification API",
    description="Classify chest CT scan images into Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, or Normal.",
    version="1.0.0"
)

MODEL_PATH = "model_VGG16.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

CLASS_NAMES = ["Adenocarcinoma", "Large cell carcinoma", "Squamous cell carcinoma", "Normal"]

def preprocess_image(image: Image.Image) -> np.ndarray:
    logger.info("Preprocessing image")
    try:
        image = image.resize((224, 224))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        logger.info("Image preprocessed successfully")
        return image_array
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Lung Cancer Classification API is running. Use /predict to classify images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if file is provided and has a valid content type
    if not file or not file.filename:
        logger.warning("No file uploaded")
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {content_type}")
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        if not contents:
            logger.warning("Empty file uploaded")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        logger.info("Image file read")
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))
            }
        }
        logger.info(f"Prediction: {predicted_class} with confidence {confidence}")
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")