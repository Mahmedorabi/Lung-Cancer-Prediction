import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import logging
# 2. FastAPI application
app = FastAPI(title="VGG16 Malignant vs Normal Classifier")

# Load model at startup
model = None
try:
    model = load_model('lung-vgg16.keras')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    logging.warning("Run train_and_save_model() to create model")

def preprocess_image(image: Image.Image):
    try:
        # Resize to 224x224
        image = image.resize((224, 224))
        # Convert to array
        image_array = np.array(image)
        # Ensure 3 channels (RGB)
        if image_array.ndim == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[-1] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        # Apply VGG16 preprocessing
        image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
        return image_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        raise

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded. Train or provide vgg16_model.h5")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        image_array = preprocess_image(image)
        
        # Predict
        prediction = model.predict(image_array)[0][0]
        label = "Malignant" if prediction <= 0.5 else "Normal"
        
        return JSONResponse({
            "label": label
        })
    except Exception as e:
        logging.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Uncomment to train model (run once, then comment out)
# if __name__ == "__main__":
#     train_and_save_model()
