
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI(title="tumor Diagnosis API")

model = tf.keras.models.load_model("model (1).keras")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preprocessed = preprocess_image(image_bytes)

    # For multi-class classification
    pred = model.predict(preprocessed)
    predicted_index = np.argmax(pred)
    class_labels = ['Viable', 'Non-Viable-Tumor', 'Non-Tumor']
    label = class_labels[predicted_index]
    confidence = float(np.max(pred) * 100)

    return JSONResponse({
        "Diagnosis": label,
        "Confidence": f"{confidence:.2f}%"
    })
