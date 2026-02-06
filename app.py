from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import keras
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware

# -----------------------
# Sanity check
# -----------------------
assert os.path.exists("model/saved_model.pb"), "SavedModel not found"

# -----------------------
# Load SavedModel (Keras 3 SAFE)
# -----------------------
model = keras.layers.TFSMLayer(
    "model",
    call_endpoint="serving_default"
)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="DNN Inference API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TEMP: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Request schema
# -----------------------
class DNNInput(BaseModel):
    features: list[float]

EXPECTED_FEATURES = 52

# -----------------------
# Health check
# -----------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "keras_version": keras.__version__,
        "tf_version": tf.__version__
    }

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict(data: DNNInput):

    # 🔒 Safety guard: feature count
    if len(data.features) != EXPECTED_FEATURES:
        return {
            "error": f"Invalid feature count. Expected {EXPECTED_FEATURES}, got {len(data.features)}"
        }
    

    X = np.array(data.features, dtype=np.float32).reshape(1, -1)

    # Forward pass
    output = model(X)

    # Handle binary vs multi-class safely
    if isinstance(output, dict):
        output = list(output.values())[0]

    output = output.numpy()


    if output.shape[-1] == 1:
        confidence = float(output[0][0])
        prediction = int(confidence >= 0.5)
    else:
        prediction = int(np.argmax(output[0]))
        confidence = float(np.max(output[0]))

    label = "Attack" if prediction == 1 else "Normal"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence
    }
