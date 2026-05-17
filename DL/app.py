from fastapi import FastAPI
import tensorflow as tf
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# 🔥 SAFE MODEL LOADING (fix for your error)
try:
    print("Loading model...")
    model = tf.keras.models.load_model(r"B:\College\C_Codes\College\SEM_8\DL\model.keras", compile=False)
    print("Model loaded successfully!")

except Exception as e:
    print("Error loading full model:", e)
    print("Trying fallback (weights loading)...")

    # 🔁 Fallback: rebuild model manually
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Load weights if available
    try:
        model.load_weights("weights.h5")
        print("Weights loaded successfully!")
    except:
        print("No weights file found. Model may not work correctly.")


# 🏠 Home route
@app.get("/")
def home():
    return {"message": "Model is running!"}


# 🔮 Prediction route
@app.post("/predict")
def predict(data: dict):
    try:
        # Validate input
        if "input" not in data:
            return {"error": "Missing 'input' key in request"}

        input_data = np.array(data["input"])

        # Validate shape
        if input_data.shape != (28, 28):
            return {"error": "Input must be 28x28 array"}

        # Normalize
        input_data = input_data / 255.0
        input_data = input_data.reshape(1, 28, 28)

        # Predict
        prediction = model.predict(input_data)
        result = int(np.argmax(prediction))

        return {
            "prediction": result,
            "confidence": float(np.max(prediction))
        }

    except Exception as e:
        return {"error": str(e)}