
import numpy as np
import tensorflow as tf
import streamlit as st

from pathlib import Path
from tensorflow.keras.preprocessing import image
from class_names import CLASS_NAMES


# =====================================================
# Paths and Configuration
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model" / "best_model.keras"

IMAGE_SIZE = (224, 224)


# =====================================================
# Load Trained Model (Cached)
# =====================================================

@st.cache_resource(show_spinner="Loading AgroVision AI model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            "Train the model first and save it as best_model.keras."
        )

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =====================================================
# Prediction Function
# =====================================================

def predict_image(image_path):

    # Load and resize image
    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    # Convert image to NumPy array
    img_array = image.img_to_array(img)

    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict without verbose console output
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]

    # Validate model output against class names
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} predictions, "
            f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes. "
            "Ensure the model and class_names.py use the same class order."
        )

    # Predicted class
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]

    # Extract plant and disease
    parts = predicted_class.split("___", maxsplit=1)

    plant = parts[0].replace("_", " ")
    disease = (
        parts[1].replace("_", " ")
        if len(parts) > 1
        else "Healthy"
    )

    confidence = float(probabilities[predicted_index] * 100)

    # =================================================
    # Top 3 Predictions
    # =================================================

    top3_indices = np.argsort(probabilities)[-3:][::-1]

    top3 = []

    for idx in top3_indices:
        class_name = CLASS_NAMES[int(idx)]
        class_parts = class_name.split("___", maxsplit=1)

        plant_name = class_parts[0].replace("_", " ")
        disease_name = (
            class_parts[1].replace("_", " ")
            if len(class_parts) > 1
            else "Healthy"
        )

        top3.append({
            "plant": plant_name,
            "disease": disease_name,
            "confidence": round(
                float(probabilities[idx] * 100), 2
            )
        })

    return {
        "plant": plant,
        "disease": disease,
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "top3": top3
    }


# =====================================================
# Test Prediction (Optional)
# =====================================================

if __name__ == "__main__":

    image_path = input("Enter image path: ").strip().strip('"')

    if not Path(image_path).is_file():
        print(f"Image not found: {image_path}")
    else:
        result = predict_image(image_path)

        print("\n" + "=" * 50)
        print("AgroVision AI - Prediction Result")
        print("=" * 50)

        print(f"\nPlant      : {result['plant']}")
        print(f"Disease    : {result['disease']}")
        print(f"Confidence : {result['confidence']}%")

        print("\nTop 3 Predictions")
        print("-" * 50)

        for i, item in enumerate(result["top3"], start=1):
            print(
                f"{i}. {item['plant']} - {item['disease']} "
                f"({item['confidence']}%)"
            )
