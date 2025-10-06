import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# === Configurations ===
IMG_SIZE = (224, 224)
DATA_DIR = r"E:\plant detection\new"  # Used to get class names
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

# === Model Paths ===
MODEL_OPTIONS = {
    "VGG16": "plant_disease_vgg16_e10.keras",
    "VGG19": "plant_disease_vgg19_e10.keras"
}

# === Sample Precautions (extend as needed) ===
precautions_dict = {
    "early_blight": "Remove infected leaves, apply copper-based fungicide, and avoid overhead watering.",
    "late_blight": "Destroy infected plants, use certified seeds, and apply recommended fungicide promptly.",
    "leaf_mold": "Ensure good air circulation, remove affected foliage, and apply appropriate fungicides.",
    "septoria_leaf_spot": "Avoid wet foliage, remove infected leaves, and use crop rotation.",
    "bacterial_spot": "Use resistant varieties, avoid splashing water, and treat with copper spray.",
    "powdery_mildew": "Increase airflow, use neem oil or sulfur spray, and keep foliage dry.",
    "rust": "Remove infected leaves, water at the base, and apply sulfur or other fungicides.",
    # Add more based on your class names
}

# === Load selected model ===
@st.cache_resource
def load_selected_model(model_name):
    model_path = MODEL_OPTIONS[model_name]
    model = load_model(model_path)
    return model

# === Preprocess image ===
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

# === Prediction function ===
def predict_disease(model, img_array):
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# === Streamlit UI ===
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a plant leaf image and select the model to detect if it's healthy or affected by disease.")

# === Model selection dropdown ===
selected_model_name = st.selectbox("Choose Model", list(MODEL_OPTIONS.keys()))

# === File uploader ===
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    # Load model and preprocess
    model = load_selected_model(selected_model_name)
    img_array, display_img = preprocess_image(uploaded_file)

    # Predict
    predicted_class, confidence = predict_disease(model, img_array)

    # Show result
    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Determine if healthy
    if "healthy" in predicted_class.lower():
        st.success("✅ The leaf looks healthy! No action needed.")
    else:
        st.error("⚠️ The leaf appears to be affected by a disease.")
        # Try to match the class to a known precaution
        matched = False
        for key in precautions_dict:
            if key in predicted_class.lower().replace(" ", "_"):
                st.markdown(f"### 🩺 Precaution for *{predicted_class}*")
                st.warning(precautions_dict[key])
                matched = True
                break
        if not matched:
            st.info("🔎 General Advice: Remove the infected parts, isolate affected plants, and consult an agricultural expert.")
