import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import filedialog, Tk

# === Configuration ===
model_path = "plant_disease_vgg16_e10.keras"
data_dir = r"E:\plant detection\new"  # folder used in training
img_size = (224, 224)

# === Load the trained model ===
model = load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

# === Get class labels from folder names ===
class_names = sorted(os.listdir(data_dir))
print(f"✅ Detected class names: {class_names}")

# === Function to preprocess and predict ===
def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    print(f"\n✅ Predicted Class: {predicted_class}")
    print(f"🧠 Confidence: {confidence:.2f}%")

# === File dialog to upload image ===
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select an Image to Predict",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if file_path:
    print(f"\n📂 Selected file: {file_path}")
    predict_single_image(file_path)
else:
    print("❌ No file selected.")
