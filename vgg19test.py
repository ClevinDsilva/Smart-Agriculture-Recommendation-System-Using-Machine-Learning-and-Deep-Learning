import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import filedialog
from tkinter import Tk

# === Configurations ===
model_path = "plant_disease_vgg19_e10.keras"
data_dir = r"E:\plant detection\new"  # same as your training folder
img_size = (224, 224)

# === Load model ===
model = load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

# === Load class labels ===
class_names = sorted(os.listdir(data_dir))
print(f"✅ Class labels: {class_names}")

# === Ask user to upload an image ===
root = Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if file_path:
    print(f"📂 Selected file: {file_path}")

    # === Preprocess image ===
    img = image.load_img(file_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # === Predict ===
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100

    print(f"\n✅ Predicted Class: {predicted_class}")
    print(f"🧠 Confidence: {confidence:.2f}%")
else:
    print("❌ No file selected.")
