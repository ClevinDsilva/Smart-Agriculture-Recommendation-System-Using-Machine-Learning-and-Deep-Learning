import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import base64
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import sqlite3

# Initialize TTS variables at the start
gtts_available = False
pyttsx3_available = False
translator_available = False  # We'll only use manual translations

# Try to import gTTS
try:
    from gtts import gTTS
    gtts_available = True
except ImportError:
    gtts_available = False

# Try to import pyttsx3
try:
    import pyttsx3
    pyttsx3_available = True
except ImportError:
    pyttsx3_available = False

# === Email Configuration ===
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# === Configurations ===
IMG_SIZE = (224, 224)
DATA_DIR = r"E:\plant detection\new"  # Your dataset path
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

MODEL_OPTIONS = {
    "VGG16": "plant_disease_vgg16_e10.keras",
    "VGG19": "plant_disease_vgg19_e10.keras"
}

# English precautions
precautions_dict = {
    "early_blight": "Remove infected leaves, apply copper-based fungicide, and avoid overhead watering.",
    "late_blight": "Destroy infected plants, use certified seeds, and apply recommended fungicide promptly.",
    "leaf_mold": "Ensure good air circulation, remove affected foliage, and apply appropriate fungicides.",
    "septoria_leaf_spot": "Avoid wet foliage, remove infected leaves, and use crop rotation.",
    "bacterial_spot": "Use resistant varieties, avoid splashing water, and treat with copper spray.",
    "powdery_mildew": "Increase airflow, use neem oil or sulfur spray, and keep foliage dry.",
    "rust": "Remove infected leaves, water at the base, and apply sulfur or other fungicides.",
}

# Kannada precautions (manually translated)
precautions_dict_kn = {
    "early_blight": "ಸೋಂಕು ಹರಡಿದ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ತಾಮ್ರ-ಆಧಾರಿತ ಫಂಗಿಸೈಡ್ ಅನ್ನು ಅನ್ವಯಿಸಿ ಮತ್ತು ಮೇಲಿನ ನೀರಾವರಿಯನ್ನು ತಪ್ಪಿಸಿ.",
    "late_blight": "ಸೋಂಕು ಹರಡಿದ ಸಸ್ಯಗಳನ್ನು ನಾಶಪಡಿಸಿ, ಪ್ರಮಾಣೀಕೃತ ಬೀಜಗಳನ್ನು ಬಳಸಿ ಮತ್ತು ಶಿಫಾರಸು ಮಾಡಿದ ಫಂಗಿಸೈಡ್ ಅನ್ನು ತಕ್ಷಣ ಅನ್ವಯಿಸಿ.",
    "leaf_mold": "ಉತ್ತಮ ಗಾಳಿ ಸಂಚಾರವನ್ನು ಖಚಿತಪಡಿಸಿ, ಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಸೂಕ್ತವಾದ ಫಂಗಿಸೈಡ್ಗಳನ್ನು ಅನ್ವಯಿಸಿ.",
    "septoria_leaf_spot": "ನೆನೆದ ಎಲೆಗಳನ್ನು ತಪ್ಪಿಸಿ, ಸೋಂಕು ಹರಡಿದ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಬೆಳೆ ತಿರುಗಾಟವನ್ನು ಬಳಸಿ.",
    "bacterial_spot": "ನಿರೋಧಕ ಪ್ರಭೇದಗಳನ್ನು ಬಳಸಿ, ನೀರನ್ನು ಚಿಮ್ಮುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ತಾಮ್ರ ಸ್ಪ್ರೇನೊಂದಿಗೆ ಚಿಕಿತ್ಸೆ ನೀಡಿ.",
    "powdery_mildew": "ಗಾಳಿಯ ಹರಿವನ್ನು ಹೆಚ್ಚಿಸಿ, ನೀಂ ಎಣ್ಣೆ ಅಥವಾ ಸಲ್ಫರ್ ಸ್ಪ್ರೇ ಬಳಸಿ ಮತ್ತು ಎಲೆಗಳನ್ನು ಒಣಗಿರಿಸಿ.",
    "rust": "ಸೋಂಕು ಹರಡಿದ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಬೇಸಿಗೆಯಲ್ಲಿ ನೀರು ಹಾಕಿ ಮತ್ತು ಸಲ್ಫರ್ ಅಥವಾ ಇತರ ಫಂಗಿಸೈಡ್ಗಳನ್ನು ಅನ್ವಯಿಸಿ.",
}

# Kannada translations for UI elements and disease names
translations_kn = {
    "Crop Prediction": "ಬೆಳೆ ಊಹೆ",
    "Plant Leaf Detection": "ಸಸ್ಯ ಎಲೆ ಪತ್ತೆ",
    "Select a location:": "ಸ್ಥಳವನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
    "Enter the Area (in acres)": "ವಿಸ್ತೀರ್ಣವನ್ನು ನಮೂದಿಸಿ (ಎಕರೆಗಳಲ್ಲಿ)",
    "Select soil type:": "ಮಣ್ಣಿನ ಪ್ರಕಾರವನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
    "Submit Crop Prediction": "ಬೆಳೆ ಊಹೆಯನ್ನು ಸಲ್ಲಿಸಿ",
    "Choose Model": "ಮಾದರಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
    "Upload Leaf Image": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
    "Prediction Result": "ಊಹೆ ಫಲಿತಾಂಶ",
    "Predicted Class:": "ಊಹಿಸಿದ ವರ್ಗ:",
    "Confidence:": "ನಂಬಿಕೆ:",
    "The leaf looks healthy! No action needed.": "ಎಲೆ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ! ಯಾವುದೇ ಕ್ರಮ ಅಗತ್ಯವಿಲ್ಲ.",
    "The leaf appears to be affected by a disease.": "ಎಲೆ ರೋಗದಿಂದ ಪೀಡಿತವಾಗಿದೆ ಎಂದು ತೋರುತ್ತದೆ.",
    "Precaution for": "ಇದಕ್ಕೆ ಮುಂಜಾಗ್ರತೆ",
    "General Advice: Remove the infected parts, isolate affected plants, and consult an agricultural expert.": "ಸಾಮಾನ್ಯ ಸಲಹೆ: ಸೋಂಕು ಹರಡಿದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಪೀಡಿತ ಸಸ್ಯಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಿ ಮತ್ತು ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    "Voice output played automatically": "ಧ್ವನಿ ಔಟ್ಪುಟ್ ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಪ್ಲೇ ಆಗಿದೆ",
    "Send Results to Email": "ಫಲಿತಾಂಶಗಳನ್ನು ಇಮೇಲ್‌ಗೆ ಕಳುಹಿಸಿ",
    "Enter your registered email:": "ನಿಮ್ಮ ನೋಂದಾಯಿತ ಇಮೇಲ್ ಅನ್ನು ನಮೂದಿಸಿ:",
    "Results sent to your email successfully!": "ಫಲಿತಾಂಶಗಳು ನಿಮ್ಮ ಇಮೇಲ್‌ಗೆ ಯಶಸ್ವಿಯಾಗಿ ಕಳುಹಿಸಲಾಗಿದೆ!",
    "Failed to send email. Please try again later.": "ಇಮೇಲ್ ಕಳುಹಿಸಲು ವಿಫಲವಾಗಿದೆ. ದಯವಿಟ್ಟು ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",

    # Crop names in Kannada
    "Rice": "ಅಕ್ಕಿ",
    "Wheat": "ಗೋಧಿ",
    "Maize": "ಮೆಕ್ಕೆ ಜೋಳ",
    "Sugarcane": "ಕಬ್ಬು",
    "Cotton": "ಹತ್ತಿ",
    "Groundnut": "ಕಡಲೆಕಾಯಿ",
    "Ragi": "ರಾಗಿ",
    "Sunflower": "ಸೂರ್ಯಕಾಂತಿ",
    "Jowar": "ಜೋಳ",
    "Bengal Gram": "ಕಡಲೆ",
    "Red Gram": "ತೊಗರಿ ಬೇಳೆ",
    "Green Gram": "ಹೆಸರು ಬೇಳೆ",
    "Black Gram": "ಉದ್ದು ಬೇಳೆ",
    
    # Locations in Kannada
    "Mangalore": "ಮಂಗಳೂರು",
    "Udupi": "ಉಡುಪಿ",
    "Raichur": "ರಾಯಚೂರು",
    "Gulbarga": "ಗುಲ್ಬರ್ಗಾ",
    "Mysuru": "ಮೈಸೂರು",
    "Hassan": "ಹಾಸನ",
    "Kasaragodu": "ಕಾಸರಗೋಡು",
    
    # Soil types in Kannada
    "Alluvial": "ಪ್ರವಾಹಿ ಮಣ್ಣು",
    "Loam": "ಎರೆಮಣ್ಣು",
    "Laterite": "ಲ್ಯಾಟರೈಟ್ ಮಣ್ಣು",
    "Sandy": "ಮರಳು ಮಣ್ಣು",
    "Red": "ಕೆಂಪು ಮಣ್ಣು",
    "Black": "ಕಪ್ಪು ಮಣ್ಣು",
    "Sandy Loam": "ಮರಳು ಎರೆಮಣ್ಣು",
    "Clay": "ಜೇಡಿ ಮಣ್ಣು",
    
    # Disease names in Kannada
    "early_blight": "ಮುಂಚಿನ ಬ್ಲೈಟ್",
    "late_blight": "ತಡವಾದ ಬ್ಲೈಟ್",
    "leaf_mold": "ಎಲೆ ಅಚ್ಚು",
    "septoria_leaf_spot": "ಸೆಪ್ಟೋರಿಯಾ ಎಲೆದುರಿತ",
    "bacterial_spot": "ಬ್ಯಾಕ್ಟೀರಿಯಾದ ದುರಿತ",
    "powdery_mildew": "ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ",
    "rust": "ತುಕ್ಕು",
    "healthy": "ಆರೋಗ್ಯಕರ"
}

def get_kannada_crop_name(english_name):
    """Get Kannada translation for crop name"""
    return translations_kn.get(english_name, english_name)

def get_kannada_disease_name(english_name):
    """Get Kannada translation for disease name, removing underscores"""
    if english_name in translations_kn:
        return translations_kn[english_name]
    clean_name = english_name.lower().replace("_", " ")
    for key, value in translations_kn.items():
        if clean_name == key.lower().replace("_", " "):
            return value
    return english_name.replace("_", " ")

def get_english_crop_name(kannada_name):
    """Get English original name from Kannada translation"""
    for eng, kan in translations_kn.items():
        if kan == kannada_name:
            return eng
    return kannada_name

def translate_text(text, dest_language=None):
    """Translate text using manual translations only"""
    if language == "English":
        return text
    return translations_kn.get(text, text)

def send_results_email(email, subject, body, image_file=None):
    """Send email with results, optionally including an image"""
    try:
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            st.error("Email configuration not set properly.")
            return False

        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        if image_file:
            image_file.seek(0)
            img = MIMEImage(image_file.read(), name="leaf_image.jpg")
            msg.attach(img)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def get_user_email(username):
    """Retrieve user's email from the database"""
    try:
        conn = sqlite3.connect('agri.db')
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        st.error(f"Error retrieving user email: {str(e)}")
        return None

# === Streamlit Page Settings ===
st.set_page_config(page_title="AgriSmart App", layout="centered")
st.markdown(
    """
<style>
/* === App Background === */
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    background-size: cover;
    color: white;
}

/* === Button Styling === */
.stButton > button {
    background-color: #007BFF;
    color: white;
    border-radius: 5px;
    padding: 0.5em 1em;
    font-weight: bold;
    border: none;
}

/* === Text Input and Text Area === */
input[type="text"], input[type="number"], textarea {
    background-color: #1e1e1e !important;
    color: white !important;
    border-radius: 5px;
    border: 1px solid #444;
}

/* === Labels and General Text === */
label, .css-1v0mbdj, .css-10trblm, .css-qbe2hs, .css-1cpxqw2, .css-1cpxqw2 span {
    color: white !important;
}

/* === Selectbox Dropdown === */
div[data-baseweb="select"] * {
    background-color: #1e1e1e !important;
    color: white !important;
}

/* === File Uploader Label === */
section[data-testid="stFileUploader"] label {
    color: white !important;
}

/* === Sidebar Text (if used) === */
.css-1d391kg, .css-1v3fvcr {
    color: white !important;
}

/* === Info / Alert Boxes (like st.info) === */
.stAlert {
    color: white !important;
    background-color: #2c3e50 !important;
    border-left: 0.5rem solid #3498db !important;
}

.stAlert div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 AgriSmart: Crop & Plant Disease Detector")

# Language selection
language = st.radio("Select Language / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ", ["English", "ಕನ್ನಡ"])

# === Session state navigation ===
if "selected_page" not in st.session_state:
    st.session_state.selected_page = None

# === Navigation buttons ===
col1, col2 = st.columns(2)
with col1:
    if st.button(translate_text("🌾 Crop Prediction", language)):
        st.session_state.selected_page = "crop"
with col2:
    if st.button(translate_text("🩺 Plant Leaf Detection", language)):
        st.session_state.selected_page = "plant"

# === Load model ===
@st.cache_resource
def load_selected_model(model_name):
    return load_model(MODEL_OPTIONS[model_name])

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict_disease(model, img_array):
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# === Voice Output Functions ===
def text_to_speech(text):
    """Convert text to speech and return audio data"""
    if not text or not text.strip():
        return None
    try:
        clean_text = text.replace("_", " ")
        if gtts_available:
            try:
                lang = 'kn' if language == "ಕನ್ನಡ" else 'en'
                tts = gTTS(text=clean_text, lang=lang, slow=False)
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                return audio_bytes
            except Exception as gtts_error:
                st.warning(f"gTTS error: {str(gtts_error)}. Trying fallback method.")
                if pyttsx3_available:
                    return pyttsx3_text_to_speech(clean_text)
                return None
        elif pyttsx3_available:
            return pyttsx3_text_to_speech(clean_text)
    except Exception as e:
        st.warning(f"Text-to-speech failed: {str(e)}")
        return None
    st.warning("No TTS engine available. Please install gTTS (pip install gTTS) or pyttsx3 (pip install pyttsx3).")
    return None

def pyttsx3_text_to_speech(text):
    """Convert text to speech using pyttsx3"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, 'rb') as f:
            audio_bytes = BytesIO(f.read())
        os.unlink(tmp_path)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"pyttsx3 error: {str(e)}")
        return None

def autoplay_audio(audio_bytes):
    """Create HTML audio player with autoplay"""
    if audio_bytes is None:
        return
    try:
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        audio_tag = f'''
        <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
        </audio>
        '''
        st.components.v1.html(audio_tag, height=0)
    except Exception as e:
        st.warning(f"Audio playback error: {str(e)}")

# === Crop Prediction Section ===
if st.session_state.selected_page == "crop":
    st.markdown(f"## {'🌾 Crop Prediction with SVM' if language == 'English' else '🌾 SVM ಜೊತೆ ಬೆಳೆ ಊಹೆ'}")
    df1 = pd.read_csv("E:/plant detection/Data1.csv")

    location_options = ['Select...'] + [
        translations_kn[loc] if language == "ಕನ್ನಡ" else loc 
        for loc in ['Mangalore', 'Udupi', 'Raichur', 'Gulbarga', 'Mysuru', 'Hassan', 'Kasaragodu']
    ]
    soil_options = ['Select...'] + [
        translations_kn[soil] if language == "ಕನ್ನಡ" else soil 
        for soil in ['Alluvial', 'Loam', 'Laterite', 'Sandy', 'Red', 'Black', 'Sandy Loam', 'Clay']
    ]

    place = st.selectbox(translate_text("Select a location:"), location_options)
    area = st.text_input(translate_text("Enter the Area (in acres)"))
    soil = st.selectbox(translate_text("Select soil type:"), soil_options)
    username = st.text_input(translate_text("Enter your registered username:"), key="crop_username")

    if st.button(translate_text("Submit Crop Prediction")):
        if place == 'Select...' or soil == 'Select...':
            st.warning(translate_text("⚠️ Please select a location and soil type."))
        elif area == '':
            st.warning(translate_text("⚠️ Please enter a valid area."))
        elif not username:
            st.warning(translate_text("⚠️ Please enter your registered username."))
        else:
            try:
                area = float(area)
                place_en = get_english_crop_name(place) if language == "ಕನ್ನಡ" else place
                soil_en = get_english_crop_name(soil) if language == "ಕನ್ನಡ" else soil
                df_filtered = df1[(df1['Location'] == place_en) & (df1['Soil type'] == soil_en)]

                if df_filtered.empty:
                    st.warning(translate_text("⚠️ No data available for the selected location and soil type."))
                else:
                    le = LabelEncoder()
                    df_filtered['Location'] = le.fit_transform(df_filtered['Location'])
                    df_filtered['Soil type'] = le.fit_transform(df_filtered['Soil type'])
                    df_filtered['Irrigation'] = le.fit_transform(df_filtered['Irrigation'])
                    df_filtered['Crops'] = le.fit_transform(df_filtered['Crops'])

                    df_filtered['yields/area'] = df_filtered['yeilds'] / df_filtered['Area']
                    df_filtered['price/area'] = df_filtered['price'] / df_filtered['Area']

                    mean_yield_per_area = df_filtered['yields/area'].mean()
                    mean_price_per_area = df_filtered['price/area'].mean()

                    estimated_yield = mean_yield_per_area * area
                    estimated_price = mean_price_per_area * area

                    df_filtered = df_filtered.drop(columns='Year')
                    X = df_filtered.drop(columns='Crops')
                    y = df_filtered['Crops']

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    svm_clf = SVC(kernel='linear', random_state=42)
                    svm_clf.fit(X_train, y_train)
                    svm_pred = svm_clf.predict(X_test)

                    predicted_crop = le.inverse_transform([svm_pred[0]])[0]
                    display_crop = get_kannada_crop_name(predicted_crop) if language == "ಕನ್ನಡ" else predicted_crop

                    if language == "English":
                        result_text = f"The predicted crop is {display_crop}. Estimated yield for {area} acres is {estimated_yield:.2f} quintals. Estimated price is {estimated_price:.2f} rupees."
                        st.success(f"🌱 **Predicted Crop:** {display_crop}")
                        st.info(f"📦 Estimated Yield (for {area} acres): {estimated_yield:.2f} quintals")
                        st.info(f"💰 Estimated Price (for {area} acres): ₹{estimated_price:.2f}")
                    else:
                        result_text = f"ಊಹಿಸಿದ ಬೆಳೆ {display_crop}. {area} ಎಕರೆಗೆ ಅಂದಾಜು ಇಳುವರಿ {estimated_yield:.2f} ಕ್ವಿಂಟಾಲ್. ಅಂದಾಜು ಬೆಲೆ {estimated_price:.2f} ರೂಪಾಯಿ."
                        st.success(f"🌱 **ಊಹಿಸಿದ ಬೆಳೆ:** {display_crop}")
                        st.info(f"📦 {area} ಎಕರೆಗೆ ಅಂದಾಜು ಇಳುವರಿ: {estimated_yield:.2f} ಕ್ವಿಂಟಾಲ್")
                        st.info(f"💰 {area} ಎಕರೆಗೆ ಅಂದಾಜು ಬೆಲೆ: ₹{estimated_price:.2f}")

                    # Voice output
                    audio_bytes = text_to_speech(result_text)
                    if audio_bytes:
                        autoplay_audio(audio_bytes)
                        st.markdown(translate_text("🔊 Voice output played automatically"))

                    # Email results
                    user_email = get_user_email(username)
                    if user_email:
                        email_subject = translate_text("AgriSmart Crop Prediction Results", language)
                        email_body = f"""
                        <html>
                            <body>
                                <h2>{translate_text("Crop Prediction Results", language)}</h2>
                                <p><strong>{translate_text("Location:", language)}</strong> {place}</p>
                                <p><strong>{translate_text("Soil Type:", language)}</strong> {soil}</p>
                                <p><strong>{translate_text("Area:", language)}</strong> {area} acres</p>
                                <p><strong>{translate_text("Predicted Crop:", language)}</strong> {display_crop}</p>
                                <p><strong>{translate_text("Estimated Yield:", language)}</strong> {estimated_yield:.2f} quintals</p>
                                <p><strong>{translate_text("Estimated Price:", language)}</strong> ₹{estimated_price:.2f}</p>
                                <br>
                                <p>Best regards,<br>AgriSmart Team</p>
                            </body>
                        </html>
                        """
                        if send_results_email(user_email, email_subject, email_body):
                            st.success(translate_text("Results sent to your email successfully!"))
                        else:
                            st.error(translate_text("Failed to send email. Please try again later."))
                    else:
                        st.error("Username not found. Please enter a valid registered username.")

            except ValueError:
                st.error(translate_text("❌ Please enter a valid number for the Area."))
            except Exception as e:
                st.error(translate_text("❌ An error occurred while processing the data."))
                st.write(f"Error details: {e}")

# === Plant Disease Detection Section ===
elif st.session_state.selected_page == "plant":
    st.markdown(f"## {'🩺 Plant Leaf Disease Detection' if language == 'English' else '🩺 ಸಸ್ಯ ಎಲೆ ರೋಗ ಪತ್ತೆ'}")

    selected_model_name = st.selectbox(translate_text("Choose Model", language), list(MODEL_OPTIONS.keys()))
    uploaded_file = st.file_uploader(translate_text("Upload Leaf Image", language), type=["jpg", "jpeg", "png"])
    username = st.text_input(translate_text("Enter your registered username:"), key="plant_username")

    if uploaded_file:
        st.image(uploaded_file, caption=translate_text("Uploaded Leaf Image", language), use_container_width=True)

        model = load_selected_model(selected_model_name)
        img_array, img = preprocess_image(uploaded_file)
        predicted_class, confidence = predict_disease(model, img_array)
        display_class = get_kannada_disease_name(predicted_class) if language == "ಕನ್ನಡ" else predicted_class.replace("_", " ")

        st.subheader(translate_text("🔍 Prediction Result", language))
        st.write(f"**{translate_text('Predicted Class:', language)}** {display_class}")
        st.write(f"**{translate_text('Confidence:', language)}** {confidence:.2f}%")

        if "healthy" in predicted_class.lower():
            if language == "English":
                result_text = f"The leaf looks healthy with {confidence:.2f} percent confidence. No action needed."
                st.success("✅ The leaf looks healthy! No action needed.")
            else:
                result_text = f"ಎಲೆ {confidence:.2f} ಶೇಕಡಾ ವಿಶ್ವಾಸದೊಂದಿಗೆ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ. ಯಾವುದೇ ಕ್ರಮ ಅಗತ್ಯವಿಲ್ಲ."
                st.success("✅ ಎಲೆ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ! ಯಾವುದೇ ಕ್ರಮ ಅಗತ್ಯವಿಲ್ಲ.")
            
            email_body = f"""
            <html>
                <body>
                    <h2>{translate_text("Plant Disease Detection Results", language)}</h2>
                    <p><strong>{translate_text("Predicted Class:", language)}</strong> {display_class}</p>
                    <p><strong>{translate_text("Confidence:", language)}</strong> {confidence:.2f}%</p>
                    <p>{translate_text("The leaf looks healthy! No action needed.", language)}</p>
                    <br>
                    <p>Best regards,<br>AgriSmart Team</p>
                </body>
            </html>
            """
        else:
            if language == "English":
                result_text = f"The leaf appears to be affected by {display_class} with {confidence:.2f} percent confidence."
                st.error("⚠️ The leaf appears to be affected by a disease.")
            else:
                result_text = f"ಎಲೆ {display_class} ನಿಂದ {confidence:.2f} ಶೇಕಡಾ ವಿಶ್ವಾಸದೊಂದಿಗೆ ಪೀಡಿತವಾಗಿದೆ ಎಂದು ತೋರುತ್ತದೆ."
                st.error("⚠️ ಎಲೆ ರೋಗದಿಂದ ಪೀಡಿತವಾಗಿದೆ ಎಂದು ತೋರುತ್ತದೆ.")
            
            matched = False
            precaution_text = ""
            for key in precautions_dict:
                if key in predicted_class.lower().replace(" ", "_"):
                    st.markdown(f"### 🩺 {translate_text('Precaution for', language)} *{display_class}*")
                    if language == "English":
                        precaution_text = precautions_dict[key]
                        st.warning(precaution_text)
                        result_text += f" Recommended precautions: {precaution_text}"
                    else:
                        precaution_text = precautions_dict_kn.get(key, precautions_dict[key])
                        st.warning(precaution_text)
                        result_text += f" ಶಿಫಾರಸು ಮಾಡಿದ ಮುಂಜಾಗ್ರತೆಗಳು: {precaution_text}"
                    matched = True
                    break
            
            if not matched:
                if language == "English":
                    general_advice = "General Advice: Remove the infected parts, isolate affected plants, and consult an agricultural expert."
                    st.info(f"🔎 {general_advice}")
                    precaution_text = general_advice
                    result_text += f" {general_advice}"
                else:
                    general_advice = "ಸಾಮಾನ್ಯ ಸಲಹೆ: ಸೋಂಕು ಹರಡಿದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಪೀಡಿತ ಸಸ್ಯಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಿ ಮತ್ತು ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
                    st.info(f"🔎 {general_advice}")
                    precaution_text = general_advice
                    result_text += f" {general_advice}"

            email_body = f"""
            <html>
                <body>
                    <h2>{translate_text("Plant Disease Detection Results", language)}</h2>
                    <p><strong>{translate_text("Predicted Class:", language)}</strong> {display_class}</p>
                    <p><strong>{translate_text("Confidence:", language)}</strong> {confidence:.2f}%</p>
                    <p>{translate_text("The leaf appears to be affected by a disease.", language)}</p>
                    <p><strong>{translate_text("Precaution for", language)} {display_class}:</strong> {precaution_text}</p>
                    <br>
                    <p>Best regards,<br>AgriSmart Team</p>
                </body>
            </html>
            """

        # Voice output
        audio_bytes = text_to_speech(result_text)
        if audio_bytes:
            autoplay_audio(audio_bytes)
            st.markdown(translate_text("🔊 Voice output played automatically"))

        # Email results
        if username:
            user_email = get_user_email(username)
            if user_email:
                email_subject = translate_text("AgriSmart Plant Disease Detection Results", language)
                if send_results_email(user_email, email_subject, email_body, uploaded_file):
                    st.success(translate_text("Results sent to your email successfully!"))
                else:
                    st.error(translate_text("Failed to send email. Please try again later."))
            else:
                st.error("Username not found. Please enter a valid registered username.")
        else:
            st.warning(translate_text("⚠️ Please enter your registered username to send results to your email."))