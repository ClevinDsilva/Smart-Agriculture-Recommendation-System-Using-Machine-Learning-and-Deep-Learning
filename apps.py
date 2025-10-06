import os
import re
import sqlite3
import threading
import subprocess
import tempfile
import base64
import json
import uuid
import random
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import secrets

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    jsonify,
    url_for,
    flash,
    session,
)
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from gtts import gTTS
import speech_recognition as sr

import google.generativeai as genai

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: your frontend helpers (ensure they exist)
from frontend import load_model, predict_disease, preprocess_image, load_selected_model

# Load environment
load_dotenv()

# Initialize Gemini AI
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("No Gemini API key found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    logger.info("Gemini AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    gemini_model = None

# Remove this line: gemini_client = genai.Client()

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

password_reset_tokens = {}

email_verification_codes = {}

# Add this right after your environment variables are loaded
print("=== CURRENT EMAIL CONFIGURATION ===")
print(f"EMAIL_ADDRESS: {EMAIL_ADDRESS}")
print(f"EMAIL_PASSWORD: {'*' * len(EMAIL_PASSWORD) if EMAIL_PASSWORD else 'NOT SET'}")
print(f"Password length: {len(EMAIL_PASSWORD) if EMAIL_PASSWORD else 0}")
print(f"SMTP_SERVER: {SMTP_SERVER}")
print(f"SMTP_PORT: {SMTP_PORT}")
print("===================================")

def get_gemini_response(user_input):
    if not gemini_model:
        return "Chat service is currently unavailable. Please try again later."
    
    try:
        # Start a new chat session with empty history for each request
        chat_session = gemini_model.start_chat(history=[])
        response = chat_session.send_message(user_input)
        return response.text
    except Exception as e:
        logger.error(f"Error in Gemini API: {str(e)}")
        return "Sorry, I encountered an error processing your request. Please try again."

# Flask app
app = Flask(__name__)
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
if not FLASK_SECRET_KEY:
    FLASK_SECRET_KEY = 'fallback-secret-key-for-development-only'  # For development only
    logger.warning("Using fallback secret key - not suitable for production")
app.secret_key = FLASK_SECRET_KEY

DB_NAME = 'agri.db'

# Use the GenerativeModel directly instead of Client
gemini_client = genai.GenerativeModel("gemini-2.0-flash")

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

# Simple in-memory voice cache (voice_id -> base64 audio). For production, replace with persistent store.
voice_cache = {}

def init_sqlite_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            mobilenumber TEXT,
            email TEXT UNIQUE,
            username TEXT UNIQUE,
            password TEXT,
            email_verified BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if email_verified column exists, if not add it
    try:
        cursor.execute("SELECT email_verified FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, so add it
        cursor.execute('ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE')
        print("Added email_verified column to users table")
    
    conn.commit()
    conn.close()

# Initialize the database
init_sqlite_db()

# Disease config
DATA_DIR = r"E:\plant detection\new"
CLASS_NAMES = sorted(os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else []
MODEL_OPTIONS = {
    "VGG16": "plant_disease_vgg16_e10.keras",
    "VGG19": "plant_disease_vgg19_e10.keras"
}

# Extract crop details
def extract_crop_details(query):
    location_match = re.search(r"(Mangalore|Udupi|Raichur|Gulbarga|Mysuru|Hassan|Kasaragodu)", query, re.IGNORECASE)
    soil_match = re.search(r"(Alluvial|Loam|Laterite|Sandy|Red|Black|Sandy Loam|Clay)", query, re.IGNORECASE)
    area_match = re.search(r"(\d+(\.\d+)?)\s*(acre|acres)?", query, re.IGNORECASE)
    if location_match and soil_match and area_match:
        return location_match.group(1), area_match.group(1), soil_match.group(1)
    return None, None, None

# Caching Gemini responses
@lru_cache(maxsize=256)
def cached_gemini_response(prompt: str, language: str):
    base_prompt = f"""
You are an agricultural expert assistant named AgriBot. Provide detailed, accurate advice about farming, crops,
plant diseases, fertilizers, and weather impacts. Keep responses concise but informative.

User question: {prompt}
"""
    if language != 'en':
        base_prompt = f"Respond in {language}.\n" + base_prompt

    try:
        # Use the gemini_model that's already initialized
        response = gemini_model.generate_content(base_prompt)
        text = getattr(response, "text", None)
        if text:
            return text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
    return None

def generate_gemini_response(user_input, language='en'):
    if not user_input:
        return "Please provide some input."
    normalized = " ".join(user_input.strip().split())
    cached = cached_gemini_response(normalized, language)
    if cached:
        return cached
    return generate_fallback_response(user_input)

def generate_fallback_response(user_input):
    low = user_input.lower()
    responses = {
        'hello': 'Hello! How can I help you with agriculture today?',
        'hi': 'Hi there! What agricultural information do you need?',
        'bye': 'Goodbye! Happy farming!',
        'crop suggestion': 'For crop suggestions, please tell me your location, soil type, and land area.',
        'plant disease': 'For plant disease detection, please upload an image of the affected plant.',
        'fertilizer': 'I can suggest fertilizers based on your soil type and crop.',
        'weather': 'Check local weather forecasts for accurate information.',
        'market price': 'Market prices vary daily. Check your nearest agricultural market.',
        'thank you': "You're welcome! Is there anything else I can help with?"
    }
    for key, val in responses.items():
        if key in low:
            return val
    return "I'm an agriculture chatbot. I can help with crop suggestions, plant diseases, fertilizers, and more. Please ask specific questions."

# TTS (synchronous, used in background)
def text_to_speech(text, language='en'):
    try:
        if not text:
            return None
        tts = gTTS(text=text, lang=language if language != 'en' else 'en', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return base64.b64encode(audio_buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return None

def speech_to_text(audio_file):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    except Exception as e:
        print(f"Error in speech_to_text: {e}")
        return ""

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/enternew')
def new_user():
    return render_template('signup.html')

@app.route('/check_email', methods=['POST'])
def check_email():
    data = request.get_json()
    email = data.get('email', '').strip()
    
    if not email:
        return jsonify({'exists': False})
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        exists = cursor.fetchone() is not None
    
    return jsonify({'exists': exists})

@app.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    if 'pending_email' not in session:
        flash('No email to verify. Please register again.', 'error')
        return redirect(url_for('new_user'))
    
    email = session['pending_email']
    username = session['pending_username']
    
    if request.method == 'POST':
        entered_code = request.form.get('verification_code', '').strip()
        
        if not entered_code:
            flash('Please enter the verification code.', 'error')
            return render_template('verify_email.html', email=email)
        
        # Check if code exists and is valid
        stored_data = email_verification_codes.get(email)
        if not stored_data:
            flash('Verification code expired. Please request a new one.', 'error')
            return render_template('verify_email.html', email=email)
        
        if datetime.now() > stored_data['expiry']:
            del email_verification_codes[email]
            flash('Verification code has expired. Please request a new one.', 'error')
            return render_template('verify_email.html', email=email)
        
        if entered_code == stored_data['code']:
            # Code verified - complete registration
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO users (name, mobilenumber, email, username, password, email_verified)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session['pending_name'],
                        session['pending_phone'],
                        email,
                        username,
                        session['pending_password'],
                        True  # Set email as verified
                    ))
                    conn.commit()
                
                # Clean up session and verification data
                session.pop('pending_email', None)
                session.pop('pending_username', None)
                session.pop('pending_name', None)
                session.pop('pending_phone', None)
                session.pop('pending_password', None)
                del email_verification_codes[email]
                
                flash('Email verified successfully! Registration complete. Please login.', 'success')
                return redirect(url_for('user_login'))
                
            except sqlite3.IntegrityError as e:
                flash('Registration failed. The username or email may already be taken. Please try again.', 'error')
                return redirect(url_for('new_user'))
            except Exception as e:
                logger.error(f"Database error during registration: {str(e)}")
                flash('Registration failed due to a system error. Please try again.', 'error')
                return redirect(url_for('new_user'))
        else:
            flash('Invalid verification code. Please try again.', 'error')
    
    return render_template('verify_email.html', email=email)

@app.route('/resend_verification', methods=['POST'])
def resend_verification():
    if 'pending_email' not in session:
        return jsonify({'success': False, 'message': 'Session expired'})
    
    email = session['pending_email']
    verification_code = generate_verification_code()
    expiry_time = datetime.now() + timedelta(minutes=10)
    
    email_verification_codes[email] = {
        'code': verification_code,
        'expiry': expiry_time
    }
    
    if send_verification_email(email, verification_code):
        return jsonify({'success': True, 'message': 'Verification code sent!'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send verification code'})

def generate_verification_code():
    return str(random.randint(1000, 9999))

def send_verification_email(email, code):
    try:
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            logger.error("Email credentials not configured")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = "AgriGrow - Email Verification Code"
        
        body = f"""
        <html>
            <body>
                <h2>AgriGrow Email Verification</h2>
                <p>Your verification code is: <strong>{code}</strong></p>
                <p>This code is valid for 10 minutes.</p>
                <p>Enter this code on the verification page to complete your registration.</p>
                <br>
                <p>Best regards,<br>AgriGrow Team</p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, email, text)
        server.quit()
        
        logger.info(f"Verification code sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Error sending verification email: {str(e)}")
        return False

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        mobilenumber = request.form.get('phone', '').strip()
        email = request.form.get('email', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not (name and mobilenumber and email and username and password and confirm_password):
            flash('All fields are required.', 'error')
            return redirect(url_for('new_user'))

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('new_user'))

        if 'terms' not in request.form:
            flash('You must agree to the Terms & Conditions.', 'error')
            return redirect(url_for('new_user'))

        # Check if email already exists
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                flash('Email already registered. Please use a different email or login.', 'error')
                return redirect(url_for('new_user'))
            
            # Check if username already exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                flash('Username already exists. Please choose another.', 'error')
                return redirect(url_for('new_user'))

        # Generate verification code
        verification_code = generate_verification_code()
        expiry_time = datetime.now() + timedelta(minutes=10)
        
        email_verification_codes[email] = {
            'code': verification_code,
            'expiry': expiry_time
        }
        
        # Store user data in session for verification
        session['pending_email'] = email
        session['pending_username'] = username
        session['pending_name'] = name
        session['pending_phone'] = mobilenumber
        session['pending_password'] = generate_password_hash(password)
        
        # Send verification email
        if send_verification_email(email, verification_code):
            flash('Verification code sent to your email. Please check your inbox.', 'success')
            return redirect(url_for('verify_email'))
        else:
            flash('Failed to send verification email. Please try again.', 'error')
            return redirect(url_for('new_user'))

    return render_template('signup.html')

@app.route('/userlogin')
def user_login():
    return render_template("login.html")

@app.route('/logindetails', methods=['POST'])
def logindetails():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if row and check_password_hash(row[0], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('user_login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('user_login'))
    return render_template("Details.html", username=session['username'])

@app.route('/launch_streamlit', methods=['POST'])
def launch_streamlit():
    def run_streamlit():
        subprocess.run(["streamlit", "run", "app.py"])
    threading.Thread(target=run_streamlit, daemon=True).start()
    flash("Streamlit app launched. It may take a few seconds to load.")
    return redirect("http://localhost:8501")

@app.route('/wheat')
def wheat():
    return render_template('wheat.html')

@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        return redirect(url_for('user_login'))
    return render_template('chatbot.html', username=session['username'])

@app.route('/chatbot_response', methods=['POST'])
def chatbot_response():
    user_input = request.form.get('user_input', '').strip()
    language = request.form.get('language', 'en').strip()

    response_text = generate_gemini_response(user_input, language)

    # Start TTS in background and give voice_id immediately
    voice_id = str(uuid.uuid4())

    def make_tts():
        audio_b64 = text_to_speech(response_text, language)
        if audio_b64:
            voice_cache[voice_id] = audio_b64

    executor.submit(make_tts)

    return jsonify({
        'response': response_text,
        'voice_id': voice_id
    })

@app.route('/chatbot_voice/<voice_id>', methods=['GET'])
def chatbot_voice(voice_id):
    audio = voice_cache.get(voice_id)
    if not audio:
        return jsonify({'status': 'pending'}), 202
    return jsonify({'voice_output': audio})

@app.route('/process_voice_input', methods=['POST'])
def process_voice_input():
    audio_data = request.files.get('audio_data')
    if not audio_data:
        return jsonify({'error': 'No audio data received'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_data.save(tmp.name)
        try:
            text = speech_to_text(tmp.name)
        finally:
            os.unlink(tmp.name)

    return jsonify({'text': text})

# Generate OTP
def generate_otp():
    return str(random.randint(1000, 9999))

# Send OTP email
def send_otp_email(email, otp):
    try:
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            logger.error("Email credentials not configured")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = "AgriGrow - Password Reset OTP"
        
        body = f"""
        <html>
            <body>
                <h2>AgriGrow Password Reset</h2>
                <p>Your OTP for password reset is: <strong>{otp}</strong></p>
                <p>This OTP is valid for 10 minutes.</p>
                <p>If you didn't request this reset, please ignore this email.</p>
                <br>
                <p>Best regards,<br>AgriGrow Team</p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, email, text)
        server.quit()
        
        logger.info(f"OTP sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

# Password reset routes
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        
        if not username:
            flash('Please enter your username.', 'error')
            return render_template('forgot_password.html')
        
        # Find user by username
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            email = user[0]
            otp = generate_otp()
            expiry_time = datetime.now() + timedelta(minutes=10)
            
            # Store OTP with expiry
            password_reset_tokens[username] = {
                'otp': otp,
                'expiry': expiry_time,
                'email': email
            }
            
            # Send OTP email
            if send_otp_email(email, otp):
                session['reset_username'] = username
                flash('OTP has been sent to your registered email address.', 'success')
                return redirect(url_for('verify_otp'))
            else:
                flash('Failed to send OTP. Please try again later.', 'error')
        else:
            flash('Username not found.', 'error')
    
    return render_template('forgot_password.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if 'reset_username' not in session:
        return redirect(url_for('forgot_password'))
    
    username = session['reset_username']
    
    if request.method == 'POST':
        entered_otp = request.form.get('otp', '').strip()
        
        if not entered_otp:
            flash('Please enter the OTP.', 'error')
            return render_template('verify_otp.html')
        
        token_data = password_reset_tokens.get(username)
        
        if not token_data:
            flash('OTP session expired. Please request a new OTP.', 'error')
            return redirect(url_for('forgot_password'))
        
        if datetime.now() > token_data['expiry']:
            del password_reset_tokens[username]
            flash('OTP has expired. Please request a new OTP.', 'error')
            return redirect(url_for('forgot_password'))
        
        if entered_otp == token_data['otp']:
            session['otp_verified'] = True
            flash('OTP verified successfully. You can now reset your password.', 'success')
            return redirect(url_for('reset_password'))
        else:
            flash('Invalid OTP. Please try again.', 'error')
    
    return render_template('verify_otp.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_username' not in session or not session.get('otp_verified'):
        return redirect(url_for('forgot_password'))
    
    username = session['reset_username']
    
    if request.method == 'POST':
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not new_password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template('reset_password.html')
        
        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html')
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('reset_password.html')
        
        # Update password in database
        hashed_password = generate_password_hash(new_password)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password = ? WHERE username = ?", 
                      (hashed_password, username))
        conn.commit()
        conn.close()
        
        # Clean up session and tokens
        session.pop('reset_username', None)
        session.pop('otp_verified', None)
        password_reset_tokens.pop(username, None)
        
        flash('Password reset successfully. You can now login with your new password.', 'success')
        return redirect(url_for('user_login'))
    
    return render_template('reset_password.html')

if __name__ == '__main__':
    app.run(debug=True)

