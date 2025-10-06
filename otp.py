# test_gmail.py
import os
import smtplib
from dotenv import load_dotenv

load_dotenv()

def test_gmail_connection():
    try:
        email = os.getenv("EMAIL_ADDRESS")
        password = os.getenv("EMAIL_PASSWORD")
        
        print("Testing Gmail connection...")
        print(f"Email: {email}")
        print(f"Password length: {len(password) if password else 0}")
        
        # Test connection
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(email, password)
        print("✅ Gmail login successful!")
        server.quit()
        return True
        
    except Exception as e:
        print(f"❌ Gmail test failed: {e}")
        return False

if __name__ == "__main__":
    test_gmail_connection()