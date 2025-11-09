from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DB Initialization
def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS uploads
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     emotion TEXT,
                     upload_time TEXT)''')
    conn.close()

init_db()

# Emotion labels (FER2013)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return render_template('index.html', emotion="No image uploaded.")
    
    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load model lazily (to save memory on Render Free tier)
    MODEL_PATH = "emotion_detection_model.h5"
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        return render_template('index.html', emotion=f"Error loading model: {str(e)}")

    # Preprocess image safely
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return render_template('index.html', emotion="Invalid image file.")
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    # Predict emotion
    prediction = model.predict(img, verbose=0)
    emotion = EMOTIONS[np.argmax(prediction)]

    # Save record in DB
    try:
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO uploads (filename, emotion, upload_time) VALUES (?, ?, ?)",
                    (file.filename, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
    except Exception as e:
        return render_template('index.html', emotion=f"Database error: {str(e)}")

    return render_template('index.html', filename=file.filename, emotion=emotion)

@app.route('/admin')
def admin():
    try:
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("SELECT filename, emotion, upload_time FROM uploads ORDER BY id DESC")
        records = cur.fetchall()
        conn.close()
    except Exception as e:
        records = []
    return render_template('admin.html', records=records)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Use threaded=True to handle multiple requests safely
    app.run(host="0.0.0.0", port=port, threaded=True)
