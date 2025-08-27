#!/usr/bin/env python3
"""
Web server for MELD Emotion Recognition
Upload audio files and get emotion predictions with feature analysis
"""

import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import tempfile
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

class SimpleEmotionNet(nn.Module):
    """Simplified neural network (same as robust trainer)"""

    def __init__(self, input_size=50, num_classes=7):
        super(SimpleEmotionNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Global model variable
model = None
emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model"""
    global model
    try:
        # Add safe globals for sklearn objects
        import torch.serialization
        torch.serialization.add_safe_globals([
            'sklearn.preprocessing._label.LabelEncoder',
            'sklearn.preprocessing.LabelEncoder'
        ])

        # Try loading with weights_only=False for backward compatibility
        try:
            checkpoint = torch.load('robust_meld_model.pth', map_location=device, weights_only=False)
        except:
            checkpoint = torch.load('robust_meld_model.pth', map_location=device)


        model = SimpleEmotionNet(input_size=50, num_classes=7)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully! Accuracy: {checkpoint['accuracy']:.2f}%")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def extract_robust_features(audio_path, target_sr=16000, max_length=5.0):
    """Simplified, robust feature extraction (same as robust trainer)"""
    try:
        if not os.path.exists(audio_path):
            return np.zeros(50, dtype=np.float32)

        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr, duration=max_length)

        if len(y) == 0:
            return np.zeros(50, dtype=np.float32)

        # Ensure consistent length
        target_length = int(max_length * target_sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        features = []

        # 1. Basic MFCC (most important for emotion)
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([
                float(np.mean(mfcc)),
                float(np.std(mfcc)),
                float(np.max(mfcc)),
                float(np.min(mfcc))
            ])

            # Individual MFCC coefficients means
            mfcc_means = np.mean(mfcc, axis=1)
            features.extend([float(x) for x in mfcc_means[:13]])

        except:
            features.extend([0.0] * 17)  # 4 + 13

        # 2. Spectral features
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.std(spectral_centroids))
            ])

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.extend([
                float(np.mean(spectral_rolloff)),
                float(np.std(spectral_rolloff))
            ])

            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([
                float(np.mean(zcr)),
                float(np.std(zcr))
            ])
        except:
            features.extend([0.0] * 6)

        # 3. Energy features
        try:
            rms = librosa.feature.rms(y=y)[0]
            features.extend([
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.max(rms)),
                float(np.min(rms))
            ])
        except:
            features.extend([0.0] * 4)

        # 4. Chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend([float(x) for x in chroma_mean[:12]])
        except:
            features.extend([0.0] * 12)

        # 5. Pitch and tempo
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(min(10, pitches.shape[1])):  # Only check first 10 frames
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features.extend([
                    float(np.mean(pitch_values)),
                    float(np.std(pitch_values)),
                    float(len(pitch_values))
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(float(tempo))

        except:
            features.extend([0.0] * 4)

        # Ensure exactly 50 features
        features = features[:50]  # Truncate if too many
        while len(features) < 50:  # Pad if too few
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error extracting audio features: {e}")
        # If anything fails, return zero features
        return np.zeros(50, dtype=np.float32)

def extract_text_features(text):
    """Extract text features (placeholder - no text for audio-only)"""
    features = []

    # Basic text statistics (10 features)
    words = text.split() if text else []
    features.extend([
        float(len(text)) if text else 0.0,
        float(len(words)),
        float(np.mean([len(w) for w in words])) if words else 0,
        float(len(set(words)) / len(words)) if words else 0,
        float(len([w for w in words if len(w) > 6])),
        float(len([w for w in words if w.isupper()])),
        float(len([w for w in words if w.islower()])),
        float(text.count(' ') + 1) if text else 0,
        float(sum(1 for c in text if c.isalpha())) if text else 0,
        float(sum(1 for c in text if c.isdigit())) if text else 0
    ])

    # Punctuation features (10 features)
    if text:
        features.extend([
            float(text.count('!')),
            float(text.count('?')),
            float(text.count('.')),
            float(text.count(',')),
            float(text.count(';')),
            float(text.count(':')),
            float(text.count('-')),
            float(text.count('"')),
            float(text.count("'")),
            float(text.count('('))
        ])
    else:
        features.extend([0.0] * 10)

    # Sentiment features (10 features)
    try:
        if text:
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
        else:
            sentiment_polarity = 0.0
            sentiment_subjectivity = 0.0
    except:
        sentiment_polarity = 0.0
        sentiment_subjectivity = 0.0

    positive_words = ['good', 'great', 'happy', 'love', 'wonderful']
    negative_words = ['bad', 'terrible', 'hate', 'awful', 'horrible']

    text_lower = text.lower() if text else ""
    features.extend([
        float(sentiment_polarity),
        float(sentiment_subjectivity),
        float(sum(1 for word in positive_words if word in text_lower)),
        float(sum(1 for word in negative_words if word in text_lower)),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Placeholder features
    ])

    # Ensure exactly 30 features
    features = features[:30]
    while len(features) < 30:
        features.append(0.0)

    return np.array(features, dtype=np.float32)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MELD Emotion Recognition</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #007bff; background-color: #f8f9fa; }
            input[type="file"] { margin: 20px 0; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }
            .emotion { font-size: 24px; font-weight: bold; color: #007bff; }
            .confidence { font-size: 14px; color: #666; }
            .features { margin-top: 20px; }
            .feature-group { margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>MELD Emotion Recognition System</h1>
        <p>Upload an audio file to analyze its emotional content and voice features.</p>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <p>Drop your audio file here or click to browse</p>
                <input type="file" id="audioFile" name="audio" accept=".wav,.mp3,.m4a,.flac,.ogg" required>
            </div>
            <button type="submit">Analyze Emotion</button>
        </form>

        <div id="result" class="result" style="display: none;"></div>

        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();

                const formData = new FormData();
                const audioFile = document.getElementById('audioFile').files[0];

                if (!audioFile) {
                    alert('Please select an audio file');
                    return;
                }

                formData.append('audio', audioFile);

                document.getElementById('result').innerHTML = '<p>Analyzing audio... This may take a moment.</p>';
                document.getElementById('result').style.display = 'block';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.error) {
                        document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                    } else {
                        displayResult(data);
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            };

            function displayResult(data) {
                const resultDiv = document.getElementById('result');

                let html = `
                    <h3>Analysis Results</h3>
                    <div class="emotion">Predicted Emotion: ${data.predicted_emotion}</div>
                    <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>

                    <h4>All Emotion Probabilities:</h4>
                    <div class="features">
                `;

                for (let [emotion, prob] of Object.entries(data.emotion_probabilities)) {
                    const percentage = (prob * 100).toFixed(1);
                    html += `
                        <div class="feature-group">
                            <strong>${emotion}:</strong> ${percentage}%
                            <div style="background-color: #e9ecef; border-radius: 3px; height: 10px; margin: 2px 0;">
                                <div style="background-color: #007bff; height: 100%; width: ${percentage}%; border-radius: 3px;"></div>
                            </div>
                        </div>
                    `;
                }

                html += `
                    </div>

                    <h4>Key Audio Features:</h4>
                    <div class="features">
                        <div class="feature-group"><strong>MFCC Mean:</strong> ${data.key_features.mfcc_mean.toFixed(3)}</div>
                        <div class="feature-group"><strong>MFCC Std:</strong> ${data.key_features.mfcc_std.toFixed(3)}</div>
                        <div class="feature-group"><strong>Spectral Centroid:</strong> ${data.key_features.spectral_centroid.toFixed(1)} Hz</div>
                        <div class="feature-group"><strong>Spectral Rolloff:</strong> ${data.key_features.spectral_rolloff.toFixed(1)} Hz</div>
                        <div class="feature-group"><strong>Energy Mean:</strong> ${data.key_features.energy_mean.toFixed(4)}</div>
                        <div class="feature-group"><strong>Tempo:</strong> ${data.key_features.tempo.toFixed(1)} BPM</div>
                    </div>
                `;

                resultDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    '''

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'error': 'File too large. Please upload an audio file smaller than 50MB.'}), 413

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Please use WAV, MP3, M4A, FLAC, or OGG. You uploaded: {file.filename}'})

    try:
        # Save uploaded file temporarily with a simple name
        import uuid
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        temp_filename = f"temp_{uuid.uuid4().hex[:8]}.{file_extension}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        # Extract features using robust method
        features = extract_robust_features(temp_path)

        # Make prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            output = model(features_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)

            predicted_emotion = emotions[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            # Create emotion probabilities dictionary
            emotion_probs = {emotions[i]: float(probabilities[i]) for i in range(len(emotions))}

            # Extract key features for display (from the 50 robust features)
            key_features = {
                'mfcc_mean': float(features[0]) if len(features) > 0 else 0,
                'mfcc_std': float(features[1]) if len(features) > 1 else 0,
                'spectral_centroid': float(features[17]) if len(features) > 17 else 0,
                'spectral_rolloff': float(features[19]) if len(features) > 19 else 0,
                'energy_mean': float(features[23]) if len(features) > 23 else 0,
                'tempo': float(features[49]) if len(features) > 49 else 0
            }

        # Clean up temporary file
        os.remove(temp_path)

        return jsonify({
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'emotion_probabilities': emotion_probs,
            'key_features': key_features
        })

    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({'error': f'Error processing audio: {str(e)}'})

    except UnicodeDecodeError as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({'error': f'Unicode error with filename: {str(e)}'})

    except OSError as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({'error': f'File system error: {str(e)}'})

if __name__ == '__main__':
    print("Loading MELD Emotion Recognition Web Server...")

    # Load the trained model
    if load_model():
        print(f"Server starting on http://localhost:5000")
        print("Upload audio files to get emotion predictions!")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to load model. Please ensure 'enhanced_meld_model.pth' exists.")
