#!/usr/bin/env python3
"""
Web server for RAVDESS Emotion Recognition - Updated for 68.9% accuracy model
Upload audio files and get emotion predictions using optimized RAVDESS model
"""

import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import tempfile
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

class OptimizedMediumNet(nn.Module):
    """Optimized model architecture for RAVDESS (65 features, 8 emotions)"""
    def __init__(self, input_size=65, num_classes=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Global variables for RAVDESS
model = None
feature_scaler = None
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_optimized_features(audio_path, target_sr=16000, max_length=6.0):
    """Extract optimized 65-feature set (matches your best model)"""
    try:
        if not os.path.exists(audio_path):
            return np.zeros(65, dtype=np.float32)

        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr, duration=max_length)

        if len(y) == 0 or np.max(np.abs(y)) == 0:
            return np.zeros(65, dtype=np.float32)

        # Normalize audio
        y = y / np.max(np.abs(y))

        # Ensure consistent length
        target_length = int(max_length * target_sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        features = []

        # 1. Enhanced MFCC features (39 features total)
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Original MFCCs (13 features)
            mfcc_means = np.mean(mfcc, axis=1)
            features.extend([float(x) for x in mfcc_means])

            # Delta MFCCs (13 features)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_means = np.mean(mfcc_delta, axis=1)
            features.extend([float(x) for x in mfcc_delta_means])

            # Delta-delta MFCCs (13 features)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_delta2_means = np.mean(mfcc_delta2, axis=1)
            features.extend([float(x) for x in mfcc_delta2_means])

        except:
            features.extend([0.0] * 39)

        # 2. Spectral features (8 features)
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]

            features.extend([
                float(np.mean(spectral_centroids) / 8000),
                float(np.std(spectral_centroids) / 8000),
                float(np.mean(spectral_rolloff) / 8000),
                float(np.std(spectral_rolloff) / 8000),
                float(np.mean(spectral_bandwidth) / 8000),
                float(np.std(spectral_bandwidth) / 8000),
                float(np.mean(zcr)),
                float(np.std(zcr))
            ])
        except:
            features.extend([0.0] * 8)

        # 3. Energy features (4 features)
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

        # 4. Spectral contrast (7 features)
        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_means = np.mean(spectral_contrast, axis=1)
            features.extend([float(x) for x in contrast_means])
        except:
            features.extend([0.0] * 7)

        # 5. Tonnetz (6 features)
        try:
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_means = np.mean(tonnetz, axis=1)
            features.extend([float(x) for x in tonnetz_means])
        except:
            features.extend([0.0] * 6)

        # 6. Advanced pitch features (1 feature)
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(float(tempo / 200))
        except:
            features.append(0.0)

        # Total: 39 + 8 + 4 + 7 + 6 + 1 = 65 features
        features = features[:65]
        while len(features) < 65:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    except Exception as e:
        return np.zeros(65, dtype=np.float32)

def load_model():
    """Load the optimized RAVDESS model"""
    global model, feature_scaler
    try:
        # Look for your best model file
        model_files = [
            'enhanced_model_with_augmentation.pth',
            'best_model_with_augmentation.pth',
            'final_optimized_ensemble.pth'
        ]

        model_path = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model_path = model_file
                break

        if not model_path:
            raise Exception(f"No model found. Expected one of: {model_files}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Load model
        model = OptimizedMediumNet(input_size=65, num_classes=8)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Load feature scaler if it exists
        if 'feature_scaler' in checkpoint and checkpoint['feature_scaler'] is not None:
            feature_scaler = checkpoint['feature_scaler']

        accuracy = checkpoint.get('accuracy', 'unknown')
        print(f"RAVDESS model loaded successfully! Accuracy: {accuracy}")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'})

    try:
        # Save uploaded file
        import uuid
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        temp_filename = f"temp_{uuid.uuid4().hex[:8]}.{file_extension}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        # Extract optimized features (65 features)
        features = extract_optimized_features(temp_path)

        # Apply feature scaling if available
        if feature_scaler is not None:
            features = feature_scaler.transform(features.reshape(1, -1))[0]

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

            # Extract key features for display
            key_features = {
                'mfcc_mean': float(np.mean(features[:13])) if len(features) >= 13 else 0,
                'delta_mfcc_mean': float(np.mean(features[13:26])) if len(features) >= 26 else 0,
                'spectral_centroid': float(features[39] * 8000) if len(features) > 39 else 0,
                'spectral_rolloff': float(features[41] * 8000) if len(features) > 41 else 0,
                'energy_mean': float(features[47]) if len(features) > 47 else 0,
                'tempo': float(features[64] * 200) if len(features) > 64 else 0
            }

        # Clean up
        os.remove(temp_path)

        return jsonify({
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'emotion_probabilities': emotion_probs,
            'key_features': key_features,
            'model_info': {
                'dataset': 'RAVDESS',
                'accuracy': '68.9%',
                'features': 65,
                'emotions': len(emotions)
            }
        })

    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Error processing audio: {str(e)}'})

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAVDESS Emotion Recognition</title>
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
            .model-info { background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>RAVDESS Emotion Recognition System</h1>
        <p>Advanced audio emotion recognition using optimized RAVDESS model (68.9% accuracy)</p>

        <div class="model-info">
            <strong>Model Information</strong><br>
            Dataset: RAVDESS Speech Database<br>
            Accuracy: 68.9% on validation set<br>
            Features: 65 enhanced audio features<br>
            Emotions: 8 categories (angry, calm, disgust, fearful, happy, neutral, sad, surprised)
        </div>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <p>Upload your audio file for emotion analysis</p>
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

                document.getElementById('result').innerHTML = '<p>Analyzing audio with RAVDESS model...</p>';
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
                    <h3>Emotion Analysis Results</h3>
                    <div class="emotion">Predicted Emotion: ${data.predicted_emotion}</div>
                    <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>

                    <h4>All Emotion Probabilities:</h4>
                    <div class="features">
                `;

                // Sort emotions by probability for better display
                const sortedEmotions = Object.entries(data.emotion_probabilities)
                    .sort(([,a], [,b]) => b - a);

                for (let [emotion, prob] of sortedEmotions) {
                    const percentage = (prob * 100).toFixed(1);
                    const barColor = emotion === data.predicted_emotion ? '#28a745' : '#007bff';

                    html += `
                        <div class="feature-group">
                            <strong>${emotion}:</strong> ${percentage}%
                            <div style="background-color: #e9ecef; border-radius: 3px; height: 12px; margin: 2px 0;">
                                <div style="background-color: ${barColor}; height: 100%; width: ${percentage}%; border-radius: 3px;"></div>
                            </div>
                        </div>
                    `;
                }

                html += `
                    </div>

                    <h4>Advanced Audio Features:</h4>
                    <div class="features">
                        <div class="feature-group"><strong>MFCC Mean:</strong> ${data.key_features.mfcc_mean.toFixed(3)}</div>
                        <div class="feature-group"><strong>Delta MFCC Mean:</strong> ${data.key_features.delta_mfcc_mean.toFixed(3)}</div>
                        <div class="feature-group"><strong>Spectral Centroid:</strong> ${data.key_features.spectral_centroid.toFixed(1)} Hz</div>
                        <div class="feature-group"><strong>Spectral Rolloff:</strong> ${data.key_features.spectral_rolloff.toFixed(1)} Hz</div>
                        <div class="feature-group"><strong>Energy Mean:</strong> ${data.key_features.energy_mean.toFixed(4)}</div>
                        <div class="feature-group"><strong>Tempo:</strong> ${data.key_features.tempo.toFixed(1)} BPM</div>
                    </div>

                    <div class="model-info" style="margin-top: 20px;">
                        <strong>Model Performance:</strong> ${data.model_info.accuracy} accuracy on ${data.model_info.dataset} dataset
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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'model_loaded': model is not None,
        'feature_scaler_loaded': feature_scaler is not None,
        'device': str(device),
        'model_type': 'RAVDESS Optimized',
        'accuracy': '68.9%',
        'features': 65,
        'emotions': len(emotions)
    }
    return jsonify(status)

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify everything works"""
    return jsonify({
        'status': 'Server running',
        'model_loaded': model is not None,
        'emotions': emotions,
        'feature_count': 65,
        'model_accuracy': '68.9%'
    })

def load_model():
    """Load the trained RAVDESS model"""
    global model, feature_scaler
    try:
        # Look for your best performing model
        model_files = [
            'enhanced_model_with_augmentation.pth',
            'best_model_with_augmentation.pth',
            'final_optimized_ensemble.pth'
        ]

        model_path = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model_path = model_file
                print(f"Found model: {model_file}")
                break

        if not model_path:
            print(f"No model found. Looking for: {model_files}")
            print("Please ensure you have run the optimized trainer and saved the model.")
            return False

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Create model with correct architecture
        model = OptimizedMediumNet(input_size=65, num_classes=8)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Load feature scaler if available
        if 'feature_scaler' in checkpoint and checkpoint['feature_scaler'] is not None:
            feature_scaler = checkpoint['feature_scaler']
            print("Feature scaler loaded")
        else:
            print("No feature scaler found - using raw features")

        accuracy = checkpoint.get('accuracy', 68.9)
        print(f"RAVDESS model loaded! Accuracy: {accuracy:.1f}%")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == '__main__':
    print("Loading RAVDESS Emotion Recognition Web Server...")
    print("Model: Optimized RAVDESS (68.9% accuracy)")
    print("Features: 65 enhanced audio features")
    print("Emotions: 8 categories")

    # Load the trained model
    if load_model():
        print(f"Server starting on http://localhost:5001")
        print("Upload audio files to get emotion predictions!")
        print("\nEndpoints:")
        print("  GET  / - Web interface")
        print("  POST /predict - Audio prediction API")
        print("  GET  /health - Health check")
        print("  GET  /test - Test endpoint")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to load model.")
        print("Make sure you have run the optimized trainer and have one of these files:")
        print("  - enhanced_model_with_augmentation.pth")
        print("  - best_model_with_augmentation.pth")
        print("  - final_optimized_ensemble.pth")
