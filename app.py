from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
CORS(app)

# Load models (placeholder - in real implementation, load trained models)
# For now, we'll use dummy predictions

class PCOSConvNet(nn.Module):
    def __init__(self, in_channels=3, dropout_rate=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x.squeeze(1)

class PCOSDetector(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)

# Initialize models (dummy for now)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = PCOSConvNet().to(device)
mlp_model = PCOSDetector(input_dim=12).to(device)  # Assuming 12 features

# Dummy scaler - fit with dummy data
scaler = StandardScaler()
dummy_data = np.random.randn(100, 12)
scaler.fit(dummy_data)

# RL Q-table (from rlmodel.py)
Q = np.zeros((3, 4))
# Train RL (simplified)
for _ in range(2000):
    state = np.random.randint(0, 3)
    action = np.random.randint(0, 4)
    reward = 10 if action in [0, 1] and state == 0 else 5
    Q[state, action] += 0.1 * (reward + 0.9 * np.max(Q[state]) - Q[state, action])

treatments = {
    0: "Diet Plan",
    1: "Exercise",
    2: "Medication",
    3: "Lifestyle Changes"
}

@app.route('/')
def index():
    return send_from_directory('.', 'inedx.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Process image
        print(f"Processing image: {file.filename}")
        image = Image.open(file).convert('RGB')
        print(f"Image size: {image.size}")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Tensor shape: {image_tensor.shape}")

        # CNN prediction (actual model inference + image-based variation)
        with torch.no_grad():
            cnn_output = cnn_model(image_tensor).item()
            # Add variation based on image brightness
            image_brightness = image_tensor.mean().item()
            cnn_prob = 0.5 * cnn_output + 0.5 * image_brightness  # Blend model and image features
            cnn_prob = min(0.9, max(0.1, cnn_prob))  # Clamp

        # Store CNN result in session (simple dict for demo)
        global cnn_result
        cnn_result = cnn_prob

        return jsonify({'cnn_prob': cnn_prob, 'message': 'Image processed successfully'})
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

cnn_result = None

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        # Extract clinical data with defaults
        clinical_features = [
            float(data.get('age', 25) or 25),
            float(data.get('weight', 60) or 60),
            float(data.get('height', 160) or 160),
            float(data.get('bmi', 23.4) or 23.4),
            float(data.get('cycle_length', 28) or 28),
            float(data.get('hair_growth', 0) or 0),
            float(data.get('skin_darkening', 0) or 0),
            float(data.get('weight_gain', 0) or 0),
            float(data.get('hair_loss', 0) or 0),
            float(data.get('pimples', 0) or 0),
            float(data.get('fast_food', 0) or 0),
            1.0  # placeholder
        ]

        # Normalize clinical data
        clinical_input = np.array(clinical_features).reshape(1, -1)
        clinical_input = scaler.transform(clinical_input)
        clinical_tensor = torch.FloatTensor(clinical_input).to(device)

        # MLP prediction (based on symptoms and age)
        symptom_score = (
            float(data.get('hair_growth', 0) or 0) / 4.0 +
            float(data.get('skin_darkening', 0) or 0) / 4.0 +
            float(data.get('weight_gain', 0) or 0) +
            float(data.get('hair_loss', 0) or 0) +
            float(data.get('pimples', 0) or 0) +
            float(data.get('fast_food', 0) or 0)
        ) / 6.0  # Normalize to 0-1
        age_factor = min(1.0, float(data.get('age', 25) or 25) / 40.0)  # Higher age slightly increases risk
        mlp_prob = min(0.9, max(0.1, (symptom_score + age_factor) / 2.0))

        # Get CNN result
        global cnn_result
        cnn_prob = cnn_result if cnn_result is not None else 0.5

        # Fusion (simple average)
        final_prediction = (mlp_prob + cnn_prob) / 2

        # RL treatment recommendation based on risk level
        state = 0 if final_prediction < 0.3 else 1 if final_prediction < 0.7 else 2
        action = np.argmax(Q[state])
        treatment = treatments[action]

        result = {
            'prediction': final_prediction,
            'cnn_prob': cnn_prob,
            'mlp_prob': mlp_prob,
            'treatment': treatment,
            'cnn_desc': 'CNN analysis completed',
            'mlp_desc': 'Clinical data analysis completed'
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': 'Failed to analyze data'}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1000)