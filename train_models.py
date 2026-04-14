#!/usr/bin/env python3
"""
PCOS Detection Model Training Script
Trains CNN for ultrasound image analysis and MLP for clinical data analysis
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ============================================================================
# CNN MODEL DEFINITION (matches app.py)
# ============================================================================

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

# ============================================================================
# MLP MODEL DEFINITION (matches app.py)
# ============================================================================

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

# ============================================================================
# DATASET CLASSES
# ============================================================================

class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_cnn_model():
    print("🏥 Training CNN Model for Ultrasound Analysis")
    print("=" * 50)

    # Data paths
    train_dir = "ultrasound_data/train"
    test_dir = "ultrasound_data/test"

    # Collect image paths and labels
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # Training data
    infected_train = []
    notinfected_train = []

    for f in os.listdir(os.path.join(train_dir, "infected")):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(train_dir, "infected", f)
            try:
                Image.open(img_path).verify()  # Check if image is valid
                infected_train.append(img_path)
            except:
                print(f"Skipping corrupted image: {img_path}")

    for f in os.listdir(os.path.join(train_dir, "notinfected")):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(train_dir, "notinfected", f)
            try:
                Image.open(img_path).verify()  # Check if image is valid
                notinfected_train.append(img_path)
            except:
                print(f"Skipping corrupted image: {img_path}")

    train_images.extend(infected_train + notinfected_train)
    train_labels.extend([1.0] * len(infected_train) + [0.0] * len(notinfected_train))

    # Test data
    infected_test = []
    notinfected_test = []

    for f in os.listdir(os.path.join(test_dir, "infected")):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, "infected", f)
            try:
                Image.open(img_path).verify()  # Check if image is valid
                infected_test.append(img_path)
            except:
                print(f"Skipping corrupted image: {img_path}")

    for f in os.listdir(os.path.join(test_dir, "notinfected")):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, "notinfected", f)
            try:
                Image.open(img_path).verify()  # Check if image is valid
                notinfected_test.append(img_path)
            except:
                print(f"Skipping corrupted image: {img_path}")

    test_images.extend(infected_test + notinfected_test)
    test_labels.extend([1.0] * len(infected_test) + [0.0] * len(notinfected_test))

    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = UltrasoundDataset(train_images, train_labels, train_transform)
    test_dataset = UltrasoundDataset(test_images, test_labels, test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = PCOSConvNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Training loop
    best_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                val_probs.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss = val_loss / len(test_loader)
        val_auc = roc_auc_score(val_labels_list, val_probs)

        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/pcos_cnn_final.pt")
            patience_counter = 0
            print("  📁 Model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation accuracy: {best_acc:.4f}")
    return model

def train_mlp_model():
    print("\n🏥 Training MLP Model for Clinical Data Analysis")
    print("=" * 50)

    # Load clinical dataset
    df = pd.read_csv("data/pcos_prediction_dataset.csv")
    print(f"Dataset shape: {df.shape}")

    # Preprocess data
    # Drop non-numeric columns for simplicity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    # Handle target column
    target_col = 'Diagnosis' if 'Diagnosis' in df.columns else df.columns[-1]
    if target_col not in df_numeric.columns:
        # Convert target to numeric if needed
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        df_numeric[target_col] = df[target_col]

    # Features and target
    feature_cols = [col for col in df_numeric.columns if col != target_col]
    X = df_numeric[feature_cols].values
    y = df_numeric[target_col].values

    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for app.py
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"Training features shape: {X_train_scaled.shape}")
    print(f"Test features shape: {X_test_scaled.shape}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    input_dim = X_train_scaled.shape[1]
    model = PCOSDetector(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                val_probs.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss = val_loss / len(test_loader)
        val_auc = roc_auc_score(val_labels_list, val_probs)

        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/mlp_model.pt")
            patience_counter = 0
            print("  📁 Model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation accuracy: {best_acc:.4f}")
    return model

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting PCOS Detection Model Training")
    print("=" * 60)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Train CNN model
    try:
        cnn_model = train_cnn_model()
        print("✅ CNN model training completed!")
    except Exception as e:
        print(f"❌ CNN training failed: {e}")

    # Train MLP model
    try:
        mlp_model = train_mlp_model()
        print("✅ MLP model training completed!")
    except Exception as e:
        print(f"❌ MLP training failed: {e}")

    print("\n🎉 Training completed!")
    print("📁 Models saved to 'models/' directory")
    print("🔄 Restart the Flask app to use the new trained models")