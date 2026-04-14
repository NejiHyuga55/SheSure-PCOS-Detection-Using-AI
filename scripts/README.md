# Training Scripts

This folder contains scripts for training the PCOS detection models.

## train_models.py

Trains both CNN and MLP models for PCOS detection.

### Usage

```bash
cd /path/to/project
python scripts/train_models.py
```

### What it does:
- Trains CNN model on ultrasound images
- Trains MLP model on clinical data
- Saves trained models to `models/` folder
- Saves data scaler for inference

### Requirements:
- PyTorch
- Scikit-learn
- Pandas
- PIL/Pillow
- All dependencies from requirements.txt

### Output:
- `models/pcos_cnn_final.pt` - Trained CNN model
- `models/mlp_model.pt` - Trained MLP model  
- `models/scaler.pkl` - Data scaler for clinical features
