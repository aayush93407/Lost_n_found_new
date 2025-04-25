import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle

# Load ResNet-18 model
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
resnet.eval()

# Image preprocessing function 
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Feature extraction function 
def extract_features(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = resnet(image_tensor)
    # Flatten the output to 1D vector of size 512 (for ResNet-18)
    features = features.view(-1).numpy()
    return features

# Load dataset (example CSV with image paths)
df = pd.read_csv("lost_items_dataset.csv")
df['features'] = df['image_path'].apply(lambda x: extract_features(x) if os.path.exists(x) else None)
df.dropna(inplace=True)

# Save features to a new pickle file
df.to_pickle("updated_features.pkl")

# Check the updated features
print(df.head())
