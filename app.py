import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Load pre-trained ResNet-18 model  
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
    try:
        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            features = resnet(image_tensor)
        
        # Flatten the output to 1D vector of size 512 (for ResNet-18)
        features = features.view(-1).numpy()
        
        print(f"Extracted features size: {features.shape}")  # Log feature size
        return features
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None

# Load dataset (ensure path is correct)
features_path = 'features.pkl'
if not os.path.exists(features_path):
    print(f"Error: {features_path} not found!")

df = pd.read_pickle(features_path)

# Check if the dataframe is loaded properly
print(df.head())  # Log the first few rows to verify

# Function to find best match
def find_best_match(found_image_path):
    try:
        found_features = extract_features(found_image_path)
        if found_features is None:
            print(f"Error: Extracted features are None for {found_image_path}")
            return None
        
        print(f"Found features: {found_features[:5]}...")  # Log part of found features for debugging
        
        stored_features = np.stack(df['features'].values)
        print(f"Stored features shape: {stored_features.shape}")  # Log stored features shape
        
        # Ensure that the dimensions match
        if found_features.shape[0] != stored_features.shape[1]:
            print(f"Feature dimension mismatch: found_features {found_features.shape}, stored_features {stored_features.shape}")
            return None
        
        similarities = cosine_similarity([found_features], stored_features)
        best_match_idx = np.argmax(similarities)
        
        return df.iloc[best_match_idx]
    except Exception as e:
        print(f"Error in find_best_match: {e}")
        return None

# Flask Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

@app.route("/match", methods=["POST"])
def match():
    if "file" not in request.files:
        print("No file part")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        print("No selected file")
        return redirect(request.url)

    if file:
        os.makedirs("static/uploads", exist_ok=True)
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)
        
        try:
            best_match = find_best_match(file_path)
            if best_match is None:
                print("No match found!")
                return "No match found.", 500
            
            print(f"Best match: {best_match}")  # Log best match data
            
            return render_template("result.html", 
                                   image=file.filename,
                                   item_name=best_match["item_name"],
                                   person_name=best_match["person_name"],
                                   description=best_match["description"],
                                   location=best_match["location"],
                                   owner_contact=best_match["owner_contact"])
        except Exception as e:
            print(f"Error while processing the image: {e}")
            return "An error occurred during the match process.", 500

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
