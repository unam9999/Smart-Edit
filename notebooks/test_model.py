from torchvision import models, transforms
from PIL import Image
import torch

# Load model
model = models.efficientnet_b0(pretrained=True)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

import os

# Get path relative to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "..", "data", "test", "test.jpg")

# Load image
img = Image.open(image_path)
img = transform(img).unsqueeze(0)

import json
import urllib.request

# Predict
with torch.no_grad():
    output = model(img)

# Since we haven't trained on our 14 custom categories yet, 
# the model still outputs predictions for the original 1000 ImageNet categories.
# Let's fetch the ImageNet class labels to see what it thinks the image is!
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    with urllib.request.urlopen(url) as f:
        categories = [s.strip().decode('utf-8') for s in f.readlines()]
    
    # Get top prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    print("\n--- PREDICTION RESULTS ---")
    print(f"Predicted Category: {categories[top_catid[0].item()]}")
    print(f"Confidence Score:   {top_prob[0].item() * 100:.2f}%\n")
    print("(Note: This is using the base ImageNet model, as custom training hasn't happened yet.)")

except Exception as e:
    print("Prediction done, but couldn't download class labels to show the name.")
    print("Raw output tensor shape:", output.shape)