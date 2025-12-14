import torch
import torch.nn as nn
from torchvision import models
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=14S66AvfW1F4Q6nY2BRvJ1Lr6yPGlO5Y5"
MODEL_PATH = "mri_xai_model.pth"

def load_model(model_path, device):
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model
