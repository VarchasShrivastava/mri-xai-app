import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from model import load_model
from gradcam import GradCAM

st.set_page_config(page_title="MRI Tumor Detection (XAI)", layout="wide")

st.title("🧠 MRI Tumor Detection with Explainable AI")
st.write("Upload a brain MRI image to detect tumor presence and visualize model explanation using Grad-CAM.")

device = torch.device("cpu")

model = load_model("mri_xai_model.pth", device)
gradcam = GradCAM(model, model.layer4)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((224, 224))
    input_tensor = transform(image_resized).unsqueeze(0)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()
    if confidence < 0.6:
    st.warning("⚠️ Low confidence prediction. The model is uncertain. Please consult a medical professional.")
    elif confidence < 0.8:
    st.info("ℹ️ Moderate confidence prediction. Use results cautiously.")
    else:
    st.success("✅ High confidence prediction.")


    cam = gradcam.generate(input_tensor, pred_class)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(np.array(image_resized), 0.6, heatmap, 0.4, 0)

    col1, col2, col3 = st.columns(3)

    col1.image(image_resized, caption="Original MRI", use_column_width=True)
    col2.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
    col3.image(overlay, caption="Overlay Explanation", use_column_width=True)

    label = "Tumor Detected" if pred_class == 1 else "No Tumor Detected"
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")

    st.caption("⚠️ Prototype system for academic demonstration only.")

