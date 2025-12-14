from datetime import datetime
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from model import load_model
from gradcam import GradCAM

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="MRI Tumor Detection (XAI)", layout="wide")

# --------------------------------------------------
# UI Styling
# --------------------------------------------------
st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 16px;
    color: #b0b0b0;
    margin-bottom: 25px;
}
.section {
    font-size: 22px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 MRI Tumor Detection with Explainable AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a brain MRI scan to detect tumor presence and understand model decisions using Grad-CAM.</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------------------------------------
# Load model
# --------------------------------------------------
device = torch.device("cpu")
model = load_model("mri_xai_model.pth", device)
gradcam = GradCAM(model, model.layer4)

# --------------------------------------------------
# Image transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Upload section
# --------------------------------------------------
st.markdown('<div class="section">📤 Upload MRI Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((224, 224))
    input_tensor = transform(image_resized).unsqueeze(0)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    # --------------------------------------------------
    # Confidence warnings
    # --------------------------------------------------
    if confidence < 0.6:
        st.warning("⚠️ Low confidence prediction. The model is uncertain. Please consult a medical professional.")
    elif confidence < 0.8:
        st.info("ℹ️ Moderate confidence prediction. Use results cautiously.")
    else:
        st.success("✅ High confidence prediction.")

    # --------------------------------------------------
    # Visual explanation
    # --------------------------------------------------
    st.markdown('<div class="section">🖼️ Model Visual Explanation</div>', unsafe_allow_html=True)

    cam = gradcam.generate(input_tensor, pred_class)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(image_resized), 0.6, heatmap, 0.4, 0)

    col1, col2, col3 = st.columns(3)
    col1.image(image_resized, caption="Original MRI", use_column_width=True)
    col2.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
    col3.image(overlay, caption="Overlay Explanation", use_column_width=True)

    # --------------------------------------------------
    # Textual explanation
    # --------------------------------------------------
    st.markdown('<div class="section">🧠 Model Interpretation</div>', unsafe_allow_html=True)
    st.markdown("""
**Why did the model decide this?**

The deep learning model analyzes visual patterns in the MRI scan such as:
- Irregular tissue textures  
- Intensity variations  
- Structural abnormalities  

Using **Grad-CAM**, highlighted regions show areas that most influenced the prediction:
- 🔴 **Red / Yellow** → Strong influence  
- 🔵 **Blue** → Minimal influence  

This improves transparency and trust in the model’s decision-making.
""")

    # --------------------------------------------------
    # Prediction summary
    # --------------------------------------------------
    label = "Tumor Detected" if pred_class == 1 else "No Tumor Detected"
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")

    # --------------------------------------------------
    # Downloadable report
    # --------------------------------------------------
    st.markdown('<div class="section">📄 Diagnostic Report</div>', unsafe_allow_html=True)

    report_text = f"""
MRI Tumor Detection – Diagnostic Report
--------------------------------------
Prediction Result : {label}
Confidence Level  : {confidence*100:.2f}%

Model Explanation:
Grad-CAM was used to identify image regions that most influenced
the model’s prediction.

Disclaimer:
This report is generated by an AI-based prototype system and is intended
for academic and research demonstration only. It should not be used for
clinical diagnosis.

Generated On:
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    st.download_button(
        label="📄 Download Diagnostic Report",
        data=report_text,
        file_name="mri_xai_report.txt",
        mime="text/plain"
    )

# --------------------------------------------------
# Footer disclaimer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ This is a prototype AI system developed for academic and research demonstration only. "
    "It is not intended for clinical diagnosis."
)
