🧠 MRI Tumor Detection with Explainable AI (XAI)

A deep learning–based web application for brain MRI tumor detection with Explainable AI (XAI) using Grad-CAM, deployed on Streamlit Cloud.

The system not only predicts tumor presence but also explains why the model made its decision, improving transparency and trust.

🚀 Features

📤 Upload brain MRI images (JPG / PNG / JPEG)

🧠 Tumor detection using a CNN-based model

🔍 Visual explanations using Grad-CAM

📝 Textual explanation of model decision

⚠️ Confidence-based prediction warnings

📄 Downloadable diagnostic report

☁️ Cloud-deployed using Streamlit

🧩 Tech Stack

Language: Python

Deep Learning: PyTorch

Explainability: Grad-CAM

Image Processing: OpenCV, PIL

Frontend & Deployment: Streamlit

Model Format: .pth (PyTorch)

🧠 Model Overview

The model processes resized MRI images (224×224)

Outputs a binary prediction:

Tumor Detected

No Tumor Detected

Softmax confidence scores are used to assess certainty

Grad-CAM highlights image regions that influenced the prediction

🔍 Explainable AI (XAI)

The application integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to:

Highlight important regions in MRI images

Show areas that influenced the model’s prediction

Provide both visual and textual explanations

This improves interpretability and allows users to assess whether the model focuses on medically relevant regions.

📄 Diagnostic Report

After prediction, users can download a diagnostic report containing:

Prediction result

Confidence level

Explanation summary

Timestamp

Academic disclaimer

This feature improves usability and documentation.

🌐 Live Demo

👉 Deployed on Streamlit Cloud
(Accessible via the repository’s Streamlit app link)

📁 Project Structure
mri-xai-app/
│
├── app.py                 # Streamlit application
├── model.py               # Model loading and architecture
├── gradcam.py             # Grad-CAM implementation
├── mri_xai_model.pth      # Trained model weights
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

▶️ How to Run Locally (Optional)
pip install -r requirements.txt
streamlit run app.py

⚠️ Disclaimer

This project is a prototype AI system developed for academic and research demonstration only.
It is not intended for clinical diagnosis or real-world medical use.

👨‍💻 Author

Varchas Shrivastava
B.Tech CSE
MRI Tumor Detection with Explainable AI Project

⭐ Acknowledgements

PyTorch & Torchvision

Streamlit

Grad-CAM research methodology

Open-source MRI datasets used for academic purposes
