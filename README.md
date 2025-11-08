# ♻️ VIT Chennai Waste Management System

### 🌱 Smart Waste Classification & Disposal Assistant  
A Streamlit-based AI web app to promote a **clean and sustainable campus** at VIT Chennai through intelligent waste segregation and automated maintenance alerts.

---

## 🚀 Live Demo
🔗 **[View on Streamlit Cloud](https://waste-managementt.streamlit.app/)**

---

## 🧠 Project Overview
This project uses **Machine Learning + Image Processing** to automatically classify waste as:
- 🟢 **Biodegradable**
- 🔵 **Non-Biodegradable**

Once classified, the system:
1. Suggests the **nearest disposal point** on campus.
2. Sends **automatic maintenance email notifications** to the collection team.
3. Tracks basic statistics like waste reports, users, and requests processed.

---

## 🧩 Features
- 📸 Upload waste images for AI-based classification  
- 🧮 Real-time confidence score with image feature extraction  
- 📍 Location-based disposal guidance  
- ✉️ Automatic email alerts to maintenance team  
- 🔐 Secure Gmail App Password integration  
- 🧠 Built with Random Forest model trained on custom waste dataset  

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Frontend** | Streamlit |
| **Backend / ML** | scikit-learn, joblib, OpenCV, NumPy, Pandas |
| **Image Processing** | PIL (Pillow), NumPy |
| **Email Notifications** | smtplib, email.mime |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git + GitHub |

---

## 🧪 Model Details
- Algorithm: **Random Forest Classifier**
- Input: Extracted visual + color + shape features
- Output: Waste type prediction (biodegradable / non-biodegradable)
- Scaling: StandardScaler used for normalization

Model files:
- `random_forest_model_fixed.pkl`
- `scaler_fixed.pkl`

---

## ⚙️ Installation (Run Locally)
1. Clone the repository:
   ```bash
   git clone https://github.com/Akhila707/Waste-Management.git
   cd Waste-Management
