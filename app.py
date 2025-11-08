# ============================================================
#  VIT Chennai Waste Management System - Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import joblib
import pickle
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="VIT Chennai Waste Management",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom Styling
# ------------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
    color: #212529;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    color: #2e7d32;
    text-align: center;
    padding-bottom: 10px;
    border-bottom: 2px solid #4caf50;
}
h2 {
    color: #388e3c;
    margin-top: 20px;
}
h3 { color: #43a047; }
div.stButton > button:first-child {
    background-color: #4caf50;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 10px 24px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
}
div.stButton > button:first-child:hover {
    background-color: #388e3c;
    color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.step-box {
    background-color: #f8f9fa;
    border-left: 4px solid #4caf50;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Initialize session state
# ------------------------------------------------------------
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'notification_sent' not in st.session_state:
    st.session_state.notification_sent = False

# ------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model_fixed.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_fixed.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        st.success("✅ Successfully loaded model and scaler!")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
# ------------------------------------------------------------
# Email Sender (fixed receiver)
# ------------------------------------------------------------
def send_maintenance_email(notification_data):
    """Send alert to maintenance team using a fixed Gmail account."""
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        # >>> INSERT RECEIVER EMAIL & APP PASSWORD HERE <<<
        sender_email = "akhila.pv2024@vitstudent.ac.in"   # fixed maintenance inbox
        sender_password = "lllsomwhzaxhzflo"        # replace locally on your system

        recipient_emails = [sender_email]  # can add more addresses if needed

        # Create message
        msg = MIMEMultipart()
        msg['From'] = "VIT Waste Management System <no-reply@vitwaste.edu>"
        msg['To'] = ", ".join(recipient_emails)
        msg['Subject'] = f"🚮 Waste Collection Alert - {notification_data['waste_type_name']} at {notification_data['location']}"

        body = f"""
        Dear Maintenance Team,

        A new waste collection request has been reported in the VIT Chennai Waste Management System.

        🗑️ Waste Type: {notification_data['waste_type_name']}
        📍 Location: {notification_data['location']}
        👤 Reported By: {notification_data['reported_by']}
        🕒 Reported At: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        🪣 Nearest Disposal Point:
        {notification_data['disposal_info']['name']} ({notification_data['disposal_info']['distance']})

        Disposal Guidelines:
        {"Use green bins for biodegradable items." if notification_data['waste_type']=="biodegradable" else "Use blue bins for recyclable items."}

        Please collect and dispose of the waste promptly and update the campus log.

        Regards,
        ♻️ VIT Chennai Waste Management System
        """

        msg.attach(MIMEText(body, "plain"))

        # Send
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_emails, msg.as_string())
        server.quit()

        return True
    except Exception as e:
        st.error(f"❌ Email send error: {e}")
        return False

# ------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------
def extract_features(image):
    """Extract simple numeric features from image for classifier."""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    # basic region features
    h, w = gray.shape
    region_h, region_w = h//2, w//4
    hog = []
    for r in range(2):
        for c in range(4):
            region = gray[r*region_h:(r+1)*region_h, c*region_w:(c+1)*region_w]
            hog.append(np.mean(region))

    # color
    if len(img_array.shape) == 3:
        mean_r, mean_g, mean_b = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
        std_r, std_g, std_b = np.std(img_array[:,:,0]), np.std(img_array[:,:,1]), np.std(img_array[:,:,2])
    else:
        mean_r = mean_g = mean_b = np.mean(gray)
        std_r = std_g = std_b = np.std(gray)

    threshold = np.mean(gray)
    binary = gray > threshold
    area = np.sum(binary)
    perimeter = np.sum(binary.astype(int)[:,1:] != binary.astype(int)[:,:-1]) + np.sum(binary.astype(int)[1:,:] != binary.astype(int)[:-1,:])
    circularity = 4*np.pi*area/(perimeter**2+1e-6)
    aspect_ratio = w/h
    extent = area/(w*h)
    solidity = min(1.0, area/(w*h))

    feats = {
        'hog_0':hog[0],'hog_1':hog[1],'hog_2':hog[2],'hog_3':hog[3],
        'hog_4':hog[4],'hog_5':hog[5],'hog_6':hog[6],'hog_7':hog[7],
        'mean_red':mean_r,'mean_green':mean_g,'mean_blue':mean_b,
        'std_red':std_r,'std_green':std_g,'std_blue':std_b,
        'area':area,'perimeter':perimeter,'circularity':circularity,
        'aspect_ratio':aspect_ratio,'extent':extent,'solidity':solidity
    }
    return pd.DataFrame([feats])

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
def predict_waste_type(image, model, scaler):
    try:
        feats = extract_features(image)
        feature_cols = feats.columns
        scaled = scaler.transform(feats[feature_cols])
        pred = model.predict(scaled)
        prob = model.predict_proba(scaled)
        return pred[0], prob[0], feats
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# ------------------------------------------------------------
# Disposal Info
# ------------------------------------------------------------
def get_nearest_disposal(location, waste_type):
    disposal_points = {
        'AB1': {'biodegradable': {'name':'AB1 Compost Pit','distance':'50 m'},
                'non-biodegradable': {'name':'AB1 Recycle Bin','distance':'30 m'}},
        'AB2': {'biodegradable': {'name':'AB2 Organic Container','distance':'40 m'},
                'non-biodegradable': {'name':'AB2 Recycling Station','distance':'60 m'}},
        'Library': {'biodegradable': {'name':'Library Food Waste Bin','distance':'20 m'},
                    'non-biodegradable': {'name':'Library Paper Recycling','distance':'15 m'}},
        'Boys Hostel': {'biodegradable': {'name':'Boys Hostel Compost','distance':'120 m'},
                        'non-biodegradable': {'name':'Boys Hostel Recycle Center','distance':'90 m'}},
    }
    return disposal_points.get(location, {}).get(waste_type, {'name':'Main Campus Disposal','distance':'200 m'})

# ------------------------------------------------------------
# Send Maintenance Alert wrapper
# ------------------------------------------------------------
def send_maintenance_alert(notification_data):
    """Wrapper to call email sender and display result in UI."""
    email_sent = send_maintenance_email(notification_data)
    if email_sent:
        st.success("📧 Maintenance team notified successfully!")
        st.balloons()
    else:
        st.warning("⚠️ Failed to send email notification.")
    st.session_state.notification_sent = True

# ------------------------------------------------------------
# Main Application
# ------------------------------------------------------------
def main():
    model, scaler = load_model()
    if model is None or scaler is None:
        st.stop()

    st.title("♻️ VIT Chennai Waste Management")
    st.markdown("Upload waste photo → Classify → Get disposal instructions → Notify maintenance team.")

    st.sidebar.header("📝 Reporter Info")
    user_name = st.sidebar.text_input("Your Name", value="VIT Student")

    st.markdown("## 📷 Upload Waste Photo")
    file = st.file_uploader("Choose image...", type=["jpg","jpeg","png"])
    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("## 📍 Select Location")
        location = st.selectbox("Select location", ["AB1","AB2","AB3","AB4","Library","Girls Hostel","Boys Hostel"])

        if st.button("🔍 Classify Waste"):
            with st.spinner("Analyzing image..."):
                pred, prob, feats = predict_waste_type(image, model, scaler)
                if pred is None:
                    st.error("Prediction failed.")
                    return
                waste_type = "biodegradable" if pred==0 else "non-biodegradable"
                waste_name = waste_type.capitalize()
                conf = prob[pred]*100 if prob is not None else 0
                disposal = get_nearest_disposal(location, waste_type)
                st.session_state.classification_results = {
                    'waste_type':waste_type,
                    'waste_type_name':waste_name,
                    'confidence':conf,
                    'disposal_info':disposal,
                    'location':location,
                    'reported_by':user_name
                }
                st.success(f"Detected: **{waste_name}** waste ({conf:.2f}% confidence)")
                st.info(f"Nearest Disposal: {disposal['name']} – {disposal['distance']}")
                with st.expander("Feature Details"):
                    st.dataframe(feats)

        if st.session_state.classification_results:
            st.markdown("## 🔔 Maintenance Notification")
            if not st.session_state.notification_sent:
                if st.button("Notify Maintenance Team"):
                    send_maintenance_alert(st.session_state.classification_results)
            else:
                st.success("✅ Already notified maintenance team.")

    st.markdown("---")
    st.markdown("### 🌱 About This Project")
    st.markdown("""
    The VIT Chennai Campus Waste Management System promotes sustainability by combining
    machine-learning waste classification and automatic maintenance alerts.
    """)

    st.markdown("### 📈 Today's Statistics (demo)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Waste Reports","42","+12")
    col2.metric("Requests Processed","38","+10")
    col3.metric("Active Users","127","+5")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
