import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/best_model.pkl", "rb"))
sc = pickle.load(open("model/scaler.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heartify", page_icon="❤️", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: white;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
}
.result-low {
    background-color: #052e16;
    padding: 20px;
    border-radius: 10px;
    color: #22c55e;
}
.result-medium {
    background-color: #3f2f00;
    padding: 20px;
    border-radius: 10px;
    color: #facc15;
}
.result-high {
    background-color: #3f0d0d;
    padding: 20px;
    border-radius: 10px;
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h2 style='color:white;'>❤️ Heartify</h2>", unsafe_allow_html=True)
st.markdown('<div class="big-title">Heart Disease <span style="color:#ef4444;">Risk Predictor</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient medical data to assess heart disease risk using ML</div>', unsafe_allow_html=True)

# ---------------- STATS ----------------
col1, col2, col3 = st.columns(3)
col1.metric("Prediction Type", "ML-Based Analysis")
col2.metric("Parameters", "13")
col3.metric("Output", "Risk Levels")

st.markdown("---")

# ---------------- INPUT ----------------
st.markdown("## 📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 20, 100, 35)
    cp = st.selectbox("Chest Pain Type", ["Typical Angina (Classic Heart Pain)", "Atypical Angina (Unusual Heart Pain)", "Non-anginal Pain (Not Heart Related)", "Asymptomatic (No Pain)"])
    chol = st.slider("Cholesterol Level (mg/dl)", 100, 600, 200)
    restecg = st.selectbox("Resting ECG Result", ["Normal", "Abnormal (ST-T wave abnormality)", "Hypertrophy (Enlarged Heart)"])
    exang = st.selectbox("Chest Pain Induced by Exercise?", ["No", "Yes"])

with col2:
    sex = st.selectbox("Biological Sex", ["Female", "Male"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.slider("ST Depression (Heart Stress Level)", 0.0, 6.0, 1.0)

col3, col4 = st.columns(2)

with col3:
    slope = st.selectbox("Physical Exertion Response (Slope)", ["Upsloping (Healthy)", "Flat (Normal)", "Downsloping (Unhealthy)"])

with col4:
    ca = st.selectbox("Number of Blocked Major Vessels", [0,1,2,3,4])
    thal = st.selectbox("Thalassemia (Blood Disorder Type)", ["Normal", "Fixed Defect (No blood flow in some parts)", "Reversible Defect (Abnormal blood flow)"])

# ---------------- CONVERSION ----------------
sex = 1 if sex == "Male" else 0

cp = {
    "Typical Angina (Classic Heart Pain)": 1,
    "Atypical Angina (Unusual Heart Pain)": 2,
    "Non-anginal Pain (Not Heart Related)": 3,
    "Asymptomatic (No Pain)": 4
}[cp]

fbs = 1 if fbs == "Yes" else 0

restecg = {
    "Normal": 0,
    "Abnormal (ST-T wave abnormality)": 1,
    "Hypertrophy (Enlarged Heart)": 2
}[restecg]

exang = 1 if exang == "Yes" else 0

slope = {
    "Upsloping (Healthy)": 1,
    "Flat (Normal)": 2,
    "Downsloping (Unhealthy)": 3
}[slope]

thal = {
    "Normal": 3,
    "Fixed Defect (No blood flow in some parts)": 6,
    "Reversible Defect (Abnormal blood flow)": 7
}[thal]

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Heart Risk", use_container_width=True):

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    input_scaled = sc.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1]

    # ---------------- SAME LOGIC (UNCHANGED) ----------------
    if probability > 0.75:
        risk = "High Risk"
        css_class = "result-high"
        recs = """
🔴 Consult a cardiologist immediately  
🔴 Avoid high-fat foods  
🔴 Monitor blood pressure regularly  
🔴 Start light physical activity  
        """
    elif probability > 0.40:
        risk = "Moderate Risk"
        css_class = "result-medium"
        recs = """
🟡 Improve diet (low sodium, low cholesterol)  
🟡 Engage in regular exercise  
🟡 Reduce stress  
        """
    else:
        risk = "Low Risk"
        css_class = "result-low"
        recs = """
🟢 Maintain a healthy lifestyle  
🟢 Regular health check-ups  
        """

    # ---------------- OUTPUT ----------------
    st.markdown("## 👉 Prediction Summary")

    st.markdown(f"""
    <div class="{css_class}">
    ⚠️ <b>{risk}</b><br>
    Probability: {round(probability*100,2)}%
    </div>
    """, unsafe_allow_html=True)

    # ---------------- RECOMMENDATIONS ----------------
    st.markdown("### 💡 Health Recommendations")
    st.markdown(recs)

    # ---------------- DISCLAIMER ----------------
    st.warning("""
⚠️ **Medical Disclaimer:**  
This application is intended for educational and research purposes only.  
It does not provide medical diagnosis. Always consult a qualified healthcare professional.
    """)

# ---------------- MODEL DETAILS ----------------
st.markdown("---")
st.markdown("## 🧠 Model Details")

st.write("""
- This application uses a Machine Learning model trained on heart disease datasets.  
- It analyzes 13 clinical parameters to predict risk probability.  
- The model was optimized using cross-validation techniques.  
- The output is categorized into Low, Moderate, and High risk levels.  
""")