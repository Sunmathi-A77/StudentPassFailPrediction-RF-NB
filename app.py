# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# 1️⃣ Load saved Random Forest model and label encoders
with open('random_forest_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)

rf_model = saved_data['model']
le_dict = saved_data['label_encoders']

# 2️⃣ Page config
st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered", page_icon="🎯")

# 3️⃣ Custom CSS
st.markdown("""
<style>
/* Animated Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Input boxes & select boxes styling */
.stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div>div>span {
    background-color: #ffffffcc;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 16px;
    color: #333;
}

/* Gradient button with hover & active animation */
.stButton>button {
    background: linear-gradient(90deg, #36D1DC, #5B86E5);
    color: white;
    font-size: 18px;
    height: 50px;
    width: 100%;
    border-radius: 12px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
}
.stButton>button:active {
    transform: scale(0.98);
    box-shadow: 0px 3px 8px rgba(0,0,0,0.3);
}

/* Hide default footer */
footer {visibility:hidden;}

/* Card style for input sections */
.card {
    background: rgba(255,255,255,0.85);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# 4️⃣ Title & description
st.title("🎓 Student Performance Predictor")
st.markdown("Predict if a student will **Pass** or **Fail** based on study habits and profile.")

# 5️⃣ Input sections with card style

# Personal Info Card
gender = st.selectbox("👤 Gender", options=le_dict['Gender'].classes_)
study_hours = st.number_input("⏰ Study Hours per Week", min_value=0, max_value=100, value=15)
attendance_rate = st.number_input("📅 Attendance Rate (%)", min_value=0, max_value=100, value=90)
past_scores = st.number_input("📊 Past Exam Scores", min_value=0, max_value=100, value=10)

# Family & Resources Card
parent_education = st.selectbox("🎓 Parental Education Level", options=le_dict['Parental_Education_Level'].classes_)
internet_access = st.selectbox("📶 Internet Access at Home", options=le_dict['Internet_Access_at_Home'].classes_)
extracurricular = st.selectbox("🏅 Extracurricular Activities", options=le_dict['Extracurricular_Activities'].classes_)
final_exam_score = st.number_input("📝 Final Exam Score (if known, else 0)", min_value=0, max_value=100, value=0)

# 6️⃣ Apply log-transform
final_exam_score_log = np.log1p(final_exam_score)

# 7️⃣ Prepare input for prediction
input_data = pd.DataFrame([{
    'Gender': le_dict['Gender'].transform([gender])[0],
    'Study_Hours_per_Week': study_hours,
    'Attendance_Rate': attendance_rate,
    'Past_Exam_Scores': past_scores,
    'Parental_Education_Level': le_dict['Parental_Education_Level'].transform([parent_education])[0],
    'Internet_Access_at_Home': le_dict['Internet_Access_at_Home'].transform([internet_access])[0],
    'Extracurricular_Activities': le_dict['Extracurricular_Activities'].transform([extracurricular])[0],
    'Final_Exam_Score': final_exam_score_log
}])

# 8️⃣ Predict button with Pass/Fail and probability
if st.button("Predict"):
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0]

    # Animated progress bar
    if prediction == 1:
        st.success("🎉 Congrats! The student has Passed")
        st.balloons()
        prog = st.progress(0)
        for i in range(int(probability[1]*100)+1):
            prog.progress(i)
            time.sleep(0.01)
        st.markdown(f"<h2 style='color:green'>Pass</h2>", unsafe_allow_html=True)
        st.info(f"✅ Pass Probability: {probability[1]*100:.2f}%")
    else:
        st.error("⚠️ The student has Failed. Keep trying!")
        prog = st.progress(0)
        for i in range(int(probability[0]*100)+1):
            prog.progress(i)
            time.sleep(0.01)
        st.markdown(f"<h2 style='color:red'>Fail</h2>", unsafe_allow_html=True)
        st.info(f"❌ Fail Probability: {probability[0]*100:.2f}%")

# 9️⃣ Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: gray; font-size:14px;'>"
    "🎓 Student Performance Predictor | Created by <b>Sunmathi</b> ❤️ | Powered by Random Forest"
    "</p>",
    unsafe_allow_html=True
)
