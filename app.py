# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1️⃣ Load the saved Random Forest model and label encoders
with open('random_forest_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)

rf_model = saved_data['model']
le_dict = saved_data['label_encoders']

# 2️⃣ Streamlit page configuration
st.set_page_config(page_title="Student Performance Predictor 🎓", layout="centered")

st.title("Student Performance Predictor")
st.markdown("Predict if a student will **Pass** or **Fail** based on their study habits and profile.")

# 3️⃣ Input fields
gender = st.selectbox("Gender", options=le_dict['Gender'].classes_)
study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=100, value=15)
attendance_rate = st.number_input("Attendance Rate (%)", min_value=0, max_value=100, value=90)
past_scores = st.number_input("Past Exam Scores", min_value=0, max_value=100, value=10)
parent_education = st.selectbox("Parental Education Level", options=le_dict['Parental_Education_Level'].classes_)
internet_access = st.selectbox("Internet Access at Home", options=le_dict['Internet_Access_at_Home'].classes_)
extracurricular = st.selectbox("Extracurricular Activities", options=le_dict['Extracurricular_Activities'].classes_)
final_exam_score = st.number_input("Final Exam Score (if known, else 0)", min_value=0, max_value=100, value=0)

# 4️⃣ Apply log-transform to final exam score (same as training)
final_exam_score_log = np.log1p(final_exam_score)

# 5️⃣ Prepare input for prediction
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

# 6️⃣ Prediction button
if st.button("Predict"):
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0]
    
    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.success(f"✅ The student is likely to **Pass** (Probability: {probability[1]*100:.2f}%)")
    else:
        st.error(f"❌ The student is likely to **Fail** (Probability: {probability[0]*100:.2f}%)")
