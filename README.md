# 🎓 Student Performance Predictor

## Project Overview

The Student Performance Predictor is a web application built with Streamlit that predicts whether a student is likely to Pass or Fail based on their study habits, attendance, parental education, and other factors. The model is trained using Random Forest Classifier and displays probability scores with a dynamic, interactive interface.

🌐 Live App: https://studentpassfailprediction-rf-nb.streamlit.app/

![WhatsApp Image 2025-10-16 at 23 29 31_691244a9](https://github.com/user-attachments/assets/5e871ba7-8f80-4580-a43c-766b8e11dd7b)

![WhatsApp Image 2025-10-16 at 23 29 31_ce880332](https://github.com/user-attachments/assets/6a179411-0724-4937-921c-604a40082a45)

## 🛠️ Technologies Used

Python 3.x – programming language

Pandas, NumPy – data manipulation and preprocessing

Scikit-learn – machine learning (Random Forest Classifier, Gaussian Naive Bayes)

Seaborn, Matplotlib – data visualization

Streamlit – interactive web application

Pickle – saving trained ML model

## 📂 Dataset

Dataset Source: Student Performance on Kaggle - https://www.kaggle.com/datasets/amrmaree/student-performance-prediction

### Features Used:

Study_Hours_per_Week

Attendance_Rate

Past_Exam_Scores

Parental_Education_Level

Internet_Access_at_Home

Extracurricular_Activities

Final_Exam_Score

Pass_Fail (target variable)

## 🔧 Installation Steps

### Clone the repository:

git clone <your-repo-link>
cd <your-repo-name>

### Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

### Install dependencies:

pip install -r requirements.txt

### Run the Streamlit app:

streamlit run app.py

## ⚡ Steps Performed

### Data Cleaning & Preprocessing:

Dropped irrelevant columns (Student_ID).

Encoded categorical variables using LabelEncoder.

Checked skewness and applied log-transform to Final_Exam_Score.

Checked correlation and outliers for numeric columns.

### Model Training:

Split dataset into training and testing sets.

Trained Random Forest Classifier and Naive Bayes.

Evaluated accuracy, precision, recall, F1-score, and confusion matrices.

### Model Saving:

Saved Random Forest model using pickle along with label encoders.

### App Development:

Created interactive UI with animated background, colored inputs, and selectboxes.

Added dynamic prediction results with probability bars.

Added footer with creator info.

## 📊 Model Results

Model	          Accuracy	Precision	  Recall	   F1-Score

Random Forest	  1.00	    1.00	      1.00	    1.00

Naive Bayes	    0.98	    0.98	      0.98	    0.98

Random Forest Confusion Matrix:

[[75  0]
 [ 0 25]]

Random Forest is chosen for deployment due to perfect accuracy and robustness.

## 📁 Project Structure

├── requirements.txt        # Dependencies
├── dataset.csv             # Kaggle dataset (optional)
├── random_forest_model.pkl # Saved Random Forest model
├── app.py                  # Streamlit web app
├── README.md               # Documentation
