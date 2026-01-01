# 🎓 Student Performance Prediction System (ML + AWS)

## 📌 Project Overview
This project is an end-to-end Machine Learning application that predicts whether a student will **Pass or Fail** based on academic performance and demographic features.  
The trained ML model is deployed as an interactive **Streamlit web application** on **AWS EC2**.

---

## 🎯 Problem Statement
Educational institutions often need early indicators of student performance to provide timely academic support.  
This system predicts student outcomes using historical academic data and presents predictions through a cloud-hosted web interface.

---

## 🧠 Solution Approach
1. Data preprocessing and feature engineering  
2. Training and evaluation of ML models  
3. Model selection and serialization  
4. Streamlit-based web application development  
5. Cloud deployment using AWS EC2  

---

## 🗂️ Dataset
- **Source:** Kaggle – Students Performance in Exams  
- **Features:**
  - Gender
  - Race/Ethnicity
  - Parental education
  - Lunch type
  - Test preparation course
  - Math, Reading, Writing scores
- **Target:** Pass / Fail classification

---

## ⚙️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- AWS EC2  
- Git & GitHub  

---

## 🏗️ Project Architecture
User Input
↓
Streamlit UI
↓
Preprocessing (Encoders + Scaler)
↓
Trained ML Model
↓
Prediction (Pass / Fail)

yaml
Copy code

---

## 📁 Project Structure
student-performance-ml/
│
├── data/ # Dataset (ignored in GitHub)
├── model/ # Trained model files
├── notebooks/ # EDA notebook
├── src/ # Training scripts
├── Output/ # Deployment screenshots
├── app.py # Streamlit application
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy code

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/Sarathh03/student-performance-ml.git
cd student-performance-ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
☁️ Cloud Deployment
Deployed on AWS EC2 (Ubuntu)

Used Python virtual environment for dependency isolation

Configured EC2 Security Group to allow Streamlit traffic (port 8501)

📊 Model Performance
Tested Logistic Regression and Random Forest

Selected Random Forest Classifier

Achieved high accuracy on test data

