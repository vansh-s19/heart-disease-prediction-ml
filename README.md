❤️ Heart Disease Prediction using Machine Learning

This project implements a Heart Disease Prediction System using Logistic Regression in Python.
The model is trained on a structured medical dataset to predict whether a person is likely to have heart disease based on clinical parameters.

⸻

📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help in timely medical intervention.
This project demonstrates how machine learning can be used to classify patients as healthy or having heart disease using medical attributes.

⸻

📂 Dataset Information
	•	Dataset contains medical attributes such as:
	•	Age
	•	Sex
	•	Chest pain type
	•	Resting blood pressure
	•	Cholesterol
	•	Fasting blood sugar
	•	Resting ECG results
	•	Maximum heart rate achieved
	•	Exercise-induced angina
	•	ST depression
	•	Slope of ST segment
	•	Number of major vessels
	•	Thalassemia

Target Variable
	•	0 → Healthy Heart
	•	1 → Defective Heart


⸻

🛠️ Technologies Used
	•	Python 3
	•	NumPy
	•	Pandas
	•	Scikit-Learn

⸻

⚙️ Project Workflow
	1.	Data Collection
	•	Load dataset using Pandas
	2.	Data Preprocessing
	•	Separate features and target
	•	Train-test split with stratification
	3.	Model Training
	•	Logistic Regression model
	4.	Model Evaluation
	•	Accuracy score on training and test data
	5.	Prediction System
	•	Accepts new patient data
	•	Predicts presence of heart disease

⸻

📁 Project Structure
Heart Disease Prediction/
│
├── data/
│   └── heart_disease_data.csv
│
├── model/
│   └── heart_diease_prediction.py
│
├── requirements.txt
│
└── README.md


⸻

🚀 How to Run the Project

1️⃣ Clone the Repository
git clone <your-repo-link>
cd Heart-Disease-Prediction

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Model
python model/heart_diease_prediction.py

____

🔍 Sample Input for Prediction
input_data = (57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2)
Output
	•	The person does not have a heart disease
	•	OR
	•	The person has heart disease


⸻

📊 Model Performance
	•	Training Accuracy: ~ High accuracy
	•	Testing Accuracy: ~ Reliable performance

(Exact values depend on dataset split)

⸻

🧠 Machine Learning Algorithm Used

Logistic Regression
	•	Suitable for binary classification
	•	Fast and interpretable
	•	Commonly used in medical prediction tasks

📌 Future Improvements
	•	Add user input support (CLI or Web App)
	•	Feature scaling and hyperparameter tuning
	•	Try advanced models (Random Forest, XGBoost)
	•	Deploy using Flask or Streamlit

⸻

🙌 Author

Vansh
Machine Learning & Data Science Enthusiast

⸻

⭐ Acknowledgment

Dataset sourced from publicly available heart disease datasets used for educational purposes.

⸻


