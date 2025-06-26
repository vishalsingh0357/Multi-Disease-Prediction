#Mutlti-Disease-Prediction
 Multi-Disease-PredictionThis project is a Streamlit-based web application for predicting two medical conditions — Diabetes and Heart Disease — using machine learning models trained on publicly available datasets (diabetic.csv and heart.csv).

Each prediction module (diabetic.py, heart.py) performs the following steps:

Loads and preprocesses the respective dataset

Trains a classification model (e.g., Logistic Regression or Random Forest)

Accepts user inputs via the UI for relevant health parameters

Outputs the prediction result (disease presence: Yes/No)

The front.py file integrates both models into a unified web interface, enabling users to select a prediction type and enter relevant input features.

The primary goal is to demonstrate end-to-end machine learning deployment, including data preprocessing, model building, and user interaction through a real-time web interface
