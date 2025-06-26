import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression

diabetes_model = pickle.load(open('F:/PythonProject/diabetes_model_save.sav', 'rb'))
heart_disease_model = pickle.load(open('F:/PythonProject/heart_disease_model_save.sav', 'rb'))

with st.sidebar:
    selected = option_menu('multiple disease prediction',
                           ['diabetic prediction',
                            'heart prediction'],
                           default_index=0)

if (selected == 'diabetic prediction'):
    # page title

    st.title('diabetic prediction using ml')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

        diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 0:
            diab_diagnosis = 'The person  not diabetic'
        else:
            diab_diagnosis = 'The person is  diabetic'

    st.success(diab_diagnosis)

if (selected == 'heart prediction'):
    st.title('heart disease prediction using ml')

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age')

    with col2:
        Sex = st.text_input('Sex')

    with col3:
        Chest_pain_type = st.text_input('Chest Pain types')

    with col1:
        BP = st.text_input('Resting Blood Pressure')

    with col2:
        Cholestrol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        FBS_over_120 = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        EKG_results = st.text_input('Resting Electrocardiographic results')

    with col2:
        MAX_HR = st.text_input('Maximum Heart Rate achieved')

    with col3:
        Exercise_angina = st.text_input('Exercise Induced Angina')

    with col1:
        ST_depression = st.text_input('ST depression induced by exercise')

    with col2:
        Slope_of_ST = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        Number_of_vessels_fluro = st.text_input('Major vessels colored by flourosopy')

    with col1:
        Thallium = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [Age, Sex, Chest_pain_type, BP, Cholestrol, FBS_over_120, EKG_results, MAX_HR, Exercise_angina,
                      ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)






