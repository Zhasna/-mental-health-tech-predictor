import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(base_dir, 'model', 'rf_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)

with open(os.path.join(base_dir, 'model', 'encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

st.title("Mental Health Treatment Predictor")
st.write("Predicts whether someone in the tech industry is likely to seek mental health treatment based on personal and workplace factors.")

st.header("About you")
age = st.slider("What is your age?", 18, 80, 30)
gender = st.selectbox("What is your gender?", ["Male", "Female", "Other"])
family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
current_disorder = st.selectbox("Do you currently have a mental health disorder?", ["Yes", "No", "Maybe"])
past_disorder = st.selectbox("Have you had a mental health disorder in the past?", ["Yes", "No", "Maybe"])
ever_diagnosed = st.selectbox("Have you ever been diagnosed with a mental health disorder?", ["Yes", "No"])

st.header("About your workplace")
self_employed = st.selectbox("Are you self-employed?", ["0", "1"])
tech_company = st.selectbox("Is your employer primarily a tech company?", ["Yes", "No"])
company_size = st.selectbox("How many employees does your company have?",
    ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
anonymity = st.selectbox("Is your anonymity protected if you use mental health resources?", ["Yes", "No", "Don't know"])
mental_health_benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
primary_role = st.selectbox("Is your primary role related to tech/IT?", ["Yes", "No"])

if st.button("Predict"):
    input_dict = {
        'SurveyID': 2019,
        'Are you self-employed?': int(self_employed),
        'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?': 0,
        'Did your previous employers provide resources to learn more about mental health disorders and how to seek help?': 0,
        'Do you believe your productivity is ever affected by a mental health issue?': 1,
        'Do you currently have a mental health disorder?': current_disorder,
        'Do you feel that your employer takes mental health as seriously as physical health?': "I don't know",
        'Do you have a family history of mental illness?': family_history,
        'Do you have medical coverage (private insurance or state-provided) that includes treatment of mental health disorders?': 1,
        'Do you have previous employers?': 1,
        'Do you know local or online resources to seek help for a mental health issue?': 1,
        'Do you know the options for mental health care available under your employer-provided health coverage?': "I don't know",
        'Do you think that discussing a physical health issue with your employer would have negative consequences?': "Maybe",
        'Does your employer offer resources to learn more about mental health disorders and options for seeking help?': "I don't know",
        'Does your employer provide mental health benefits as part of healthcare coverage?': mental_health_benefits,
        'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?': "No",
        'Have you ever been diagnosed with a mental health disorder?': ever_diagnosed,
        'Have you had a mental health disorder in the past?': past_disorder,
        'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?': "No",
        'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?': "No",
        'Have your previous employers provided mental health benefits?': "No",
        'How many employees does your company or organization have?': company_size,
        'How willing would you be to share with friends and family that you have a mental illness?': "Somewhat open",
        'If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?': "Somewhat easy",
        'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?': "1-25%",
        'If you have a mental health disorder, how often do you feel that it interferes with your work when being treated effectively?': "Sometimes",
        'If you have a mental health disorder, how often do you feel that it interferes with your work when not being treated effectively (i.e., when you are experiencing symptoms)?': "Often",
        'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?': "No",
        'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?': "No",
        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': anonymity,
        'Is your employer primarily a tech company/organization?': tech_company,
        'Is your primary role within your company related to tech/IT?': primary_role,
        'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?': "I don't know",
        'Were you aware of the options for mental health care provided by your previous employers?': "No",
        'What is your age?': age,
        'What is your gender?': gender,
        'Would you be willing to bring up a physical health issue with a potential employer in an interview?': "Maybe",
        'Would you bring up a mental health issue with a potential employer in an interview?': "Maybe",
        'Would you feel comfortable discussing a mental health issue with your coworkers?': "Maybe",
        'Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?': "Maybe",
        'Would you have been willing to discuss your mental health with your direct supervisor(s)?': "Maybe"
    }

    input_df = pd.DataFrame([input_dict])

    for col in input_df.select_dtypes(include='object').columns:
        if col in encoders:
            le = encoders[col]
            input_df[col] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success("This person is **likely to seek** mental health treatment")
    else:
        st.warning("This person is **unlikely to seek** mental health treatment")

    st.metric("Confidence", f"{max(probability)*100:.1f}%")
    st.caption("This prediction is based on patterns in the OSMI Mental Health in Tech Survey dataset and is not a clinical assessment.")