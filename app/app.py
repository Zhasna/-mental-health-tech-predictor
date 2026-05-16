import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Mental Health in Tech — Treatment Predictor",
    layout="wide"
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(base_dir, 'model', 'rf_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)

with open(os.path.join(base_dir, 'model', 'encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

# Header
st.title("Mental Health Treatment Predictor")
st.markdown("*Will someone in the tech industry seek mental health treatment?*")
st.divider()

# Key stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", "85%")
with col2:
    st.metric("Respondents Analyzed", "4,200+")
with col3:
    st.metric("Survey Years", "2014 – 2019")

st.divider()

# Key finding callout
st.info("""
**Key Finding:** Personal mental health history — prior diagnosis, past disorders, 
and how symptoms affect work — is a far stronger predictor of seeking treatment 
than workplace factors like employer benefits or company size.
""")

st.divider()

# Feature importance chart
st.subheader("What predicts treatment-seeking behavior?")
st.caption("Top 10 most influential factors from the Random Forest model")

feature_names = [
    "Ever been diagnosed",
    "Had disorder in the past",
    "Disorder interferes (treated)",
    "Disorder interferes (untreated)",
    "Currently have disorder",
    "Family history",
    "Age",
    "Previous employer benefits",
    "Willingness to share with family",
    "Observed unsupportive response"
]
importance_values = [0.189, 0.143, 0.105, 0.102, 0.049, 0.033, 0.030, 0.022, 0.021, 0.018]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#2ecc71' if i < 4 else '#95a5a6' for i in range(len(feature_names))]
bars = ax.barh(feature_names[::-1], importance_values[::-1], color=colors[::-1])
ax.set_xlabel("Importance Score")
ax.set_facecolor('#0e1117')
fig.patch.set_facecolor('#0e1117')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.spines['bottom'].set_color('#444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#444')
st.pyplot(fig)

st.divider()

# Predictor form
st.subheader("Try it yourself")
st.caption("Fill in your profile and see what the model predicts")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**About you**")
    age = st.slider("Age", 18, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    current_disorder = st.selectbox("Currently have a mental health disorder?", ["Yes", "No", "Maybe"])
    past_disorder = st.selectbox("Had a mental health disorder in the past?", ["Yes", "No", "Maybe"])
    ever_diagnosed = st.selectbox("Ever been diagnosed with a mental health disorder?", ["Yes", "No"])

with col_right:
    st.markdown("**About your workplace**")
    self_employed = st.selectbox("Self-employed?", ["No", "Yes"])
    tech_company = st.selectbox("Employer primarily a tech company?", ["Yes", "No"])
    company_size = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    anonymity = st.selectbox("Anonymity protected if using mental health resources?", ["Yes", "No", "Don't know"])
    mental_health_benefits = st.selectbox("Employer provides mental health benefits?", ["Yes", "No", "Don't know"])
    primary_role = st.selectbox("Primary role related to tech/IT?", ["Yes", "No"])

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    input_dict = {
        'SurveyID': 2019,
        'Are you self-employed?': 1 if self_employed == "Yes" else 0,
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
    confidence = max(probability) * 100

    st.divider()
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            st.success("### Likely to seek mental health treatment")
        else:
            st.warning("### Unlikely to seek mental health treatment")

    with res_col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    # Probability bar
    st.markdown("**Prediction breakdown**")
    prob_df = pd.DataFrame({
        'Outcome': ['Unlikely to seek treatment', 'Likely to seek treatment'],
        'Probability': [probability[0] * 100, probability[1] * 100]
    })

    fig2, ax2 = plt.subplots(figsize=(8, 1.5))
    ax2.barh([''], [probability[0] * 100], color='#e74c3c', label='Unlikely')
    ax2.barh([''], [probability[1] * 100], left=[probability[0] * 100], color='#2ecc71', label='Likely')
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Probability (%)')
    ax2.set_facecolor('#0e1117')
    fig2.patch.set_facecolor('#0e1117')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.legend(loc='upper right', facecolor='#0e1117', labelcolor='white')
    st.pyplot(fig2)

    st.caption("This prediction is based on patterns in the OSMI Mental Health in Tech Survey and is not a clinical assessment.")

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85rem;'>
Built by Hasna Fahima · 
<a href='https://github.com/Zhasna/-mental-health-tech-predictor' style='color: gray;'>GitHub</a> · 
Data: OSMI Mental Health in Tech Survey (2014–2019)
</div>
""", unsafe_allow_html=True)