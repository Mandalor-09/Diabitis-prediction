from helper import *
import streamlit
import streamlit as st
import os
from dotenv import load_dotenv
import pathlib
import textwrap
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro-vision')

def main(prediction):
    st.title('Diabetes Analysis')

    analyze_diet = False  # Flag to control analysis of diet

    st.subheader('Diabetes Prediction')
    col1 ,col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input("Pregnancies count:")
        BloodPressure = st.text_input("BloodPressure count:")
        Insulin = st.text_input("Insulin count:")
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction count:")
    with col2:
        Glucose = st.text_input("Glucose count:")
        SkinThickness = st.text_input("SkinThickness count:")
        BMI = st.text_input("BMI count:")
        Age = st.text_input("Age count:")

    if st.button("Check Prediction", key='prediction_button', type="primary"):
        if all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            predicted, parameters = prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            st.write(predicted)
            st.write('Do you wish to analyze your diet?') 
            analyze_diet = st.button("Yes")
        else:
            st.warning("Please fill in all input values before submitting.")

    # File uploader for uploading image
    photo = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
    if photo is not None:
        st.image(photo, caption="Uploaded Image", use_column_width=True)
        
        # Perform diet analysis if necessary
        if analyze_diet:
            analysis_diet(photo)

if __name__ == "__main__":
    main(prediction=prediction)









    