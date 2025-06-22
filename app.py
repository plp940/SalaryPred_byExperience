import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('predict_salary.pkl')
scaler = joblib.load('scaler.pkl')
#designing the layout of the web app
st.set_page_config(page_title="Salary Prediction App", layout="wide")
#layout centered 
st.title("Salary Prediction App")
st.markdown("This app predicts the salary based on years of experience.")
st.write("Enter the number of years of experience to get the predicted salary.")
# dropdown for years of experience
#years_of_experience = st.selectbox("Select Years of Experience", options=np.arange(0, 21, 1))
years = [x for x in range(0, 21)]
years_of_experience = st.selectbox("Select Years of Experience", options=years)
# Predict the salary based on the input years of experience
if st.button("Predict Salary"):
    # Prepare the input data for prediction
    input_data = np.array([[years_of_experience]])
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction
    predicted_salary = model.predict(input_data_scaled)
    
    # Display the predicted salary
    st.success(f"The predicted salary for {years_of_experience} years of experience is ${predicted_salary[0][0]:,.2f}.")