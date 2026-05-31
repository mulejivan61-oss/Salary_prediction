import streamlit as st
import joblib
import numpy as np

model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Salary Prediction App")

experience = st.number_input("Experience")
education = st.number_input("Education Level")
skills = st.number_input("Skills Count")
certifications = st.number_input("Certifications")
projects = st.number_input("Projects Done")
interview = st.number_input("Interview Score")

if st.button("Predict Salary"):

    if experience == 0 or education == 0 or skills == 0:
        st.warning("Enter valid values greater than 0")

    else:
        skill_efficiency = skills / experience

        features = np.array([[experience, education, skills,
                              certifications, projects,
                              interview, skill_efficiency]])

        scaled = scaler.transform(features)

        prediction = abs(model.predict(scaled)[0])

        st.success(f"Estimated Annual Salary: ₹{prediction:,.0f}")
