import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/pcos_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("PCOS Detection System")

age = st.number_input("Age")
bmi = st.number_input("BMI")
cycle = st.number_input("Cycle Length")

if st.button("Predict"):

    data = np.array([[age, bmi, cycle]])
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("High chances of PCOS")
    else:
        st.success("Low chances of PCOS")