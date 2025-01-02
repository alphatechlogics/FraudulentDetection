import streamlit as st
import pickle
import time
import numpy as np

st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Load model
with open("fraud_detection.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Input fields for each feature (Principal components and Time, Amount)
principal_component_1 = st.number_input("Principal Component 1", value=0.0)
principal_component_2 = st.number_input("Principal Component 2", value=0.0)
principal_component_3 = st.number_input("Principal Component 3", value=0.0)
principal_component_4 = st.number_input("Principal Component 4", value=0.0)
principal_component_5 = st.number_input("Principal Component 5", value=0.0)
principal_component_6 = st.number_input("Principal Component 6", value=0.0)
principal_component_7 = st.number_input("Principal Component 7", value=0.0)


# Time and Amount fields
time_ = st.number_input("Time (seconds since first transaction)", value=0.0)
amount = st.number_input("Amount", value=0.0)

# Create a list of input features (matching the model's expected features)
input_features = [
    principal_component_1, principal_component_2, principal_component_3, 
    principal_component_4, principal_component_5, principal_component_6, 
    principal_component_7, 
    time_, amount
]

# Submit button
submit = st.button("Submit")

if submit:
    start = time.time()
    # Reshape input_features to match the expected input shape for the model
    prediction = model.predict([input_features])
    end = time.time()
    
    st.write("Prediction time taken: ", round(end - start, 2), "seconds")
    
    if prediction[0] == 1:
        st.write("Fraudulent transaction")
    else:
        st.write("Legitimate transaction")
