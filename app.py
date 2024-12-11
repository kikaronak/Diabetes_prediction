import numpy as np
import pickle
import joblib
import streamlit as st

# Load the trained model
with open('trained_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the scaler
scaler = joblib.load('scaler.pkl')  # Assuming you saved the scaler as 'scaler.pkl'

# Function to make predictions
def predict_diabetes(input_data):
    input_as_np = np.asarray(input_data, dtype=float)
    input_reshaped = input_as_np.reshape(1, -1)  # Reshape for single input
    input_scaled = scaler.transform(input_reshaped)
    st.write(f"Scaled Input: {input_scaled}")
    prediction = loaded_model.predict(input_scaled)

    # Debugging: Print the prediction to check the value
    st.write(f"Model Prediction (Raw): {prediction}")

    # If prediction is an array, get the first element
    prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction

    # Debugging: Print the prediction value after conversion
    st.write(f"Prediction Value (After Conversion): {prediction_value}")

    if prediction_value == 0:
        return 'The person is not diabetic.'
    elif prediction_value == 1:
        return 'The person is diabetic.'
    else:
        return 'Invalid prediction.'

# Main function for Streamlit web app
def main():
    # Streamlit app title
    st.title("Diabetes Prediction Web App")

    # Taking input from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BMI = st.text_input("BMI")
    Age = st.text_input("Age")

    # Button for making predictions
    if st.button("Predict Diabetes"):
        try:
            # Convert input to float for prediction
            input_data = [float(Pregnancies), float(Glucose), float(BMI), float(Age)]
            # Call the prediction function
            diagnosis = predict_diabetes(input_data)
            st.success(diagnosis)  # Display the result
        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")

if __name__ == '__main__':
    main()
