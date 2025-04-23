import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("linear")

# Streamlit app UI
st.title("Linear Regression Predictor")
st.write("This app predicts a target value using a linear regression model.")

# Input slider for user input
feature_value = st.slider("Select input feature value", -3.0, 3.0, 0.0, step=0.1)

# Make prediction
prediction = model.predict(np.array([[feature_value]]))[0]
st.write(f"### Predicted value: {prediction:.2f}")

# Optional: Show plot
if st.checkbox("Show regression plot"):
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.plot(X, y_pred, color="red", label="Regression line")
    ax.scatter([feature_value], [prediction], color="blue", label="Your input")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Prediction")
    ax.legend()
    st.pyplot(fig)
