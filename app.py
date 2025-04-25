
import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('best_iris_model.pkl', 'rb'))

st.title("🌸 Iris Flower Species Classifier")
st.write("Enter flower measurements to predict the species.")

# Sliders for input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict on button click
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"The predicted species is: **{species[prediction]}** 🌼")
