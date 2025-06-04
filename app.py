import streamlit as st
import pickle
import joblib
import torch
from ultralytics import YOLO
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
from PIL import Image

st.title("Multi-Model Streamlit App")

model_type = st.sidebar.selectbox("Choose model type", ["Classification (.pkl)", "Keras (.json + .h5)", "YOLO Object Detection (.pt)"])

# ----- 1. Scikit-learn or XGBoost (.pkl) -----
if model_type == "Classification (.pkl)":
    model = joblib.load("models/extra_trees_model.pkl")  # or pickle.load(open("model.pkl", "rb"))
    st.success("Scikit-learn or XGBoost model loaded!")

    age = st.slider("Age", 0, 100, 30)
    tsh = st.number_input("TSH", 0.0, 100.0, 1.2)
    t3 = st.number_input("T3", 0.0, 10.0, 2.5)
    input_data = np.array([[age, tsh, t3]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write("Prediction:", prediction[0])

# ----- 2. Keras JSON Model (.json + .h5) -----
elif model_type == "Keras (.json + .h5)":
    with open("models/xgboost.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("models/model_weights.h5")
    st.success("Keras model loaded!")

    val1 = st.number_input("Feature 1")
    val2 = st.number_input("Feature 2")
    input_data = np.array([[val1, val2]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write("Prediction:", prediction)

# ----- 3. YOLO Object Detection (.pt) -----
elif model_type == "YOLO Object Detection (.pt)":
    yolo_model_path = st.selectbox("Select YOLO model", ["models/best_yolo11.pt", "models/best_yolo8.pt"])
    model = YOLO(yolo_model_path)
    st.success(f"{yolo_model_path} loaded!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            results = model.predict(image)
            res_img = results[0].plot()
            st.image(res_img, caption="Detection Result", use_column_width=True)
