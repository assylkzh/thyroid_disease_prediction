import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from catboost import CatBoostClassifier
import xgboost as xgb

# Label mapping
label_map = {
    0: 'Hyperthyroid',
    1: 'Hypothyroid',
    2: 'Negative',
    3: 'Non-thyroidal Illness',
    4: 'Replacement Therapy',
    5: 'Treatment Effect'
}

st.set_page_config(page_title="Thyroid AI & YOLO", layout="wide")
st.title("🧠 Thyroid Detection & YOLO App")

model_type = st.sidebar.selectbox("Select Model Type", [
    "Classification (.pkl/.json)",
    "YOLO Object Detection (.pt)"
])

def binarize(val): return 1 if val == "t" else 0

def get_patient_input():
    age = st.slider("Age", 0, 100, 30, help="Patient's age in years")
    sex = st.selectbox("Sex", ["F", "M"])

    def tf(label): return st.selectbox(label, ["f", "t"])

    on_thyroxine = tf("On thyroxine?")
    query_on_thyroxine = tf("Query on thyroxine?")
    on_antithyroid_meds = tf("On antithyroid meds?")
    sick = tf("Sick?")
    pregnant = tf("Pregnant?")
    thyroid_surgery = tf("Thyroid surgery?")
    I131_treatment = tf("I131 treatment?")
    query_hypothyroid = tf("Query hypothyroid?")
    query_hyperthyroid = tf("Query hyperthyroid?")
    lithium = tf("Lithium?")
    goitre = tf("Goitre?")
    tumor = tf("Tumor?")
    hypopituitary = tf("Hypopituitary?")
    psych = tf("Psych?")
    TSH_measured = tf("TSH measured?")
    T3_measured = tf("T3 measured?")
    TT4_measured = tf("TT4 measured?")
    T4U_measured = tf("T4U measured?")
    FTI_measured = tf("FTI measured?")

    TSH = st.number_input("TSH", 0.0, 100.0, 1.2)
    T3 = st.number_input("T3", 0.0, 10.0, 2.5)
    TT4 = st.number_input("TT4", 0.0, 300.0, 120.0)
    T4U = st.number_input("T4U", 0.0, 5.0, 1.0)
    FTI = st.number_input("FTI", 0.0, 200.0, 110.0)

    input_data = {
        'age': age,
        'sex': 0 if sex == "F" else 1,
        'on_thyroxine': binarize(on_thyroxine),
        'query_on_thyroxine': binarize(query_on_thyroxine),
        'on_antithyroid_meds': binarize(on_antithyroid_meds),
        'sick': binarize(sick),
        'pregnant': binarize(pregnant),
        'thyroid_surgery': binarize(thyroid_surgery),
        'I131_treatment': binarize(I131_treatment),
        'query_hypothyroid': binarize(query_hypothyroid),
        'query_hyperthyroid': binarize(query_hyperthyroid),
        'lithium': binarize(lithium),
        'goitre': binarize(goitre),
        'tumor': binarize(tumor),
        'hypopituitary': binarize(hypopituitary),
        'psych': binarize(psych),
        'TSH_measured': binarize(TSH_measured),
        'TSH': TSH,
        'T3_measured': binarize(T3_measured),
        'T3': T3,
        'TT4_measured': binarize(TT4_measured),
        'TT4': TT4,
        'T4U_measured': binarize(T4U_measured),
        'T4U': T4U,
        'FTI_measured': binarize(FTI_measured),
        'FTI': FTI
    }
    return pd.DataFrame([input_data])

# Cache model loading to speed up repeated predictions
@st.cache_resource
def load_model(model_name):
    if model_name == "Extra Trees":
        return joblib.load("models/extra_trees_model.pkl")
    elif model_name == "Random Forest":
        return joblib.load("models/random_forest.pkl")
    elif model_name == "LightGBM":
        return joblib.load("models/light_gbm.pkl")
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier()
        model.load_model("models/xgboost_model.json")
        return model
    elif model_name == "CatBoost":
        model = CatBoostClassifier()
        model.load_model("models/catboost.json")
        return model
    return None

# ---- Classification ----
if model_type == "Classification (.pkl/.json)":
    model_name = st.selectbox("Choose Model", [
        "Extra Trees", "Random Forest", "LightGBM", "XGBoost", "CatBoost"
    ])
    input_df = get_patient_input()

    model = load_model(model_name)

    if st.button("🔮 Predict"):
        try:
            prediction = model.predict(input_df)
            pred_label = label_map.get(prediction[0], "Unknown")
            st.success(f"Prediction: {pred_label}")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                prob_df = pd.DataFrame({
                    'Class': list(label_map.values()),
                    'Probability': probs
                })
                st.table(prob_df.style.format({"Probability": "{:.2%}"}))
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ---- YOLO Object Detection ----
elif model_type == "YOLO Object Detection (.pt)":
    yolo_model_path = st.selectbox("Choose YOLO model", [
        "models/best_yolo8.pt", "models/best_yolo11.pt"
    ])
    model = YOLO(yolo_model_path)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("🧠 Run YOLO Detection"):
            results = model.predict(image)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Detection Result", use_column_width=True)
            count = len(results[0].boxes)
            st.write(f"Detected objects: {count}")
