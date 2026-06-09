import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import MedicalDataLoader
from src.model import DiagnosticModel
from src.explainer import XAIManager

st.set_page_config(page_title="XAI Cancer Diagnostic Portal", layout="wide")

st.title("🩺 Explainable AI (XAI) Breast Cancer Diagnostic Portal")
st.write("This production-ready portal leverages a Layered Architecture to deliver robust ML predictions backed by SHAP explanations.")

@st.cache_resource
def initialize_backend():
    data_pipeline = MedicalDataLoader()
    X_train, X_test, y_train, y_test, target_names = data_pipeline.load_breast_cancer_data()
    
    classifier = DiagnosticModel()
    trained_rf = classifier.train(X_train, y_train)
    
    xai_engine = XAIManager(trained_rf)
    return trained_rf, X_test, xai_engine, target_names

model, X_test, xai_engine, target_names = initialize_backend()

st.sidebar.header("🔬 Patient Clinical Features")
st.sidebar.write("Simulate a patient profile by adjusting clinical bounds:")

base_features = X_test.mean().to_dict()
base_features['mean radius'] = st.sidebar.slider("Mean Radius", 5.0, 30.0, float(X_test['mean radius'].mean()))
base_features['mean texture'] = st.sidebar.slider("Mean Texture", 5.0, 40.0, float(X_test['mean texture'].mean()))
base_features['mean perimeter'] = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, float(X_test['mean perimeter'].mean()))
base_features['mean area'] = st.sidebar.slider("Mean Area", 100.0, 2500.0, float(X_test['mean area'].mean()))

input_df = pd.DataFrame([base_features])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔮 Diagnostic Prediction")
    if st.button("Run Diagnostic Analysis", type="primary"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        if prediction == 1:
            st.success(f"Result: **BENIGN (İyi Huylu)**")
            st.metric(label="Confidence Level", value=f"{probability[1]*100:.2f}%")
        else:
            st.error(f"Result: **MALIGNANT (Kötü Huylu)**")
            st.metric(label="Confidence Level", value=f"{probability[0]*100:.2f}%")

with col2:
    st.subheader("📊 Explainable AI Insights")
    if st.button("Generate SHAP Explanation"):
        with st.spinner("Computing Shapley values across the Random Forest ensemble..."):
            shap_values_input = xai_engine.compute_shap_values(input_df)
            fig = xai_engine.generate_waterfall_plot(shap_values_input, sample_index=0)
            st.pyplot(fig)