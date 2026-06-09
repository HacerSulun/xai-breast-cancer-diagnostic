import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import MedicalDataLoader
from src.model import DiagnosticModelManager
from src.explainer import XAIManager

st.set_page_config(page_title="XAI Cancer Benchmark Portal", layout="wide")

st.title("🩺 Multi-Model Diagnostic Benchmark & XAI Portal")
st.write("Compare different ML architectures (Tree-based vs. Linear) and see how their feature importance profiles shift using SHAP.")

@st.cache_resource
def initialize_backend():
    data_pipeline = MedicalDataLoader()
    X_train, X_test, y_train, y_test, target_names = data_pipeline.load_breast_cancer_data()
    
    # Tüm modelleri eğit
    manager = DiagnosticModelManager()
    trained_models = manager.train_all(X_train, y_train)
    comparison_df = manager.compare_performance(X_test, y_test)
    
    xai_engine = XAIManager()
    return trained_models, comparison_df, X_test, xai_engine, target_names

trained_models, comparison_df, X_test, xai_engine, target_names = initialize_backend()

# --- PANEL 1: MODEL PERFORMANS KARŞILAŞTIRMASI ---
st.subheader("📊 Model Performance Benchmark")
st.dataframe(comparison_df.style.highlight_max(axis=0, color="#b3e6b3", subset=["Accuracy", "F1-Score"]))

st.markdown("---")

# Yan Menü (Sidebar)
st.sidebar.header("⚙️ Simulation Settings")

# Kullanıcı hangi modeli analiz etmek istiyor?
selected_model_name = st.sidebar.selectbox("Select Model for XAI Analysis", list(trained_models.keys()))
selected_model = trained_models[selected_model_name]

st.sidebar.subheader("🔬 Patient Clinical Features")
base_features = X_test.mean().to_dict()
base_features['mean radius'] = st.sidebar.slider("Mean Radius", 5.0, 30.0, float(X_test['mean radius'].mean()))
base_features['mean texture'] = st.sidebar.slider("Mean Texture", 5.0, 40.0, float(X_test['mean texture'].mean()))
base_features['mean perimeter'] = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, float(X_test['mean perimeter'].mean()))
base_features['mean area'] = st.sidebar.slider("Mean Area", 100.0, 2500.0, float(X_test['mean area'].mean()))

input_df = pd.DataFrame([base_features])

# --- PANEL 2: SEÇİLEN MODELİN ANALİZİ ---
st.subheader(f"🔍 Active Model Analysis: {selected_model_name}")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("### 🔮 Prediction")
    if st.button("Run Diagnostic", type="primary"):
        prediction = selected_model.predict(input_df)[0]
        probability = selected_model.predict_proba(input_df)[0]
        
        if prediction == 1:
            st.success(f"Result: **BENIGN**")
            st.metric(label="Confidence", value=f"{probability[1]*100:.2f}%")
        else:
            st.error(f"Result: **MALIGNANT**")
            st.metric(label="Confidence", value=f"{probability[0]*100:.2f}%")

with col2:
    st.write("### 📊 XAI Explanation")
    if st.button("Generate SHAP"):
        with st.spinner(f"Computing weights via {selected_model_name} Explainer..."):
            # Model türüne göre dinamik explainer al
            explainer = xai_engine.get_explainer(selected_model, selected_model_name)
            shap_values_input = xai_engine.compute_shap_values(explainer, input_df)
            
            fig = xai_engine.generate_waterfall_plot(shap_values_input, sample_index=0)
            st.pyplot(fig)