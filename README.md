# 🩺 Multi-Model Clinical Diagnostic Benchmark & XAI Portal

A production-ready machine learning, benchmarking, and explainability pipeline that evaluates diverse model architectures (Tree-based vs. Linear) on the Breast Cancer Wisconsin dataset and leverages **SHAP (SHapley Additive exPlanations)** to deliver transparent, interpretable clinical insights.

## 🚀 Key Features
- **Layered Production Architecture:** Built using an industry-standard modular pattern separating data ingestion, multi-model lifecycle management, and explainability layers (`src/`).
- **Multi-Model Benchmarking:** Evaluates and compares three distinct classifiers: **Random Forest**, **XGBoost**, and **Logistic Regression** based on Precision, Recall, Accuracy, and F1-Score.
- **Dynamic Explainable AI (XAI):** Resolves the "black-box" dilemma of machine learning by dynamically routing traffic through appropriate SHAP explainers (`TreeExplainer` vs. `LinearExplainer`), providing patient-specific local waterfall attributions.
- **Interactive UI Dashboard:** Implemented an end-user interface via Streamlit allowing clinicians to seamlessly simulate patient features and observe shifting feature weights across different model backends.

## 📁 Repository Structure
```text
├── src/
│   ├── __init__.py
│   ├── data_loader.py   # Data ingestion, processing & stratification
│   ├── model.py         # Multi-model training lifecycle & benchmarking
│   └── explainer.py     # Dynamic SHAP explainer assignment & routing
├── app.py               # Streamlit Multi-Page UI Layer
└── requirements.txt     # Dependency management & profiling