import shap
import matplotlib.pyplot as plt

class XAIManager:
    """Farklı model mimarilerine göre uygun SHAP patikalarını seçen katman."""
    def __init__(self):
        pass
        
    def get_explainer(self, model, model_name):
        """Modelin türüne göre doğru SHAP Explainer nesnesini döner."""
        if model_name in ["Random Forest", "XGBoost"]:
            return shap.TreeExplainer(model)
        elif model_name == "Logistic Regression":
            return shap.LinearExplainer(model, masker=shap.maskers.Independent)
        else:
            return shap.Explainer(model)

    def compute_shap_values(self, explainer, X_data):
        """Seçilen explainer ile SHAP değerlerini hesaplar."""
        return explainer(X_data)
        
    def generate_waterfall_plot(self, shap_values, sample_index=0, class_index=1):
        """Tek bir hasta için yerel şelale grafiğini üretir."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # SHAP çıktı formatı model türüne göre (2D veya 3D) değişebilir, bunu standartlaştırıyoruz
        if len(shap_values.shape) == 3:
            shap_values_class = shap_values[:, :, class_index]
        else:
            shap_values_class = shap_values
            
        shap.plots.waterfall(shap_values_class[sample_index], show=False)
        plt.tight_layout()
        return fig