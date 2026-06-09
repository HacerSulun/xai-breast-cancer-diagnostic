import shap
import matplotlib.pyplot as plt

class XAIManager:
    """Model kararlarını SHAP ile şeffaflaştıran açıklanabilirlik katmanı."""
    def __init__(self, trained_model):
        self.explainer = shap.TreeExplainer(trained_model)
        
    def compute_shap_values(self, X_data):
        """Verilen veri seti için SHAP değerlerini hesaplar."""
        return self.explainer(X_data)
        
    def generate_waterfall_plot(self, shap_values, sample_index=0, class_index=1):
        """Tek bir hasta için yerel (local) kararın şelale grafiğini üretir."""
        fig, ax = plt.subplots(figsize=(10, 6))
        shap_values_class = shap_values[:, :, class_index]
        shap.plots.waterfall(shap_values_class[sample_index], show=False)
        plt.tight_layout()
        return fig