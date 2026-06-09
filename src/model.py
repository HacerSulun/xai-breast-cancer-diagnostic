from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

class DiagnosticModelManager:
    """Birden fazla ML modelini eğiten ve karşılaştıran katman."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss'),
            "Logistic Regression": LogisticRegression(max_iter=5000, random_state=random_state)
        }
        self.trained_models = {}

    def train_all(self, X_train, y_train):
        """Tanımlı tüm modelleri eğitir."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        return self.trained_models

    def compare_performance(self, X_test, y_test):
        """Tüm modellerin performans metriklerini bir DataFrame olarak döner."""
        metrics_list = []
        for name, model in self.trained_models.items():
            preds = model.predict(X_test)
            metrics_list.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds),
                "Recall": recall_score(y_test, preds),
                "F1-Score": f1_score(y_test, preds)
            })
        return pd.DataFrame(metrics_list)