from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class DiagnosticModel:
    """Makine öğrenmesi model yönetimi ve değerlendirme katmanı."""
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
    def train(self, X_train, y_train):
        """Modeli eğitir."""
        self.model.fit(X_train, y_train)
        return self.model
        
    def evaluate(self, X_test, y_test, target_names):
        """Model performans metriklerini döner."""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=target_names)
        return accuracy, report