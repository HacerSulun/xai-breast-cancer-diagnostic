import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class MedicalDataLoader:
    """Medikal veri setlerini yükleme ve ön işleme katmanı."""
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        
    def load_breast_cancer_data(self):
        """Breast Cancer Wisconsin veri setini yükler ve Train/Test olarak böler."""
        cancer_data = load_breast_cancer()
        X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
        y = cancer_data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        return X_train, X_test, y_train, y_test, cancer_data.target_names