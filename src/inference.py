import streamlit as st
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """
    Simple winsorization for all numerical columns.
    Clips values outside [Q1-1.5*IQR, Q3+1.5*IQR].
    """

    def fit(self, X, y=None):
        self.lower_ = X.quantile(0.25) - 1.5 * (X.quantile(0.75) - X.quantile(0.25))
        self.upper_ = X.quantile(0.75) + 1.5 * (X.quantile(0.75) - X.quantile(0.25))
        return self

    def transform(self, X):
        return X.clip(self.lower_, self.upper_, axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


@st.cache_resource
def load_pipeline():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.joblib"
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

def predict(data):
    return pipeline.predict(data)