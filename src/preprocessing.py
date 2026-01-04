from sklearn.base import BaseEstimator, TransformerMixin

class IQRWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (
                Q1 - self.factor * IQR,
                Q3 + self.factor * IQR
            )
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.bounds_.items():
            X[col] = X[col].clip(low, high)
        return X
