from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]

class MedianImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with median for numeric columns."""
    def __init__(self):
        self.medians_ = None
    
    def fit(self, X, y=None):
        """Compute medians for each numeric column."""
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            self.medians_ = X[numeric_cols].median().to_dict()
        else:
            # For numpy arrays, compute median along axis 0
            self.medians_ = np.median(X, axis=0)
        return self
    
    def transform(self, X):
        """Impute missing values with stored medians."""
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for col, median_val in self.medians_.items():
                if col in X.columns:
                    X[col] = X[col].fillna(median_val)
        else:
            # For numpy arrays
            X = np.where(np.isnan(X), self.medians_, X)
        return X
