import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        return X[self.attribute_names].values

def age_cats(df, cut_points, labels):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_cat"] = pd.cut(df["Age"], cut_points, labels=labels)
    return df