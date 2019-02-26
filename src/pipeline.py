import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        return X[self.attribute_names].values


class DataFramePipeline():
    def __init__(self, cat_attribs, num_attribs):
        self.num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('std_scaler', StandardScaler()),
        ])

        self.cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('cat_encoder', OneHotEncoder(sparse=False)),
        ])

        self.full_pipeline = FeatureUnion(transformer_list=[
            #("num_pipeline", self.num_pipeline),
            ("cat_pipeline", self.cat_pipeline), 
        ])


def attrib_cats(df, orig_cat, new_cat, cut_points, labels):
    df[orig_cat] = df[orig_cat].fillna(-0.5)
    df[new_cat] = pd.cut(df[orig_cat], cut_points, labels=labels)
    return df