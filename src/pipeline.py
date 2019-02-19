import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def prep_df(df, drop_columns):
    """ Prepare dataframe for use in ML algorithm.
        1) We'll impute median values where needed.
        2) One-hot encode categorical features
        
        parameters:
        df - dataframe to clean up and prep for ML
        columns (list, array) - since we generally want to avoid 
            non-numerical columns, we'll have to specify which ones 
            to exclude. ToDo: Make this function auto-do this.
    """

    # Impute step
    imputer = SimpleImputer(strategy="median")
    df_imputted = df.drop(columns=drop_columns, axis=1)
    imputer.fit(df_imputted)
    X = imputer.transform(df_imputted)
    df_transformed = pd.DataFrame(X, columns=df_imputted.columns)

    # One-hot encoding




    return(df_transformed)
