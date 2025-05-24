"""
This is a boilerplate pipeline 'split_data'
generated using Kedro 0.19.12
"""


import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame):
    target = "y"
    test_size = 0.2
    random_state = 42

    X = df.drop(columns=[target])
    y = df[target]
    columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, columns
