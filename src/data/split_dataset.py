# Importing all dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from typing import List
import pickle


# Splitting dataset into train and test datasets
def split_dataset(df: pd.DataFrame, test_size: float, random_state: int) -> None:
    X = df.drop(["TenYearCHD"], axis=1)
    y = df["TenYearCHD"]
    feat_col = X.columns
    scaler = preprocessing.StandardScaler()
    X= scaler.fit_transform(X)
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    X = pd.DataFrame(X, columns=feat_col)
    return train_test_split(X, y, test_size = test_size, random_state = random_state)