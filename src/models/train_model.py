# Importing all dependencies 
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


# Saving the model
def save_model(model):
    return pickle.dump(model, open('models/model.pkl', 'wb'))

# Fitting model
def fit_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression()
    model = model.fit(X_train, y_train)
    save_model(model=model)
    return model