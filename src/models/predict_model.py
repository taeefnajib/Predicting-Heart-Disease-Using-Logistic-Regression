# Importing all dependencies
import pickle
from sklearn.metrics import accuracy_score

def predict_model(X_test, y_test):
    model = pickle.load(open('models/model.pkl', 'rb'))
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    return acc