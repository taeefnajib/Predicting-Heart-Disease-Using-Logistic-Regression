# importing Flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

# instantiating the app
app = Flask(__name__, template_folder='templates')

# loading the model and scaler from the model.pkl file
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# this is the function that the app will run to load GUI
@app.route('/')
def home():
    return render_template('index.html')

# this is the function that the app will run to predict
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input = [float(x) for x in request.form.values()]
    
    features = [np.array(input)]
    
    scaled_features = scaler.transform(features)
    
    prediction = model.predict(scaled_features)
    
    if round(prediction[0])==0:
        output = "Zero or minimum risk of heart disease!"
        color="green"
    else:
        output = "High risk of heart disease!"
        color="red"
        
    
    return render_template('index.html', prediction_text=output, color=color)


if __name__=="__main__":
	app.run(debug=True, port=8000) 