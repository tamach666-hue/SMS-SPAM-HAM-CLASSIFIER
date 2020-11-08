from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
filename = 'sms_model.pkl'
nbclf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		sms_prediction = nbclf.predict(vect)
		return render_template('result.html', prediction=sms_prediction)
      
if __name__ == "__main__":
    app.run(debug=True)