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
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		prediction = np.array([[nbclf.predict(vect)]])
    if prediction == "spam":
        return render_template('index.html', prediction_text = "Your SMS is Spam!")
    else:
        return render_template('index.html', prediction_text= "Your SMS is Ham!")
      
if __name__ == "__main__":
    app.run(debug=True)