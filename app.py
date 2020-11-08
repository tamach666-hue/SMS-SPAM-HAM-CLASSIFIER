import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
transformer = pickle.load(open("tfr.pkl", "rb"))
#pred = model.predict(vec)
#print('Your SMS Type is:', type(pred), pred)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    SMS = float(request.form['smsInput'])
    vec = transformer.transform(SMS)
    prediction = model.predict(vec)
    if prediction == 'spam':
        return render_template('index.html', prediction_text = "Your SMS is Spam!")
    else:
        return render_template('index.html', prediction_text= "Your SMS is Ham!")
      
if __name__ == "__main__":
    app.run(debug=True)