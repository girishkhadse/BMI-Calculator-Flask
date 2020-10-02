from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('classifier.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['Height']
    data2 = request.form['Weight']
    arr = np.array([[data1, data2]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__ == "__main__":
    app.run(debug=True)