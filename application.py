from flask import Flask, render_template, request
import pickle
from calculate import get_ratio

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():

    name1 = request.form['accname']
    name2 = request.form['dlname']
    ra = get_ratio(name1, name2)
    feat = [[ra]]
    result = model.predict(feat)
    if result == 1:
        result = "Yes"
    elif result == 0:
        result = "No"
    return render_template("predict.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

