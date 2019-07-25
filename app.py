from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from sklearn.externals import joblib
import requests
import json

app = Flask(__name__)

@app.route("/")
def index():
    response = make_response(render_template("index.html"))
    return response


@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        try:
            regressor = joblib.load("./linear_regression_model.pkl")
            data = dict(request.form.items())
            years_of_experience = float(data["YearsExperience"])
            prediction = regressor.predict([[years_of_experience]])
            response = make_response(render_template(
            "predicted.html",
            prediction = float(prediction)
            ))
        except ValueError as e:
            print(e)
            return jsonify("Please enter a number.")
        return response


if __name__ == '__main__':
    app.run(debug=True)