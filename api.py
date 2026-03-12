# import required libraries
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# create Flask application
app = Flask(__name__)

# load trained ML model
model = pickle.load(open("model.pkl", "rb"))

# create API home route (for testing)
@app.route("/")
def home():
    return "Customer Churn Prediction API is running"


# create prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    # get JSON data from request
    data = request.get_json()

    # convert JSON to dataframe
    input_data = pd.DataFrame([data])

    # make prediction using ML model
    prediction = model.predict(input_data)

    # convert prediction to readable result
    result = int(prediction[0])

    # return result as JSON
    return jsonify({
        "prediction": result
    })


# run Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)