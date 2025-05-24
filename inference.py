import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from wine_lr_model import model_fn, predict_fn, input_fn, output_fn

app = Flask(__name__)
model = model_fn(".")

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

@app.route('/invocations', methods=['POST'])
def invocations():
    data = request.data.decode('utf-8')
    input_data = input_fn(data, 'text/csv')
    prediction = predict_fn(input_data, model)
    result = output_fn(prediction, 'text/csv')
    return result
