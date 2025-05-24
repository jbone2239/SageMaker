import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
from io import StringIO

# load model function for SageMaker
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

# prediction function for SageMaker
def predict_fn(input_data, model):
    return model.predict(input_data)

# input function to process incoming CSV request
def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        # Explicitly define the feature column names (used during training)
        columns = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'wine_type'
        ]
        return pd.read_csv(StringIO(request_body), names=columns)
    else:
        raise ValueError("This model only supports CSV input")

# output function to return CSV-formatted prediction
def output_fn(prediction, response_content_type):
    if response_content_type == 'text/csv':
        return ','.join(str(x) for x in prediction)
    else:
        raise ValueError("This model only supports CSV output")

# main training logic
def main():
    # load training data
    df = pd.read_csv("/opt/ml/input/data/training/winequality_combined.csv")
    X = df.drop(columns=["quality"])
    y = df["quality"]

    # train model
    model = LinearRegression()
    model.fit(X, y)

    # save model to the expected location
    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    main()
