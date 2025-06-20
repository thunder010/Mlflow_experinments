# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import dagshub
dagshub.init(repo_owner='thunder010', repo_name='Mlflow_experinments', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)    # Set logging level to WARN
logger = logging.getLogger(__name__)      # Create a logger for the module 


def eval_metrics(actual, pred):            # Function to evaluate the model performance
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":                 # Main function to execute the script
    warnings.filterwarnings("ignore")        # Ignore warnings
    np.random.seed(40)                      # Set random seed for reproducibility so that results are consistent across runs

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")        # Read the CSV file from the URL using pandas
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e    # Log the exception if any error occurs
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)                     # Split the data into training and test sets using train_test_split from sklearn

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)        # Drop the "quality" column from the training set to get the features
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5     # Default value for alpha is 0.5 if not provided as a command line argument
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # Default value for l1_ratio is 0.5 if not provided as a command line argument

    with mlflow.start_run():                           # it starts a new MLflow run, which is a way to track experiments and log parameters, metrics, and models, this is how we can track our experiments
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)  # Create an instance of ElasticNet with the specified alpha and l1_ratio values
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)     

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)  # Evaluate the model performance using the eval_metrics function defined above

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))    # Print the model parameters
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        # ## For Remote server only(DAGShub)

        remote_server_uri="https://dagshub.com/thunder010/Mlflow_experinments.mlflow" # Replace with your DagsHub repository URI
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme   # Get the tracking URI scheme to determine the type of store being used, here store means the backend where the MLflow tracking server is running, it can be a file store, database, or remote server 

        # Model registry does not work with file store
        if tracking_url_type_store != "file":    # If the tracking URI is not a file store, we can register the model in the Model Registry
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)  # Log the model to MLflow, this will save the model in the MLflow tracking server, which can be used later for inference or deployment