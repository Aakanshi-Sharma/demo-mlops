import os
import numpy as np
import pandas as pd
import warnings
import sys  
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from get_data import read_config
import argparse
import json
import joblib
import mlflow
from urllib.parse import urlparse


def eval_metrics(y_true, y_pred):
    mse=mean_squared_error(y_true, y_pred)
    mae=mean_absolute_error(y_true, y_pred)
    r2=r2_score(y_true, y_pred)
    return mse, mae, r2

def train_and_eval(config_path):
    config=read_config(config_path)
    train_data_path=config["split_data"]["train_path"]
    test_data_path=config["split_data"]["test_path"]
    random_state=config["base"]["random_state"]
    target_col=config["base"]["target_col"]
    model_dir=config["model_dir"]
    alpha=config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio=config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    train=pd.read_csv(train_data_path, sep=',')
    test=pd.read_csv(test_data_path, sep=',')
    x_train=  train.drop(columns=[target_col])
    x_test= test.drop(columns=[target_col])
    y_train= train[target_col]
    y_test= test[target_col]

################ MLFLOW #############################

    mlflow_config=config["mlflow_config"]
    remote_server_uri=mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        model=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        rmse, mae, r2=eval_metrics(y_test, y_pred)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        mlflow.log_param("random_state",random_state)
        mlflow.log_metric("rmse" ,rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)

        tracking_url_type_store=urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store !="file":
            mlflow.sklearn.log_model(model, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")




if __name__ == '__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_args=arg.parse_args()
    train_and_eval(parsed_args.config)