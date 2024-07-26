import os 
import argparse
import yaml
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from get_data import read_config

def split_data(config_path):
    config=read_config(config_path)
    train_path=config["split_data"]["train_path"]
    test_path=config["split_data"]["test_path"]
    split_ratio=config["split_data"]["test_size"]
    raw_data_path=config["load_data"]["raw_dataset_csv"]
    random_state=config["base"]["random_state"]
    df=pd.read_csv(raw_data_path)
    train, test=train_test_split(df, test_size=split_ratio, random_state=random_state)
    return


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=argparse.parse_args()
