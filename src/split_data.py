import os 
import argparse
import yaml
import pandas as pd
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
    train.to_csv(train_path, index=False, sep=",", encoding="utf8")
    test.to_csv(test_path, index=False, sep=",", encoding="utf8")
    return


if __name__ == '__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_args=arg.parse_args()
    split_data(parsed_args.config)
