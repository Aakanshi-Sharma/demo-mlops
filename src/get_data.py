import os
import yaml
import pandas as pd
import argparse

def get_data(config_path):
    config=read_config(config_path)
    data_path=config["data_source"]["s3_source"]
    df=pd.read_csv(data_path, encoding="utf-8")
    return df

def read_config(config_path):
    with open(config_path, 'r') as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args=args.parse_args()
    data=get_data(parsed_args.config)
