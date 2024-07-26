import os
from get_data import get_data, read_config
import argparse


def load_and_save(config_path):
    config=read_config(config_path)
    df=get_data(config_path)
    new_cols=[col.replace(" ","_") for col in df.columns]
    raw_data_path=config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, header=new_cols, index=False, sep=",")
    

if __name__ == "__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_args=arg.parse_args()
    load_and_save(parsed_args.config)