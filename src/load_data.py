import os
from get_data import get_data, read_config
import argparse


def load_and_save(config_path):
    config=read_config(config_path)
    return