import yaml
import pytest
import os
import json


@pytest.fixture
def config(config_path="params.yaml"):
    with open(config_path, 'r') as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

@pytest.fixture
def get_schema(schema_path="schema_min_max.json"):
    with open(schema_path) as json_path:
        schema=json.load(json_path)
    return schema
