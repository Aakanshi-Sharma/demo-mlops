import yaml 
import os
import json
import joblib
import numpy as np

params_path="params.yaml"
schema_path=os.path.join("prediction_service", "schema_in.json")


class NotInRange(Exception):
    def __init__(self, message="Not in range"):
        self.message = message
        super().__init__(self.message)


class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message=message
        super().__init__(self.message)


def read_config(config_path):
    with open(config_path, 'r') as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config


def predict(data):
    config=read_config(params_path)
    model_dir=config["webapp_model_dir"]
    model=joblib.load(model_dir)
    prediction=model.predict(data).tolist()[0]

    try:
        if 3<=prediction<=8:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"

def get_schema(schema_path=schema_path):
    with open(schema_path) as json_path:
        schema=json.load(json_path)
    return schema


def validate_input(dict_request):
    def _validate_cols(col):
        schema=get_schema()
        actual_cols=schema.keys()
        if col not in actual_cols:
            raise NotInCols
    
    def _validate_values(col, val):
        schema=get_schema()
        min_max_dict=schema[col]
        if float(val) >min_max_dict["max"] or float(val)< min_max_dict["min"]: 
            raise NotInRange

        


    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    return True

def form_response(dict_request):
    pass

def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data=np.array([list(dict_request.values())])
            response=predict(data)
            response={"response": response}
            return response
        
    except Exception as e:
        response={"Excepted_range":get_schema(), "response":str(e)}
        return response