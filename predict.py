from __future__ import print_function
import joblib
import pandas as pd
import os
import sys
from transform_data import combine_load_weather_df

MODEL_FILE = os.getenv("MODEL_PATH", "regr.model")

model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

def load_model():
    if not os.path.exists(MODEL_FILE):
        print("Unable to find the model file regr.model", file=sys.stderr)
        return None
    return joblib.load(MODEL_FILE)

def predict(row):
    model = get_model()
    if not model:
        return 'error-no-model'

    dataset = [row]
    result = model.predict(dataset)
    print("prediction: {}".format(result))
    return result



if __name__ == '__main__':
    row = [
     5031.266666666667, 2017.0, 0.0, -0.6387435376801648, -0.8415697739684432,
     -1.2442150032543748, -1.2772769186799113, -1.0023194992188573, -0.8203540979800136,
    -1.3047092259299256, -1.1211052512542388, -0.8859669409647838, -0.8082014419285851,
    -0.2915183619055513, 0.27341614249180335, -1.1669882182493352, -1.2592356722123388,
    -0.7891909064581724, -0.5638315723420105, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0]
    
    print(predict(row))