import os
import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

path_dir = os.path.dirname(__file__)
path = os.path.join(path_dir, 'cars_dill_pipe.pkl')
with open(path, 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int

class Prediction(BaseModel):
    id: int
    pred: str
    price: int

@app.route('/status')
def status():
    return {'status': 'Active'}


@app.route('/version')
def version():
    return model['metadata']

@app.route('/predict', methods = ['POST'])
def predict():
    form = request.get_json()
    df = pd.DataFrame.from_dict([form])
    pipe = model['pipe']
    y = pipe.predict(df)
    r = {}
    r["id"] = int(df["id"][0])
    r["price"] = int(df["price"][0])
    r["pred"] = y[0]
    return r

app.run(host='0.0.0.0', port=8000)
