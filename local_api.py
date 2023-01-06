import os
import dill
import pandas as pd
import json

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
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

@app.get('/status')
def status():
	return 'Active'


@app.get('/version')
def version():
	return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
	df = pd.DataFrame.from_dict([form.dict()])
	pipe = model['pipe']
	y = pipe.predict(df)
	r = {}
	r["id"] = df["id"][0]
	r["price"] = df["price"][0]
	r["pred"] = y[0]
	return r

