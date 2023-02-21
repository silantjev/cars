FROM python:3.10

WORKDIR /code

#COPY ./requirements.txt ./requirements.txt
#RUN pip install -r requirements.txt

RUN pip install --upgrade pip

#RUN pip install dill pandas scikit-learn pydantic fastapi uvicorn 
COPY ./requirements_local_api.txt ./
RUN pip install -r requirements_local_api.txt

COPY . ./

EXPOSE 8000

CMD uvicorn local_api:app --host 0.0.0.0 --port 8000
