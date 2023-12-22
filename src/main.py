import pickle
import uuid
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel #  for Data validation as class models
from typing import List
import uvicorn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score



app = FastAPI()


MODELS = {
    "lr": {
        "model": LogisticRegression,
        "name": "Logistic Regression",
        "api_model_code": "lr",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    },
    "dt": {
        "model": DecisionTreeClassifier,
        "name": "Decision Tree",
        "api_model_code": "dt",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"
    },
    "knn": {
        "model": KNeighborsClassifier,
        "name": "K Nearest Neighbors",
        "api_model_code": "knn",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"

    },
    "sgd": {
       "model": SGDClassifier,
       "name": "K Nearest Neighbors",
       "api_model_code": "STOCHASTIC GRADIENT DESCENT",
       "documentation": "https://scikit-learn.org/stable/modules/sgd.html"
    }
}



class ModelInfo(BaseModel):
    name: str
    api_model_code: str
    documentation: str = None

class ModelTrainIn(BaseModel):
    api_model_code: str
    trained_model_name: str

class ModelTrainOut(BaseModel):
    train_id: str = None
    api_model_code: str = None
    trained_model_name: str = None
    recall: float = None  
    f1: float = None  
        

@app.get("/model/info/", response_model=List[ModelInfo])
async def info_models():
    result = []
    for model_key in MODELS:
        model_info = MODELS[model_key]
        result.append(model_info)
    return result


@app.post("/model/train/", response_model=ModelTrainOut) # here we're defining the data model for our response
async def train_model(model_train: ModelTrainIn): # this defines the data model for our request
    # we transform our data model object into a dictionary
    model_train_dict = model_train.dict() 
    
    # we initialize our ML model
    model = MODELS[model_train_dict['api_model_code']]['model']() # we initialize our ML model
    
    # load and split data
    df = pd.read_pickle("../data/data_retino_preprocessed.pkl")
    print(df.columns)

    X = df.drop('Class',axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    
    # fit model and get a prediction and a score
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    
    # we define a unique ID
    unique_id = uuid.uuid4().hex 
    
    # saving the model locally 
    filename = f"trained_models/{unique_id}.sav"
    pickle.dump(model, open(filename, 'wb')) 
    model_train_dict.update({"train_id": unique_id, "recall": recall, "f1": f1})
    
    # return our response that has the same data structure as ModelTrainOut
    return model_train_dict 


class PredictionParams(BaseModel):
    train_id: str = "e2a94cc00a284a2a865804926769e21f.pkl"
    ma1:int
    exudate1:float
    exudate2:float
    exudate3:float
    exudate31:float
    exudate5: float
    macula_opticdisc_distance:float
    opticdisc_diameter:float
    am_fm_classification:int
    #model_name: str = 'lr_model.pkl'
                    

class Predictionresult(BaseModel):
    train_id: str
    prediction:int


DIABETIC_RETINOPATHY = {
    0: 'negative dr',
    1: 'positive dr'
    }

def load_data(model_name):
    with open(model_name,'rb') as f:
        model = pickle.load(f)
    return model    

@app.post("/model/predict/", response_model=Predictionresult)
async def predict_dr(query_data: PredictionParams):
    query_data_dict = query_data.dict()
    model_id = query_data_dict['train_id']
    # loading the saved ML Model
    model = pickle.load(open(f"trained_models/{model_id}.pkl", 'rb'))
	
    ma1 = query_data.ma1
    exudate1 = query_data.exudate1
    exudate2 = query_data.exudate2
    exudate3 = query_data.exudate3
    exudate31 = query_data.exudate1
    exudate5 = query_data.exudate5
    macula_opticdisc_distance = query_data.macula_opticdisc_distance
    opticdisc_diameter = query_data.opticdisc_diameter
    am_fm_classification = query_data.am_fm_classification

    pred = model.predict([[
        ma1,
        exudate1,
        exudate2,
        exudate3,
        exudate31,
        exudate5,
        macula_opticdisc_distance,
        opticdisc_diameter,
        am_fm_classification

    ]])

    return {
        "prediction": pred,
        "train_id": model_id
    }


if __name__ == "__main__":
    
    debug=True
    uvicorn.run(app, host="localhost", port=8000)
