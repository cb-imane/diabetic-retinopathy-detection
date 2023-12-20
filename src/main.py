import pickle
import uuid
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel #  for Data validation as class models
from typing import List

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


