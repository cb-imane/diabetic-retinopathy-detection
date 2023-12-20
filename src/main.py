from fastapi import FastAPI
from pydantic import BaseModel #  for Data validation as class models
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


app = FastAPI()

@app.get("/")
async def say():
    return {'message': 'hello world'}

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
        

@app.get("/model/info/", response_model=List[ModelInfo])
async def info_models():
    result = []
    for model_key in MODELS:
        model_info = MODELS[model_key]
        result.append(model_info)
    return result
