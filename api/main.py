from fastapi import FastAPI
import pickle
from pydantic import BaseModel, Field
import sys
import os
sys.path.append(os.path.abspath("../src"))
import functions as fnc
import pandas as pd

#define what we are expecting to be sent to our api
class Item(BaseModel):
    Pclass: int = Field()
    Name: str = Field()
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str
        
    class Config:
        schema_extra = {"example": {"Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171", "Fare": 7.25, "Cabin": "C34", "Embarked": "S"}}

class ResponseItem(BaseModel):
    probability_of_survival: float
        
with open('model/xgb_clf.pickle', 'rb') as f:
    model = pickle.load(f)


def score_model(data: Item):
    data_df = pd.DataFrame(data, index=[0])
    result = model.predict_proba(data_df)[:,1]
    return result
    
    
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "welcome to the api. navigate to /docs to view the api documentation"}

@app.post("/score")
async def score_model(data: Item):
    data_df = pd.DataFrame(dict(data), index=[0])
    result = model.predict_proba(data_df)[:,1]
    print('THIS IS THE RESULT: {}'.format(result[0]))
    returner = {'probability_of_survival': float(result[0])}
    return returner
