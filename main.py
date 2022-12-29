from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load

app = FastAPI()
@app.on_event("startup")
def load_model():
    app.model = load("model/final_model.joblib")

class Input(BaseModel):
    sepal_length: float = Field(gt=0, description="The sepal length in cm, must be greater than zero")
    sepal_width: float = Field(gt=0, description="The sepal width in cm, must be greater than zero")
    petal_length: float = Field(gt=0, description="The petal length in cm, must be greater than zero")
    petal_width: float = Field(gt=0, description="The petal width in cm, must be greater than zero")


class Output(BaseModel):
    label: float
    probs: dict


@app.post("/predict", response_model=Output)
def model_predict(input: Input): #or async def?
    prediction = app.model.predict([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])[0]
    probs = dict(enumerate(app.model.predict_proba([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])[0]))

    return Output(label=prediction, probs=probs)