# setup a fastapi server that will use the model.joblib file to make predictions
from fastapi import FastAPI
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI()

MODEL_PATH = 'data/models/model.joblib'
model = joblib.load(MODEL_PATH)

class PredictionRequest(BaseModel):
    features: Dict[str, float]

@app.post('/predict')
def predict(request: PredictionRequest):
    # sklearn models trained with pandas expect a DataFrame so columns match names
    df = pd.DataFrame([request.features])
    df.drop(columns=['ID'], inplace=True)
    prediction = model.predict(df)
    # Ensure JSON serializable response
    return {"prediction": prediction.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
