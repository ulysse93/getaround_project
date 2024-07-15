import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import pandas as pd
import joblib
#import xgboost
import xgboost

description = """
Get Around API helps you to determine the optimal rental price of a car. 
The goal of this API is to provide predictions based on an XGBoost model. 

To get the prediction, you have to specify the features of the car you want to rent.

The features to enter to get a prediction are:
- 'model_key': the brand of the car
- 'mileage': the mileage in kilometers
- 'engine_power': the engine power
- 'fuel': the type of fuel
- 'paint_color': the color of the car
- 'car_type': the type of car
- 'private_parking_available': The availability of a private parking for the car 
- 'has_gps': If the car is equipped with a GPS
- 'has_air_conditioning': Does the car have air conditioning
- 'automatic_car': If the car is an automatic car
- 'has_getaround_connect': If the car is equipped with Getaround connect
- 'has_speed_regulator': If the car is equipped with a speed regulator
- 'winter_tires': If the car is equipped with winter tires

API Endpoints:

## Preview
* '/preview': visualize a few rows of your dataset
## Predictions 
* '/predict': give you a rental price proposition for the given features 
"""

tags_metadata = [
    {
        "name": "Preview",
        "description": "Endpoints that quickly explore dataset"
    },
    {
        "name": "Predictions",
        "description": "Endpoints that use our Machine Learning model to suggest car pricing"
    },
    {
        "name": "Default",
        "description": "Default endpoint"
    }
]

app = FastAPI(
    title="GetAround API",
    description=description,
    version="1.0",
    contact={
        "name": "Seddik AMROUN",
    },
    openapi_tags=tags_metadata
)

class PredictionFeatures(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

    model_config = ConfigDict(protected_namespaces=())

@app.get("/", tags=["Default"])
async def read_root():
    return {"message": "Welcome to the Get Around API. Use /docs to see the API documentation."}

@app.post("/predict", tags=["Predictions"])
async def predict(new_line: PredictionFeatures):
    """
    Price prediction for given car features. Endpoint will return a dictionary like this:
    '''
    {'prediction': prediction_value}
    '''
    You need to give this endpoint all columns values as dictionary.
    """
    new_line = dict(new_line)
    pred_features = pd.DataFrame([new_line])

    # Load model & predict
    loaded_model = xgboost.XGBRegressor()
    loaded_model.load_model('model_Getaround.json')
    loaded_preprocessor = joblib.load('preprocessor.joblib')
    df = loaded_preprocessor.transform(pred_features)
    prediction = loaded_model.predict(df)
    
    response = {"prediction": prediction.tolist()[0]}
    return response

@app.get("/preview", tags=["Preview"])
async def preview(rows: int):
    """ Give a preview of the dataset: Input the number of rows"""
    data = pd.read_csv('get_around_pricing_project.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)
    preview = data.head(rows)
    return preview.to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)