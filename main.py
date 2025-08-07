from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os, sys
from sensor.logger import logging
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import load_object, read_yaml_file
from sensor.constants.training_pipeline import SAVED_MODEL_DIR
from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from sensor.constants.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Response models
class TrainingResponse(BaseModel):
    message: str
    status: str

class PredictionResponse(BaseModel):
    message: str
    status: str

app = FastAPI(
    title="Sensor Live API", 
    description="ML Pipeline API for Sensor Data",
    version="1.0.0"
)

# Updated CORS configuration for better Swagger UI compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", response_model=TrainingResponse, tags=["training"])
async def train():
    try:
        training_pipeline = TrainPipeline()
        
        if training_pipeline.is_pipeline_running:
            return TrainingResponse(
                message="Training pipeline is already running.",
                status="running"
            )
        
        training_pipeline.run_pipeline()
        return TrainingResponse(
            message="Training successfully completed!",
            status="completed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict():
    try:
        # get data and from the csv file 
        # convert it into dataframe 
        df = None

        Model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not Model_resolver.is_model_exists():
            raise HTTPException(status_code=404, detail="Model is not available")
        
        best_model_path = Model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping, inplace=True)

        # get the prediction output as you want 
        return PredictionResponse(
            message="Prediction completed",
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def main():
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ == "__main__":
    # For local development
    app_run(app, host=APP_HOST, port=APP_PORT)