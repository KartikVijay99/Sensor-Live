from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os, sys
from sensor.logger import logging
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import load_object, read_yaml_file
from sensor.constants.training_pipeline import SAVED_MODEL_DIR
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, BackgroundTasks
from sensor.constants.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json

# ==================== PYDANTIC MODELS ====================

class SensorDataInput(BaseModel):
    """Input model for single sensor data prediction"""
    aa_000: Optional[float] = Field(None, description="Sensor reading aa_000")
    ac_000: Optional[float] = Field(None, description="Sensor reading ac_000")
    ad_000: Optional[float] = Field(None, description="Sensor reading ad_000")
    ae_000: Optional[float] = Field(None, description="Sensor reading ae_000")
    af_000: Optional[float] = Field(None, description="Sensor reading af_000")
    ag_000: Optional[float] = Field(None, description="Sensor reading ag_000")
    ag_001: Optional[float] = Field(None, description="Sensor reading ag_001")
    ag_002: Optional[float] = Field(None, description="Sensor reading ag_002")
    ag_003: Optional[float] = Field(None, description="Sensor reading ag_003")
    ag_004: Optional[float] = Field(None, description="Sensor reading ag_004")
    ag_005: Optional[float] = Field(None, description="Sensor reading ag_005")
    ag_006: Optional[float] = Field(None, description="Sensor reading ag_006")
    ag_007: Optional[float] = Field(None, description="Sensor reading ag_007")
    ag_008: Optional[float] = Field(None, description="Sensor reading ag_008")
    ag_009: Optional[float] = Field(None, description="Sensor reading ag_009")
    ah_000: Optional[float] = Field(None, description="Sensor reading ah_000")
    ai_000: Optional[float] = Field(None, description="Sensor reading ai_000")
    aj_000: Optional[float] = Field(None, description="Sensor reading aj_000")
    ak_000: Optional[float] = Field(None, description="Sensor reading ak_000")
    al_000: Optional[float] = Field(None, description="Sensor reading al_000")
    am_0: Optional[float] = Field(None, description="Sensor reading am_0")
    an_000: Optional[float] = Field(None, description="Sensor reading an_000")
    ao_000: Optional[float] = Field(None, description="Sensor reading ao_000")
    ap_000: Optional[float] = Field(None, description="Sensor reading ap_000")
    aq_000: Optional[float] = Field(None, description="Sensor reading aq_000")
    ar_000: Optional[float] = Field(None, description="Sensor reading ar_000")
    as_000: Optional[float] = Field(None, description="Sensor reading as_000")
    at_000: Optional[float] = Field(None, description="Sensor reading at_000")
    au_000: Optional[float] = Field(None, description="Sensor reading au_000")
    av_000: Optional[float] = Field(None, description="Sensor reading av_000")
    ax_000: Optional[float] = Field(None, description="Sensor reading ax_000")
    ay_000: Optional[float] = Field(None, description="Sensor reading ay_000")
    ay_001: Optional[float] = Field(None, description="Sensor reading ay_001")
    ay_002: Optional[float] = Field(None, description="Sensor reading ay_002")
    ay_003: Optional[float] = Field(None, description="Sensor reading ay_003")
    ay_004: Optional[float] = Field(None, description="Sensor reading ay_004")
    ay_005: Optional[float] = Field(None, description="Sensor reading ay_005")
    ay_006: Optional[float] = Field(None, description="Sensor reading ay_006")
    ay_007: Optional[float] = Field(None, description="Sensor reading ay_007")
    ay_008: Optional[float] = Field(None, description="Sensor reading ay_008")
    ay_009: Optional[float] = Field(None, description="Sensor reading ay_009")
    az_000: Optional[float] = Field(None, description="Sensor reading az_000")
    az_001: Optional[float] = Field(None, description="Sensor reading az_001")
    az_002: Optional[float] = Field(None, description="Sensor reading az_002")
    az_003: Optional[float] = Field(None, description="Sensor reading az_003")
    az_004: Optional[float] = Field(None, description="Sensor reading az_004")
    az_005: Optional[float] = Field(None, description="Sensor reading az_005")
    az_006: Optional[float] = Field(None, description="Sensor reading az_006")
    az_007: Optional[float] = Field(None, description="Sensor reading az_007")
    az_008: Optional[float] = Field(None, description="Sensor reading az_008")
    az_009: Optional[float] = Field(None, description="Sensor reading az_009")
    ba_000: Optional[float] = Field(None, description="Sensor reading ba_000")
    ba_001: Optional[float] = Field(None, description="Sensor reading ba_001")
    ba_002: Optional[float] = Field(None, description="Sensor reading ba_002")
    ba_003: Optional[float] = Field(None, description="Sensor reading ba_003")
    ba_004: Optional[float] = Field(None, description="Sensor reading ba_004")
    ba_005: Optional[float] = Field(None, description="Sensor reading ba_005")
    ba_006: Optional[float] = Field(None, description="Sensor reading ba_006")
    ba_007: Optional[float] = Field(None, description="Sensor reading ba_007")
    ba_008: Optional[float] = Field(None, description="Sensor reading ba_008")
    ba_009: Optional[float] = Field(None, description="Sensor reading ba_009")
    bb_000: Optional[float] = Field(None, description="Sensor reading bb_000")
    bc_000: Optional[float] = Field(None, description="Sensor reading bc_000")
    bd_000: Optional[float] = Field(None, description="Sensor reading bd_000")
    be_000: Optional[float] = Field(None, description="Sensor reading be_000")
    bf_000: Optional[float] = Field(None, description="Sensor reading bf_000")
    bg_000: Optional[float] = Field(None, description="Sensor reading bg_000")
    bh_000: Optional[float] = Field(None, description="Sensor reading bh_000")
    bi_000: Optional[float] = Field(None, description="Sensor reading bi_000")
    bj_000: Optional[float] = Field(None, description="Sensor reading bj_000")
    bk_000: Optional[float] = Field(None, description="Sensor reading bk_000")
    bl_000: Optional[float] = Field(None, description="Sensor reading bl_000")
    bm_000: Optional[float] = Field(None, description="Sensor reading bm_000")
    bs_000: Optional[float] = Field(None, description="Sensor reading bs_000")
    bt_000: Optional[float] = Field(None, description="Sensor reading bt_000")
    bu_000: Optional[float] = Field(None, description="Sensor reading bu_000")
    bv_000: Optional[float] = Field(None, description="Sensor reading bv_000")
    bx_000: Optional[float] = Field(None, description="Sensor reading bx_000")
    by_000: Optional[float] = Field(None, description="Sensor reading by_000")
    bz_000: Optional[float] = Field(None, description="Sensor reading bz_000")
    ca_000: Optional[float] = Field(None, description="Sensor reading ca_000")
    cb_000: Optional[float] = Field(None, description="Sensor reading cb_000")
    cc_000: Optional[float] = Field(None, description="Sensor reading cc_000")
    cd_000: Optional[float] = Field(None, description="Sensor reading cd_000")
    ce_000: Optional[float] = Field(None, description="Sensor reading ce_000")
    cf_000: Optional[float] = Field(None, description="Sensor reading cf_000")
    cg_000: Optional[float] = Field(None, description="Sensor reading cg_000")
    ch_000: Optional[float] = Field(None, description="Sensor reading ch_000")
    ci_000: Optional[float] = Field(None, description="Sensor reading ci_000")
    cj_000: Optional[float] = Field(None, description="Sensor reading cj_000")
    ck_000: Optional[float] = Field(None, description="Sensor reading ck_000")
    cl_000: Optional[float] = Field(None, description="Sensor reading cl_000")
    cm_000: Optional[float] = Field(None, description="Sensor reading cm_000")
    cn_000: Optional[float] = Field(None, description="Sensor reading cn_000")
    cn_001: Optional[float] = Field(None, description="Sensor reading cn_001")
    cn_002: Optional[float] = Field(None, description="Sensor reading cn_002")
    cn_003: Optional[float] = Field(None, description="Sensor reading cn_003")
    cn_004: Optional[float] = Field(None, description="Sensor reading cn_004")
    cn_005: Optional[float] = Field(None, description="Sensor reading cn_005")
    cn_006: Optional[float] = Field(None, description="Sensor reading cn_006")
    cn_007: Optional[float] = Field(None, description="Sensor reading cn_007")
    cn_008: Optional[float] = Field(None, description="Sensor reading cn_008")
    cn_009: Optional[float] = Field(None, description="Sensor reading cn_009")
    co_000: Optional[float] = Field(None, description="Sensor reading co_000")
    cp_000: Optional[float] = Field(None, description="Sensor reading cp_000")
    cq_000: Optional[float] = Field(None, description="Sensor reading cq_000")
    cs_000: Optional[float] = Field(None, description="Sensor reading cs_000")
    cs_001: Optional[float] = Field(None, description="Sensor reading cs_001")
    cs_002: Optional[float] = Field(None, description="Sensor reading cs_002")
    cs_003: Optional[float] = Field(None, description="Sensor reading cs_003")
    cs_004: Optional[float] = Field(None, description="Sensor reading cs_004")
    cs_005: Optional[float] = Field(None, description="Sensor reading cs_005")
    cs_006: Optional[float] = Field(None, description="Sensor reading cs_006")
    cs_007: Optional[float] = Field(None, description="Sensor reading cs_007")
    cs_008: Optional[float] = Field(None, description="Sensor reading cs_008")
    cs_009: Optional[float] = Field(None, description="Sensor reading cs_009")
    ct_000: Optional[float] = Field(None, description="Sensor reading ct_000")
    cu_000: Optional[float] = Field(None, description="Sensor reading cu_000")
    cv_000: Optional[float] = Field(None, description="Sensor reading cv_000")
    cx_000: Optional[float] = Field(None, description="Sensor reading cx_000")
    cy_000: Optional[float] = Field(None, description="Sensor reading cy_000")
    cz_000: Optional[float] = Field(None, description="Sensor reading cz_000")
    da_000: Optional[float] = Field(None, description="Sensor reading da_000")
    db_000: Optional[float] = Field(None, description="Sensor reading db_000")
    dc_000: Optional[float] = Field(None, description="Sensor reading dc_000")
    dd_000: Optional[float] = Field(None, description="Sensor reading dd_000")
    de_000: Optional[float] = Field(None, description="Sensor reading de_000")
    df_000: Optional[float] = Field(None, description="Sensor reading df_000")
    dg_000: Optional[float] = Field(None, description="Sensor reading dg_000")
    dh_000: Optional[float] = Field(None, description="Sensor reading dh_000")
    di_000: Optional[float] = Field(None, description="Sensor reading di_000")
    dj_000: Optional[float] = Field(None, description="Sensor reading dj_000")
    dk_000: Optional[float] = Field(None, description="Sensor reading dk_000")
    dl_000: Optional[float] = Field(None, description="Sensor reading dl_000")
    dm_000: Optional[float] = Field(None, description="Sensor reading dm_000")
    dn_000: Optional[float] = Field(None, description="Sensor reading dn_000")
    do_000: Optional[float] = Field(None, description="Sensor reading do_000")
    dp_000: Optional[float] = Field(None, description="Sensor reading dp_000")
    dq_000: Optional[float] = Field(None, description="Sensor reading dq_000")
    dr_000: Optional[float] = Field(None, description="Sensor reading dr_000")
    ds_000: Optional[float] = Field(None, description="Sensor reading ds_000")
    dt_000: Optional[float] = Field(None, description="Sensor reading dt_000")
    du_000: Optional[float] = Field(None, description="Sensor reading du_000")
    dv_000: Optional[float] = Field(None, description="Sensor reading dv_000")
    dx_000: Optional[float] = Field(None, description="Sensor reading dx_000")
    dy_000: Optional[float] = Field(None, description="Sensor reading dy_000")
    dz_000: Optional[float] = Field(None, description="Sensor reading dz_000")
    ea_000: Optional[float] = Field(None, description="Sensor reading ea_000")
    eb_000: Optional[float] = Field(None, description="Sensor reading eb_000")
    ec_00: Optional[float] = Field(None, description="Sensor reading ec_00")
    ed_000: Optional[float] = Field(None, description="Sensor reading ed_000")
    ee_000: Optional[float] = Field(None, description="Sensor reading ee_000")
    ee_001: Optional[float] = Field(None, description="Sensor reading ee_001")
    ee_002: Optional[float] = Field(None, description="Sensor reading ee_002")
    ee_003: Optional[float] = Field(None, description="Sensor reading ee_003")
    ee_004: Optional[float] = Field(None, description="Sensor reading ee_004")
    ee_005: Optional[float] = Field(None, description="Sensor reading ee_005")
    ee_006: Optional[float] = Field(None, description="Sensor reading ee_006")
    ee_007: Optional[float] = Field(None, description="Sensor reading ee_007")
    ee_008: Optional[float] = Field(None, description="Sensor reading ee_008")
    ee_009: Optional[float] = Field(None, description="Sensor reading ee_009")
    ef_000: Optional[float] = Field(None, description="Sensor reading ef_000")
    eg_000: Optional[float] = Field(None, description="Sensor reading eg_000")

    class Config:
        schema_extra = {
            "example": {
                "aa_000": 0.0,
                "ac_000": 0.0,
                "ad_000": 0.0,
                "ae_000": 0.0,
                "af_000": 0.0,
                "ag_000": 0.0,
                "ag_001": 0.0,
                "ag_002": 0.0,
                "ag_003": 0.0,
                "ag_004": 0.0,
                "ag_005": 0.0,
                "ag_006": 0.0,
                "ag_007": 0.0,
                "ag_008": 0.0,
                "ag_009": 0.0,
                "ah_000": 0.0,
                "ai_000": 0.0,
                "aj_000": 0.0,
                "ak_000": 0.0,
                "al_000": 0.0,
                "am_0": 0.0,
                "an_000": 0.0,
                "ao_000": 0.0,
                "ap_000": 0.0,
                "aq_000": 0.0,
                "ar_000": 0.0,
                "as_000": 0.0,
                "at_000": 0.0,
                "au_000": 0.0,
                "av_000": 0.0,
                "ax_000": 0.0,
                "ay_000": 0.0,
                "ay_001": 0.0,
                "ay_002": 0.0,
                "ay_003": 0.0,
                "ay_004": 0.0,
                "ay_005": 0.0,
                "ay_006": 0.0,
                "ay_007": 0.0,
                "ay_008": 0.0,
                "ay_009": 0.0,
                "az_000": 0.0,
                "az_001": 0.0,
                "az_002": 0.0,
                "az_003": 0.0,
                "az_004": 0.0,
                "az_005": 0.0,
                "az_006": 0.0,
                "az_007": 0.0,
                "az_008": 0.0,
                "az_009": 0.0,
                "ba_000": 0.0,
                "ba_001": 0.0,
                "ba_002": 0.0,
                "ba_003": 0.0,
                "ba_004": 0.0,
                "ba_005": 0.0,
                "ba_006": 0.0,
                "ba_007": 0.0,
                "ba_008": 0.0,
                "ba_009": 0.0,
                "bb_000": 0.0,
                "bc_000": 0.0,
                "bd_000": 0.0,
                "be_000": 0.0,
                "bf_000": 0.0,
                "bg_000": 0.0,
                "bh_000": 0.0,
                "bi_000": 0.0,
                "bj_000": 0.0,
                "bk_000": 0.0,
                "bl_000": 0.0,
                "bm_000": 0.0,
                "bs_000": 0.0,
                "bt_000": 0.0,
                "bu_000": 0.0,
                "bv_000": 0.0,
                "bx_000": 0.0,
                "by_000": 0.0,
                "bz_000": 0.0,
                "ca_000": 0.0,
                "cb_000": 0.0,
                "cc_000": 0.0,
                "cd_000": 0.0,
                "ce_000": 0.0,
                "cf_000": 0.0,
                "cg_000": 0.0,
                "ch_000": 0.0,
                "ci_000": 0.0,
                "cj_000": 0.0,
                "ck_000": 0.0,
                "cl_000": 0.0,
                "cm_000": 0.0,
                "cn_000": 0.0,
                "cn_001": 0.0,
                "cn_002": 0.0,
                "cn_003": 0.0,
                "cn_004": 0.0,
                "cn_005": 0.0,
                "cn_006": 0.0,
                "cn_007": 0.0,
                "cn_008": 0.0,
                "cn_009": 0.0,
                "co_000": 0.0,
                "cp_000": 0.0,
                "cq_000": 0.0,
                "cs_000": 0.0,
                "cs_001": 0.0,
                "cs_002": 0.0,
                "cs_003": 0.0,
                "cs_004": 0.0,
                "cs_005": 0.0,
                "cs_006": 0.0,
                "cs_007": 0.0,
                "cs_008": 0.0,
                "cs_009": 0.0,
                "ct_000": 0.0,
                "cu_000": 0.0,
                "cv_000": 0.0,
                "cx_000": 0.0,
                "cy_000": 0.0,
                "cz_000": 0.0,
                "da_000": 0.0,
                "db_000": 0.0,
                "dc_000": 0.0,
                "dd_000": 0.0,
                "de_000": 0.0,
                "df_000": 0.0,
                "dg_000": 0.0,
                "dh_000": 0.0,
                "di_000": 0.0,
                "dj_000": 0.0,
                "dk_000": 0.0,
                "dl_000": 0.0,
                "dm_000": 0.0,
                "dn_000": 0.0,
                "do_000": 0.0,
                "dp_000": 0.0,
                "dq_000": 0.0,
                "dr_000": 0.0,
                "ds_000": 0.0,
                "dt_000": 0.0,
                "du_000": 0.0,
                "dv_000": 0.0,
                "dx_000": 0.0,
                "dy_000": 0.0,
                "dz_000": 0.0,
                "ea_000": 0.0,
                "eb_000": 0.0,
                "ec_00": 0.0,
                "ed_000": 0.0,
                "ee_000": 0.0,
                "ee_001": 0.0,
                "ee_002": 0.0,
                "ee_003": 0.0,
                "ee_004": 0.0,
                "ee_005": 0.0,
                "ee_006": 0.0,
                "ee_007": 0.0,
                "ee_008": 0.0,
                "ee_009": 0.0,
                "ef_000": 0.0,
                "eg_000": 0.0
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    prediction: str = Field(..., description="Predicted class (pos/neg)")
    confidence: float = Field(..., description="Prediction confidence score")
    status: str = Field(..., description="Prediction status")
    message: str = Field(..., description="Additional information")

class TrainingResponse(BaseModel):
    """Response model for training operations"""
    message: str = Field(..., description="Training status message")
    status: str = Field(..., description="Training status")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional training details")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Health check message")
    model_available: bool = Field(..., description="Whether model is available")

class DataUploadResponse(BaseModel):
    """Response model for data upload"""
    message: str = Field(..., description="Upload status message")
    records_processed: int = Field(..., description="Number of records processed")
    status: str = Field(..., description="Upload status")

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Sensor Failure Detection API",
    description="ML API for predicting sensor failures using 170+ sensor features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== UTILITY FUNCTIONS ====================

def get_model():
    """Get the trained model"""
    try:
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return None
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def prepare_input_data(sensor_data: SensorDataInput) -> pd.DataFrame:
    """Convert sensor data input to DataFrame"""
    try:
        # Convert to dict and handle None values
        data_dict = sensor_data.dict()
        # Replace None with 0 for missing values
        for key, value in data_dict.items():
            if value is None:
                data_dict[key] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([data_dict])
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preparing input data: {str(e)}")

# ==================== API ENDPOINTS ====================

@app.get("/", tags=["authentication"])
async def index():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    model = get_model()
    return HealthResponse(
        status="healthy",
        message="Sensor Failure Detection API is running",
        model_available=model is not None
    )

@app.get("/test", tags=["testing"])
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "API is working!", "status": "success"}

@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_sensor_failure(sensor_data: SensorDataInput):
    """
    Predict sensor failure based on sensor readings
    
    This endpoint accepts sensor data and returns a prediction of whether
    the sensor system will fail (pos) or continue operating normally (neg).
    """
    try:
        # Get the trained model
        model = get_model()
        if model is None:
            raise HTTPException(status_code=404, detail="Model not available. Please train the model first.")
        
        # Prepare input data
        input_df = prepare_input_data(sensor_data)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Convert prediction to human-readable format
        target_mapping = TargetValueMapping()
        reverse_mapping = target_mapping.reverse_mapping()
        
        predicted_class = reverse_mapping.get(prediction[0], "unknown")
        
        # For now, we'll use a simple confidence score
        # In a real application, you might want to use model.predict_proba()
        confidence = 0.85  # Placeholder confidence score
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            status="success",
            message=f"Prediction completed successfully. Predicted class: {predicted_class}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=List[PredictionResponse], tags=["prediction"])
async def predict_batch_sensor_failures(sensor_data_list: List[SensorDataInput]):
    """
    Predict sensor failures for multiple sensor readings
    
    This endpoint accepts a list of sensor data and returns predictions
    for each sensor reading.
    """
    try:
        # Get the trained model
        model = get_model()
        if model is None:
            raise HTTPException(status_code=404, detail="Model not available. Please train the model first.")
        
        predictions = []
        
        for sensor_data in sensor_data_list:
            try:
                # Prepare input data
                input_df = prepare_input_data(sensor_data)
                
                # Make prediction
                prediction = model.predict(input_df)
                
                # Convert prediction to human-readable format
                target_mapping = TargetValueMapping()
                reverse_mapping = target_mapping.reverse_mapping()
                
                predicted_class = reverse_mapping.get(prediction[0], "unknown")
                
                predictions.append(PredictionResponse(
                    prediction=predicted_class,
                    confidence=0.85,  # Placeholder confidence score
                    status="success",
                    message=f"Prediction completed successfully. Predicted class: {predicted_class}"
                ))
                
            except Exception as e:
                predictions.append(PredictionResponse(
                    prediction="error",
                    confidence=0.0,
                    status="error",
                    message=f"Error processing prediction: {str(e)}"
                ))
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/upload-data", response_model=DataUploadResponse, tags=["data"])
async def upload_sensor_data(file: UploadFile = File(...)):
    """
    Upload sensor data CSV file to MongoDB
    
    This endpoint allows you to upload a CSV file containing sensor data
    which will be stored in MongoDB for training.
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Read CSV file
        df = pd.read_csv(file.file)
        
        # Save to MongoDB
        from sensor.data_access.sensor_data import SensorData
        sensor_data = SensorData()
        
        # Save to temporary file first
        temp_file = "temp_upload.csv"
        df.to_csv(temp_file, index=False)
        
        # Upload to MongoDB
        records_count = sensor_data.save_csv_file(
            file_path=temp_file,
            collection_name="Data"
        )
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return DataUploadResponse(
            message=f"Successfully uploaded {records_count} records to MongoDB",
            records_processed=records_count,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data upload failed: {str(e)}")

@app.post("/train", response_model=TrainingResponse, tags=["training"])
async def train_model(background_tasks: BackgroundTasks):
    """
    Train the sensor failure detection model
    
    This endpoint initiates the complete training pipeline including:
    - Data ingestion from MongoDB
    - Data validation
    - Data transformation
    - Model training
    - Model evaluation
    - Model deployment
    """
    try:
        # Check if pipeline is already running
        if TrainPipeline.is_pipeline_running:
            return TrainingResponse(
                message="Training pipeline is already running. Please wait for completion.",
                status="running"
            )
        
        # Start training in background
        def run_training():
            try:
                training_pipeline = TrainPipeline()
                training_pipeline.run_pipeline()
            except Exception as e:
                logging.error(f"Training failed: {e}")
        
        background_tasks.add_task(run_training)
        
        return TrainingResponse(
            message="Training pipeline started successfully. Check logs for progress.",
            status="started",
            details={
                "pipeline_steps": [
                    "Data Ingestion",
                    "Data Validation", 
                    "Data Transformation",
                    "Model Training",
                    "Model Evaluation",
                    "Model Deployment"
                ]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model-status", tags=["model"])
async def get_model_status():
    """
    Get the status of the trained model
    
    Returns information about the current model including:
    - Model availability
    - Model path
    - Training timestamp
    """
    try:
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        
        if not model_resolver.is_model_exists():
            return {
                "model_available": False,
                "message": "No trained model found",
                "model_path": None,
                "timestamp": None
            }
        
        best_model_path = model_resolver.get_best_model_path()
        timestamp = os.path.basename(os.path.dirname(best_model_path))
        
        return {
            "model_available": True,
            "message": "Model is available for predictions",
            "model_path": best_model_path,
            "timestamp": timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")

def main():
    """Main function for running the training pipeline directly"""
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ == "__main__":
    # For local development
    app_run(app, host=APP_HOST, port=APP_PORT)