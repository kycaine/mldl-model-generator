from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
import pandas as pd

from pipeline.file_loader import load_file
from pipeline.schema_detector import detect_schema
from pipeline.data_cleaning import clean_data
from pipeline.feature_engineering import engineer_features
from pipeline.feature_selector import select_features
from pipeline.model_selector import select_model
from pipeline.trainer import train_model
from pipeline.evaluator import evaluate_model
from pipeline.exporter import export_model

router = APIRouter()
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

class CleanRequest(BaseModel):
    file_id: str
    
class FeatureEngineRequest(BaseModel):
    file_id: str
    target_column: str
    feature_columns: list[str]

class TrainRequest(BaseModel):
    file_id: str
    model_type: str # classification or regression
    model_name: str # e.g., Logistic Regression, Random Forest
    target_column: str
    feature_columns: list[str]

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".xls", ".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"message": "File uploaded successfully", "file_id": file.filename}

@router.get("/dataset-columns")
async def get_dataset_columns(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        df = load_file(file_path)
        schema_info = detect_schema(df)
        return schema_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean-data")
async def trigger_clean_data(request: CleanRequest):
    file_path = os.path.join(UPLOAD_DIR, request.file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        df = load_file(file_path)
        cleaned_df = clean_data(df)
        
        # Save cleaned file to processed dir
        processed_path = os.path.join(PROCESSED_DIR, f"cleaned_{request.file_id}")
        cleaned_df.to_csv(processed_path, index=False)
        
        return {"message": "Data cleaned successfully", "processed_file_id": f"cleaned_{request.file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature-engineering")
async def trigger_feature_engineering(request: FeatureEngineRequest):
    # Depending on flow, we should probably read the CLEANED file
    processed_path = os.path.join(PROCESSED_DIR, f"cleaned_{request.file_id}")
    file_path = processed_path if os.path.exists(processed_path) else os.path.join(UPLOAD_DIR, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        df = load_file(file_path)
        
        # Select subset if needed, or transform everything
        # Actually usually you split first, but here we'll transform the whole dataset based on prompt
        transformed_df, preprocessors = engineer_features(df, request.file_id)
        
        # Save transformed file
        final_path = os.path.join(PROCESSED_DIR, f"engineered_{request.file_id}")
        transformed_df.to_csv(final_path, index=False)
        
        # In a real app we'd save the preprocessors too so the trainer endpoint can access them
        import joblib
        joblib.dump(preprocessors, os.path.join(PROCESSED_DIR, f"prep_{request.file_id}.joblib"))
        
        return {"message": "Feature engineering completed", "engineered_file_id": f"engineered_{request.file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def trigger_train_model(request: TrainRequest):
    engineered_path = os.path.join(PROCESSED_DIR, f"engineered_{request.file_id}")
    if not os.path.exists(engineered_path):
        print(f"Error: Engineered file not found at {engineered_path}")
        raise HTTPException(status_code=404, detail="Engineered file not found. Please run feature engineering first.")
        
    try:
        print(f"Loading data for training: {engineered_path}")
        df = load_file(engineered_path)
        
        print(f"Selecting features: target={request.target_column}, features={request.feature_columns}")
        X, y = select_features(df, request.target_column, request.feature_columns)
        
        print(f"Selecting model: {request.model_type} - {request.model_name}")
        model = select_model(request.model_type, request.model_name)
        
        print("Training model...")
        trained_model = train_model(model, X, y)
        
        print("Evaluating model...")
        metrics = evaluate_model(trained_model, X, y, request.model_type)
        print(f"Metrics: {metrics}")
        
        # Load preprocessors to bundle with the model
        import joblib
        prep_path = os.path.join(PROCESSED_DIR, f"prep_{request.file_id}.joblib")
        preprocessors = joblib.load(prep_path) if os.path.exists(prep_path) else {}
        
        model_path = export_model(trained_model, preprocessors, request.file_id, request.model_name)
        
        return {
            "metrics": metrics["metrics"],
            "plot_data": metrics["plot_data"],
            "model_path": model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-model")
async def download_trained_model(file_id: str, model_name: str):
    filename = f"{file_id}_{model_name.replace(' ', '_')}.joblib"
    file_path = os.path.join("saved_models", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
        
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')
