from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, List
import numpy as np
import joblib
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Exam Score Predictor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and scaler
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading scaler from: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# Define the input data models
class TermScores(BaseModel):
    term1: float
    term2: float
    term3: float

    @validator('term1', 'term2', 'term3')
    def validate_scores(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Score must be between 0 and 100')
        return v

class SubjectTermScores(BaseModel):
    mathematics: TermScores
    english: TermScores
    science: TermScores
    history: TermScores
    geography: TermScores

class Attendance(BaseModel):
    monday: float
    tuesday: float
    wednesday: float
    thursday: float
    friday: float
    saturday: float
    sunday: float

    @validator('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
    def validate_hours(cls, v):
        if v < 0 or v > 24:
            raise ValueError('Hours must be between 0 and 24')
        return v

class PredictionInput(BaseModel):
    subject_term_scores: SubjectTermScores
    attendance: Attendance
    assignment_completion_rate: float

    @validator('assignment_completion_rate')
    def validate_completion_rate(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Assignment completion rate must be between 0 and 100')
        return v

@app.get("/")
async def root():
    return {"message": "Welcome to the Exam Score Predictor API"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Log input data for debugging
        logger.info(f"Received prediction request with input: {input_data.dict()}")
        
        # Convert input data to numpy array
        features = []
        
        # Add subject scores
        for subject in ['mathematics', 'english', 'science', 'history', 'geography']:
            subject_scores = getattr(input_data.subject_term_scores, subject)
            features.extend([
                subject_scores.term1,
                subject_scores.term2,
                subject_scores.term3
            ])
        
        # Add attendance hours
        attendance = input_data.attendance
        features.extend([
            attendance.monday,
            attendance.tuesday,
            attendance.wednesday,
            attendance.thursday,
            attendance.friday,
            attendance.saturday,
            attendance.sunday
        ])
        
        # Add assignment completion rate
        features.append(input_data.assignment_completion_rate)
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Log feature array for debugging
        logger.info(f"Feature array shape: {X.shape}")
        logger.info(f"Feature values: {X[0]}")
        
        # Scale the features
        X_scaled = scaler.transform(X)
        logger.info(f"Scaled feature values: {X_scaled[0]}")
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        return {
            "predicted_score": float(prediction),
            "input_data": input_data.dict()
        }
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 