# National Exam Score Predictor

This application predicts a student's national exam scores based on their performance in various subjects using machine learning.

## Project Structure

```
exam-predictor/
├── backend/           # FastAPI backend
│   ├── main.py       # FastAPI application
│   ├── train_model.py # Model training script
│   └── model.joblib  # Trained model (generated after running train_model.py)
└── frontend/         # React frontend
    └── src/          # React source files
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy joblib python-multipart
   ```

4. Train the model:
   ```bash
   python train_model.py
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

The backend will be available at http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend will be available at http://localhost:3000

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Enter the student's scores for each subject (0-100)
3. Click "Predict Score" to get the predicted national exam score
4. View the visualization comparing subject scores with the predicted score

## API Documentation

Once the backend is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technologies Used

- Frontend:
  - React
  - TypeScript
  - Material-UI
  - Recharts
  - Axios

- Backend:
  - FastAPI
  - Scikit-learn
  - Pandas
  - NumPy
  - Joblib 