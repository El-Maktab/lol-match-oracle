
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .predict import PredictionService
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureImportanceResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the prediction service on startup.
    This ensures preprocessing/model loading is deterministic and cached safely.
    """
    get_prediction_service()
    yield

app = FastAPI(
    title="League of Legends Match Oracle API",
    description="Inference API for predicting League of Legends match outcomes",
    version="1.0.0",
    lifespan=lifespan,
)

# NOTE: Allow all origins for local dashboard dev; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
_prediction_service: PredictionService = None

def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        try:
            _prediction_service = PredictionService()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return _prediction_service

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    """
    service = get_prediction_service()
    if service and service.model and service.preprocessor:
        return HealthResponse(status="ok")
    return HealthResponse(status="degraded")

@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(service: PredictionService = Depends(get_prediction_service)):
    """
    Get metadata about the currently loaded champion model.
    """
    return ModelInfoResponse(
        model_name=service.metadata.get("model_name", "unknown"),
        run_name=service.metadata.get("run_name", "unknown"),
        experiment_name=service.metadata.get("experiment_name", "unknown"),
        features=service.feature_columns
    )

@app.get("/features/importance", response_model=FeatureImportanceResponse)
def get_feature_importance(service: PredictionService = Depends(get_prediction_service)):
    """
    Get the feature importance from the loaded model.
    """
    importances = service.get_feature_importance()
    return FeatureImportanceResponse(importances=importances)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, service: PredictionService = Depends(get_prediction_service)):
    """
    Predict the outcome for a single match (team-level).
    """
    try:
        return service.predict(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    request: BatchPredictRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict outcomes for a batch of matches (team-level).
    """
    try:
        responses = service.predict_batch(request.requests)
        return BatchPredictResponse(predictions=responses)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")
