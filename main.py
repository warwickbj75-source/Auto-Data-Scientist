"""
Auto Data Scientist — Production API
Single-user local web app for automated data science pipelines.
"""
import os
import json
import uuid
import shutil
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from services.profiler import profile_dataset
from services.explorer import explore_dataset
from services.trainer import train_pipeline
from services.interpreter import generate_generic_report, generate_ai_report
from services.predictor import (
    predict_single, predict_batch, compute_what_ifs,
    list_models, get_model_metadata, delete_model
)

# --- App Setup ---
app = FastAPI(title="Auto Data Scientist", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory session state (single user)
session = {
    'datasets': {},      # dataset_id -> {path, name, profile, exploration}
    'current': None,     # current dataset_id
    'config': {},        # target, excluded, etc.
    'review': {},        # human review decisions
}


# --- Pydantic Models ---
class ConfigRequest(BaseModel):
    target: str
    excluded: list[str] = []
    question: Optional[str] = None

class ReviewRequest(BaseModel):
    decisions: dict  # claim_id -> 'agree' | 'challenge' | 'investigate'

class PredictRequest(BaseModel):
    inputs: dict

class InterpretRequest(BaseModel):
    use_ai: bool = False
    api_key: Optional[str] = None


# --- Upload ---
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV file and get a dataset ID + preview."""
    if not file.filename.endswith(('.csv', '.tsv')):
        raise HTTPException(400, "Only CSV/TSV files supported")

    dataset_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(UPLOADS_DIR, f"{dataset_id}_{file.filename}")

    with open(filepath, 'wb') as f:
        content = await file.read()
        f.write(content)

    # Read and validate
    try:
        sep = '\t' if file.filename.endswith('.tsv') else ','
        df = pd.read_csv(filepath, sep=sep)
    except Exception as e:
        os.remove(filepath)
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    if len(df) < 10:
        os.remove(filepath)
        raise HTTPException(400, "Dataset must have at least 10 rows")

    # Store in session
    session['datasets'][dataset_id] = {
        'path': filepath,
        'name': file.filename,
        'rows': len(df),
        'cols': len(df.columns),
        'columns': list(df.columns),
    }
    session['current'] = dataset_id

    # Quick preview
    preview = {
        'dataset_id': dataset_id,
        'name': file.filename,
        'rows': len(df),
        'cols': len(df.columns),
        'columns': [
            {
                'name': col,
                'dtype': str(df[col].dtype),
                'sample': str(df[col].dropna().iloc[:3].tolist()) if len(df[col].dropna()) > 0 else '[]',
                'missing': int(df[col].isnull().sum()),
            }
            for col in df.columns
        ],
        'head': df.head(5).to_dict(orient='records'),
    }

    return preview


# --- Profile ---
@app.post("/api/profile/{dataset_id}")
async def profile(dataset_id: str):
    """Run full profiling on a dataset."""
    ds = session['datasets'].get(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    df = pd.read_csv(ds['path'])
    result = profile_dataset(df)

    # Cache profile
    session['datasets'][dataset_id]['profile'] = result
    return result


# --- Config ---
@app.post("/api/config/{dataset_id}")
async def set_config(dataset_id: str, config: ConfigRequest):
    """Set target variable and exclusions."""
    ds = session['datasets'].get(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    session['config'] = {
        'target': config.target,
        'excluded': config.excluded,
        'question': config.question,
        'dataset_id': dataset_id,
    }

    return {'status': 'ok', 'config': session['config']}


# --- Explore ---
@app.post("/api/explore/{dataset_id}")
async def explore(dataset_id: str):
    """Run exploration: feature importance, segments, claims."""
    ds = session['datasets'].get(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    config = session.get('config', {})
    target = config.get('target')
    if not target:
        raise HTTPException(400, "Set target variable first via /api/config")

    df = pd.read_csv(ds['path'])
    excluded = config.get('excluded', [])

    result = explore_dataset(df, target, excluded)
    session['datasets'][dataset_id]['exploration'] = result
    return result


# --- Human Review ---
@app.post("/api/review/{dataset_id}")
async def submit_review(dataset_id: str, review: ReviewRequest):
    """Submit human review decisions on AI claims."""
    session['review'] = review.decisions

    # Auto-apply overrides: if user challenged a feature, exclude it
    config = session.get('config', {})
    exploration = session['datasets'].get(dataset_id, {}).get('exploration', {})
    claims = exploration.get('claims', [])
    excluded = list(config.get('excluded', []))

    for claim in claims:
        cid = claim['id']
        if review.decisions.get(cid) == 'challenge':
            # If the claim references a top feature, exclude it
            if cid == 'top_feature':
                imp = exploration.get('feature_importance', [])
                if imp:
                    feat = imp[0]['feature']
                    if feat not in excluded:
                        excluded.append(feat)

    session['config']['excluded'] = excluded

    return {
        'status': 'ok',
        'decisions': review.decisions,
        'excluded': excluded,
        'overrides': sum(1 for v in review.decisions.values() if v == 'challenge'),
    }


# --- Train ---
@app.post("/api/model/{dataset_id}")
async def train_model(dataset_id: str):
    """Train models and save the best one."""
    ds = session['datasets'].get(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    config = session.get('config', {})
    target = config.get('target')
    if not target:
        raise HTTPException(400, "Set target variable first")

    df = pd.read_csv(ds['path'])
    excluded = config.get('excluded', [])

    result = train_pipeline(df, target, excluded, dataset_name=ds['name'])
    session['datasets'][dataset_id]['training'] = result
    return result


# --- Interpret ---
@app.post("/api/interpret/{dataset_id}")
async def interpret(dataset_id: str, req: InterpretRequest = InterpretRequest()):
    """Generate interpretation report."""
    ds = session['datasets'].get(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    profile_data = ds.get('profile', {})
    exploration = ds.get('exploration', {})
    training = ds.get('training', {})

    if not training:
        raise HTTPException(400, "Train a model first")

    if req.use_ai:
        result = generate_ai_report(profile_data, exploration, training, req.api_key)
    else:
        result = generate_generic_report(profile_data, exploration, training)

    return result


# --- Predict ---
@app.post("/api/predict/{model_id}")
async def predict(model_id: str, req: PredictRequest):
    """Score a single data point."""
    try:
        result = predict_single(model_id, req.inputs)
        return result
    except FileNotFoundError:
        raise HTTPException(404, f"Model {model_id} not found")


@app.post("/api/predict/{model_id}/what-if")
async def what_if(model_id: str, req: PredictRequest):
    """Compute what-if scenarios."""
    try:
        result = compute_what_ifs(model_id, req.inputs)
        return result
    except FileNotFoundError:
        raise HTTPException(404, f"Model {model_id} not found")


# --- Model Registry ---
@app.get("/api/models")
async def get_models():
    """List all saved models."""
    return list_models()


@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    try:
        return get_model_metadata(model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model {model_id} not found")


@app.delete("/api/models/{model_id}")
async def remove_model(model_id: str):
    """Delete a saved model."""
    if delete_model(model_id):
        return {'status': 'deleted', 'model_id': model_id}
    raise HTTPException(404, f"Model {model_id} not found")


# --- Session ---
@app.get("/api/session")
async def get_session():
    """Get current session state (for frontend)."""
    return {
        'current_dataset': session.get('current'),
        'config': session.get('config', {}),
        'review': session.get('review', {}),
        'datasets': {
            k: {kk: vv for kk, vv in v.items() if kk != 'path'}
            for k, v in session['datasets'].items()
        },
    }


@app.delete("/api/session")
async def reset_session():
    """Reset session (start over)."""
    session['datasets'] = {}
    session['current'] = None
    session['config'] = {}
    session['review'] = {}
    return {'status': 'reset'}


# --- Health ---
@app.get("/api/health")
async def health():
    return {
        'status': 'ok',
        'models_saved': len(list_models()),
        'timestamp': datetime.now().isoformat(),
    }


# --- Serve Frontend ---
STATIC_DIR = os.path.join(BASE_DIR, 'static')

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


# Serve static assets (must be after API routes)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    
if __name__ == '__main__':\
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
