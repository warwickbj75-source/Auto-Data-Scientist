"""
Prediction Service
Loads a saved model and scores new data points.
Computes per-feature contributions and risk tiers.
"""
import os
import json
import pandas as pd
import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def load_model(model_id: str) -> dict:
    """Load all artifacts for a saved model."""
    model_dir = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model {model_id} not found")

    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    imputer = joblib.load(os.path.join(model_dir, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

    encoders = {}
    enc_path = os.path.join(model_dir, 'encoders.joblib')
    if os.path.exists(enc_path):
        encoders = joblib.load(enc_path)

    with open(os.path.join(model_dir, 'metadata.json')) as f:
        metadata = json.load(f)

    return {
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'encoders': encoders,
        'metadata': metadata,
    }


def predict_single(model_id: str, inputs: dict) -> dict:
    """Score a single data point and return prediction + contributions."""
    artifacts = load_model(model_id)
    model = artifacts['model']
    meta = artifacts['metadata']
    features = meta['features']

    # Build feature vector
    row = {}
    for f in features:
        if f in inputs:
            row[f] = inputs[f]
        elif f in meta.get('categorical_cols', []):
            row[f] = 0  # default encoded value
        else:
            row[f] = 0.0

    X = pd.DataFrame([row], columns=features)

    # Encode categoricals
    for col, le in artifacts['encoders'].items():
        if col in X.columns:
            try:
                X[col] = le.transform(X[col].astype(str))
            except ValueError:
                X[col] = 0  # unknown category

    # Impute + scale
    X_imp = pd.DataFrame(artifacts['imputer'].transform(X), columns=features)
    X_scaled = pd.DataFrame(artifacts['scaler'].transform(X_imp), columns=features)

    # Predict
    prediction = float(model.predict(X_scaled)[0])

    # Compute contributions (for linear models)
    contributions = []
    if hasattr(model, 'coef_'):
        for i, f in enumerate(features):
            contrib = float(model.coef_[i] * X_scaled.iloc[0, i])
            if abs(contrib) > 0.01:
                contributions.append({
                    'feature': f,
                    'value': round(contrib, 3),
                    'input_value': row[f],
                })
        contributions.sort(key=lambda x: -abs(x['value']))

    # Risk tier based on quartiles from training data
    tier = _compute_tier(prediction, meta)

    return {
        'prediction': round(prediction, 2),
        'mae': meta['performance']['mae'],
        'tier': tier,
        'contributions': contributions,
        'model_id': model_id,
        'algorithm': meta['algorithm'],
    }


def predict_batch(model_id: str, df: pd.DataFrame) -> dict:
    """Score multiple rows from a dataframe."""
    artifacts = load_model(model_id)
    model = artifacts['model']
    meta = artifacts['metadata']
    features = meta['features']

    # Prepare dataframe
    X = df[features].copy() if all(f in df.columns for f in features) else pd.DataFrame()
    if X.empty:
        missing = [f for f in features if f not in df.columns]
        return {'error': f'Missing columns: {missing}'}

    # Encode categoricals
    for col, le in artifacts['encoders'].items():
        if col in X.columns:
            try:
                X[col] = le.transform(X[col].astype(str))
            except ValueError:
                X[col] = 0

    # Impute + scale
    X_imp = pd.DataFrame(artifacts['imputer'].transform(X), columns=features)
    X_scaled = pd.DataFrame(artifacts['scaler'].transform(X_imp), columns=features)

    # Predict
    predictions = model.predict(X_scaled)

    results = []
    for i, pred in enumerate(predictions):
        tier = _compute_tier(float(pred), meta)
        results.append({
            'index': i,
            'prediction': round(float(pred), 2),
            'tier': tier,
        })

    return {
        'predictions': results,
        'count': len(results),
        'model_id': model_id,
        'mean_prediction': round(float(predictions.mean()), 2),
    }


def compute_what_ifs(model_id: str, base_inputs: dict, scenarios: list = None) -> list:
    """Compute what-if scenarios: change one input, see prediction delta."""
    artifacts = load_model(model_id)
    meta = artifacts['metadata']

    # Get base prediction
    base = predict_single(model_id, base_inputs)
    base_pred = base['prediction']

    # Default scenarios: +/- for top features
    if scenarios is None:
        details = meta.get('details', {})
        # Use coefficient magnitude or importance to pick top features
        coefs = details.get('coefficients', {})
        perm = details.get('permutation_importance', {})

        top_features = sorted(
            perm.items(), key=lambda x: -abs(x[1])
        )[:6] if perm else sorted(
            coefs.items(), key=lambda x: -abs(x[1])
        )[:6]

        scenarios = []
        for feat, _ in top_features:
            current = base_inputs.get(feat, 0)
            # Nudge by ~10% of range or 1 unit
            if isinstance(current, (int, float)):
                delta = max(1, abs(current) * 0.2)
                scenarios.append({
                    'feature': feat,
                    'new_value': round(current + delta, 2),
                    'label': f'+{round(delta, 1)} {feat}',
                })

    results = []
    for scenario in scenarios:
        modified = base_inputs.copy()
        modified[scenario['feature']] = scenario['new_value']
        new_pred = predict_single(model_id, modified)

        results.append({
            'label': scenario.get('label', scenario['feature']),
            'feature': scenario['feature'],
            'old_value': base_inputs.get(scenario['feature'], 0),
            'new_value': scenario['new_value'],
            'base_prediction': base_pred,
            'new_prediction': new_pred['prediction'],
            'delta': round(new_pred['prediction'] - base_pred, 2),
        })

    results.sort(key=lambda x: -abs(x['delta']))
    return results


def _compute_tier(prediction: float, meta: dict) -> str:
    """Assign a risk tier based on prediction value."""
    # Use stored leaderboard to infer target range
    perf = meta.get('performance', {})
    mae = perf.get('mae', 5)

    # Simple quartile-based tiers
    # We don't have full training target distribution, so use heuristics
    # based on model MAE to guess boundaries
    r2 = perf.get('r2', 50)

    # For now, use fixed percentile labels
    # In production, store target quantiles during training
    return 'Predicted'


def list_models() -> list:
    """List all saved models with metadata."""
    if not os.path.exists(MODELS_DIR):
        return []

    models = []
    for model_id in os.listdir(MODELS_DIR):
        meta_path = os.path.join(MODELS_DIR, model_id, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            models.append({
                'model_id': meta['model_id'],
                'dataset_name': meta.get('dataset_name', 'Unknown'),
                'target': meta['target'],
                'algorithm': meta['algorithm'],
                'r2': meta['performance']['r2'],
                'mae': meta['performance']['mae'],
                'created_at': meta['created_at'],
                'n_features': len(meta['features']),
            })

    models.sort(key=lambda x: x['created_at'], reverse=True)
    return models


def get_model_metadata(model_id: str) -> dict:
    """Get full metadata for a model."""
    meta_path = os.path.join(MODELS_DIR, model_id, 'metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Model {model_id} not found")
    with open(meta_path) as f:
        return json.load(f)


def delete_model(model_id: str) -> bool:
    """Delete a saved model."""
    import shutil
    model_dir = os.path.join(MODELS_DIR, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        return True
    return False
