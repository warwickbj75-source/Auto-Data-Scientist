"""
Model Training Service
Trains multiple algorithms, evaluates with cross-validation,
selects the best, computes feature importance, and saves versioned models.
"""
import os
import json
import uuid
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def _build_algorithms() -> dict:
    """Return dict of algorithm name -> sklearn estimator."""
    return {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=10),
    }


def prepare_training_data(df: pd.DataFrame, target: str, excluded: list = None) -> dict:
    """Prepare data for training with encoding and imputation."""
    excluded = excluded or []
    feature_cols = [c for c in df.columns if c != target and c not in excluded]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Drop rows where target is missing
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Encode categoricals
    encoders = {}
    cat_cols = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('__MISSING__').astype(str))
            encoders[col] = le
            cat_cols.append(col)

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Impute + scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns)

    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'X_full': X, 'y_full': y,
        'feature_cols': list(X.columns),
        'imputer': imputer, 'scaler': scaler, 'encoders': encoders,
        'cat_cols': cat_cols,
        'n_train': len(X_train), 'n_test': len(X_test),
    }


def train_and_evaluate(data: dict) -> dict:
    """Train all algorithms and return leaderboard."""
    X_tr, X_te = data['X_train'], data['X_test']
    y_tr, y_te = data['y_train'], data['y_test']

    algorithms = _build_algorithms()
    results = []

    for name, model in algorithms.items():
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            # Test metrics
            r2 = r2_score(y_te, y_pred)
            mae = mean_absolute_error(y_te, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

            # Cross-validation
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2')

            results.append({
                'name': name,
                'r2': round(float(r2) * 100, 2),
                'mae': round(float(mae), 3),
                'rmse': round(float(rmse), 3),
                'cv_mean': round(float(cv_scores.mean()) * 100, 2),
                'cv_std': round(float(cv_scores.std()) * 100, 2),
                'model': model,
            })
        except Exception as e:
            results.append({
                'name': name, 'r2': 0, 'mae': 999, 'rmse': 999,
                'cv_mean': 0, 'cv_std': 0, 'model': None, 'error': str(e),
            })

    # Sort by test R²
    results.sort(key=lambda x: -x['r2'])
    return results


def compute_model_details(best_model, data: dict) -> dict:
    """Compute feature importance and coefficients for the best model."""
    model = best_model['model']
    X_te, y_te = data['X_test'], data['y_test']
    features = data['feature_cols']

    details = {}

    # Coefficients (for linear models)
    if hasattr(model, 'coef_'):
        coefs = {}
        for i, col in enumerate(features):
            coefs[col] = round(float(model.coef_[i]), 4)
        details['coefficients'] = coefs
        if hasattr(model, 'intercept_'):
            details['intercept'] = round(float(model.intercept_), 4)

    # Built-in feature importance (for tree models)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        total = imp.sum()
        details['builtin_importance'] = {
            col: round(float(imp[i] / total * 100), 2)
            for i, col in enumerate(features)
        }

    # Permutation importance (model-agnostic)
    try:
        perm = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
        perm_imp = perm.importances_mean
        total = perm_imp.sum()
        if total > 0:
            details['permutation_importance'] = {
                col: round(float(perm_imp[i] / total * 100), 2)
                for i, col in enumerate(features)
            }
    except Exception:
        pass

    return details


def save_model(model_obj, data: dict, leaderboard: list, target: str,
               dataset_name: str, excluded: list, details: dict) -> str:
    """Save model artifacts and metadata to disk. Returns model_id."""
    model_id = str(uuid.uuid4())[:8]
    model_dir = os.path.join(MODELS_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Save sklearn objects
    joblib.dump(model_obj['model'], os.path.join(model_dir, 'model.joblib'))
    joblib.dump(data['imputer'], os.path.join(model_dir, 'imputer.joblib'))
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.joblib'))
    if data['encoders']:
        joblib.dump(data['encoders'], os.path.join(model_dir, 'encoders.joblib'))

    # Save metadata
    metadata = {
        'model_id': model_id,
        'created_at': datetime.now().isoformat(),
        'dataset_name': dataset_name,
        'target': target,
        'features': data['feature_cols'],
        'excluded': excluded,
        'categorical_cols': data['cat_cols'],
        'n_train': data['n_train'],
        'n_test': data['n_test'],
        'algorithm': model_obj['name'],
        'performance': {
            'r2': model_obj['r2'],
            'mae': model_obj['mae'],
            'rmse': model_obj['rmse'],
            'cv_mean': model_obj['cv_mean'],
            'cv_std': model_obj['cv_std'],
        },
        'leaderboard': [{k: v for k, v in m.items() if k != 'model'} for m in leaderboard],
        'details': details,
    }
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return model_id


def train_pipeline(df: pd.DataFrame, target: str, excluded: list = None,
                   dataset_name: str = 'unknown') -> dict:
    """Full training pipeline: prepare → train → evaluate → save."""
    excluded = excluded or []

    # Auto-exclude ID columns
    for col in df.columns:
        if col != target and col not in excluded:
            if df[col].nunique() == len(df):
                col_lower = col.lower()
                is_id_name = any(x in col_lower for x in ['_id', 'id_', 'index', 'key', 'row'])
                is_integer = df[col].dtype in ['int64', 'int32']
                if is_id_name or (is_integer and df[col].is_monotonic_increasing):
                    excluded.append(col)

    data = prepare_training_data(df, target, excluded)
    leaderboard = train_and_evaluate(data)

    # Pick best model
    best = leaderboard[0]
    details = compute_model_details(best, data)

    # Save
    model_id = save_model(best, data, leaderboard, target, dataset_name, excluded, details)

    # Build response (strip sklearn objects)
    clean_leaderboard = [{k: v for k, v in m.items() if k != 'model'} for m in leaderboard]

    return {
        'model_id': model_id,
        'best_model': {
            'name': best['name'],
            'r2': best['r2'],
            'mae': best['mae'],
            'rmse': best['rmse'],
            'cv_mean': best['cv_mean'],
        },
        'leaderboard': clean_leaderboard,
        'details': details,
        'features': data['feature_cols'],
        'target': target,
        'excluded': excluded,
        'n_train': data['n_train'],
        'n_test': data['n_test'],
    }
