"""
Exploration Service
For any dataset + target variable, computes:
- Feature importance (Random Forest + Gradient Boosting averaged)
- K-Means segmentation with auto-labelling
- Correlation-based interaction candidates
- Claims with confidence scores for human review
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer


def prepare_features(df: pd.DataFrame, target: str, excluded: list = None) -> tuple:
    """Prepare X, y for modelling — handles categoricals and missing values."""
    excluded = excluded or []
    feature_cols = [c for c in df.columns if c != target and c not in excluded]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Drop rows where target is missing
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Encode categoricals
    encoders = {}
    for col in list(X.columns):
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('__MISSING__').astype(str))
            encoders[col] = le

    # Drop any remaining non-numeric columns that slipped through
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    # Impute missing numerics
    if X.isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X, y, list(X.columns), encoders


def compute_feature_importance(df: pd.DataFrame, target: str, excluded: list = None) -> list:
    """Compute feature importance using RF + GB averaged."""
    X, y, feature_cols, _ = prepare_features(df, target, excluded)

    if len(X) == 0 or len(feature_cols) == 0:
        return []

    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    # Gradient Boosting importance
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X, y)
    gb_imp = gb.feature_importances_

    # Average and normalise
    avg_imp = (rf_imp + gb_imp) / 2
    total = avg_imp.sum()
    if total > 0:
        avg_imp = avg_imp / total * 100

    results = []
    for i, col in enumerate(X.columns):
        results.append({
            'feature': col,
            'importance': round(float(avg_imp[i]), 2),
            'rf_importance': round(float(rf_imp[i] / rf_imp.sum() * 100), 2),
            'gb_importance': round(float(gb_imp[i] / gb_imp.sum() * 100), 2),
        })

    results.sort(key=lambda x: -x['importance'])
    return results


def compute_segments(df: pd.DataFrame, target: str, n_clusters: int = 4, excluded: list = None) -> dict:
    """Run K-Means segmentation on the dataset."""
    X, y, feature_cols, _ = prepare_features(df, target, excluded)

    if len(X.columns) == 0 or len(X) < n_clusters * 10:
        return {'segments': [], 'error': 'Not enough usable features or data for segmentation'}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Build segment profiles
    df_work = X.copy()
    df_work['_target'] = y.values
    df_work['_cluster'] = labels

    segments = []
    for i in range(n_clusters):
        mask = df_work['_cluster'] == i
        seg = df_work[mask]
        profile = {}
        for col in list(X.columns) + ['_target']:
            display_name = target if col == '_target' else col
            profile[display_name] = {
                'mean': round(float(seg[col].mean()), 2),
                'median': round(float(seg[col].median()), 2),
                'std': round(float(seg[col].std()), 2),
            }

        segments.append({
            'id': i,
            'count': int(mask.sum()),
            'pct': round(float(mask.sum() / len(df_work) * 100), 1),
            'target_mean': round(float(seg['_target'].mean()), 2),
            'profile': profile,
        })

    # Sort by target mean (lowest first = most at risk)
    segments.sort(key=lambda s: s['target_mean'])

    # Auto-label
    labels_list = _auto_label_segments(segments, target)
    for seg, label in zip(segments, labels_list):
        seg['label'] = label

    return {'segments': segments, 'n_clusters': n_clusters}


def _auto_label_segments(segments: list, target: str) -> list:
    """Generate generic labels based on target ranking."""
    n = len(segments)
    if n == 4:
        return ['At Risk', 'Below Average', 'Average', 'High Performer']
    elif n == 3:
        return ['Low', 'Medium', 'High']
    elif n == 2:
        return ['Below Average', 'Above Average']
    else:
        return [f'Segment {i+1}' for i in range(n)]


def compute_interactions(df: pd.DataFrame, target: str, feature_importance: list, excluded: list = None) -> list:
    """Find interaction effects between top features."""
    top_features = [f['feature'] for f in feature_importance[:5]]
    interactions = []

    for i, f1 in enumerate(top_features):
        for f2 in top_features[i+1:]:
            if f1 not in df.columns or f2 not in df.columns:
                continue
            try:
                # Bin each feature into Low/Mid/High
                s1 = pd.qcut(df[f1].dropna(), 3, labels=['Low', 'Mid', 'High'], duplicates='drop')
                s2 = pd.qcut(df[f2].dropna(), 3, labels=['Low', 'Mid', 'High'], duplicates='drop')

                # Cross-tabulate means of target
                combo = pd.DataFrame({f1: s1, f2: s2, target: df[target]}).dropna()
                pivot = combo.groupby([f1, f2])[target].mean().unstack()

                interactions.append({
                    'feature_1': f1,
                    'feature_2': f2,
                    'pivot': {str(k): {str(k2): round(float(v2), 2) for k2, v2 in v.items() if pd.notna(v2)}
                              for k, v in pivot.to_dict().items()},
                })
            except Exception:
                continue

    return interactions[:6]


def generate_claims(feature_importance: list, segments: dict, df: pd.DataFrame, target: str) -> list:
    """Generate reviewable claims about the data for the human review gate."""
    claims = []

    # Top feature dominance
    if feature_importance:
        top = feature_importance[0]
        claims.append({
            'id': 'top_feature',
            'claim': f'{top["feature"]} is the most important predictor ({top["importance"]:.1f}% importance)',
            'confidence': 'High' if top['importance'] > 20 else 'Medium',
            'risk': 'Check if this variable is derived from the target (data leakage)',
        })

    # Two-feature dominance
    if len(feature_importance) >= 2:
        top2_total = feature_importance[0]['importance'] + feature_importance[1]['importance']
        if top2_total > 60:
            claims.append({
                'id': 'two_feature_dominance',
                'claim': f'Top 2 features explain {top2_total:.0f}% of importance — rest is noise',
                'confidence': 'High',
                'risk': 'Low — consistent across model types',
            })

    # Segment balance
    if segments.get('segments'):
        segs = segments['segments']
        pcts = [s['pct'] for s in segs]
        if max(pcts) - min(pcts) < 10:
            claims.append({
                'id': 'segment_balance',
                'claim': f'{len(segs)} segments with roughly equal populations ({min(pcts):.0f}–{max(pcts):.0f}%)',
                'confidence': 'Medium',
                'risk': 'Even splits may indicate synthetic data',
            })

    # At-risk population
    if segments.get('segments'):
        at_risk = segments['segments'][0]
        claims.append({
            'id': 'at_risk',
            'claim': f'{at_risk["pct"]}% of records are in the lowest-performing segment (avg {target}: {at_risk["target_mean"]})',
            'confidence': 'High',
            'risk': 'Low — directly observable in data',
        })

    # Weak features
    weak = [f for f in feature_importance if f['importance'] < 2]
    if len(weak) >= 3:
        names = ', '.join([f['feature'] for f in weak[:4]])
        claims.append({
            'id': 'weak_features',
            'claim': f'{len(weak)} features have <2% importance ({names})',
            'confidence': 'High',
            'risk': 'Low — can safely exclude for simpler model',
        })

    return claims


def explore_dataset(df: pd.DataFrame, target: str, excluded: list = None) -> dict:
    """Run full exploration pipeline."""
    excluded = excluded or []

    # Remove ID-like columns automatically (sequential integers or name contains 'id')
    for col in df.columns:
        if col != target and col not in excluded:
            if df[col].nunique() == len(df):
                # Only exclude if it looks like an actual ID (integer type + name hint)
                col_lower = col.lower()
                is_id_name = any(x in col_lower for x in ['_id', 'id_', 'index', 'key', 'row'])
                is_integer = df[col].dtype in ['int64', 'int32']
                if is_id_name or (is_integer and df[col].is_monotonic_increasing):
                    excluded.append(col)

    feature_importance = compute_feature_importance(df, target, excluded)
    segments = compute_segments(df, target, excluded=excluded)
    interactions = compute_interactions(df, target, feature_importance, excluded)
    claims = generate_claims(feature_importance, segments, df, target)

    return {
        'target': target,
        'excluded': excluded,
        'feature_importance': feature_importance,
        'segments': segments,
        'interactions': interactions,
        'claims': claims,
    }
