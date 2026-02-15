"""
Data Profiling Service
Accepts any tabular CSV and produces a comprehensive profile:
- Column type detection (numeric, categorical, binary, id, datetime)
- Distribution stats (mean, std, skew, percentiles, histogram bins)
- Correlation matrix
- Quality flags (missing, duplicates, outliers, synthetic indicators)
- Categorical breakdowns
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify each column by its semantic type."""
    types = {}
    for col in df.columns:
        series = df[col]
        nunique = series.nunique()
        dtype = series.dtype
        col_lower = col.lower()

        # ID detection: unique count ~= row count AND name suggests ID
        is_id_name = any(x in col_lower for x in ['_id', 'id_', 'index', 'key'])
        is_sequential = False
        if dtype in ['int64', 'float64'] and nunique > len(df) * 0.9:
            vals = series.dropna().sort_values().values
            if len(vals) > 2:
                diffs = np.diff(vals)
                is_sequential = np.std(diffs) < 0.01 * np.mean(np.abs(diffs) + 1e-10)

        if (nunique == len(df) and is_id_name) or (is_sequential and is_id_name):
            types[col] = 'id'
        elif dtype == 'object' or dtype.name == 'category':
            types[col] = 'categorical'
        elif dtype in ['datetime64[ns]', 'datetime64']:
            types[col] = 'datetime'
        elif nunique <= 2 and dtype in ['int64', 'float64']:
            vals = set(series.dropna().unique())
            if vals.issubset({0, 1, 0.0, 1.0, True, False}):
                types[col] = 'binary'
            else:
                types[col] = 'numeric'
        elif dtype in ['int64', 'float64']:
            types[col] = 'numeric'
        else:
            types[col] = 'categorical'  # fallback: treat unknown as categorical

    return types


def compute_distributions(df: pd.DataFrame, col_types: dict) -> dict:
    """Compute histogram bins and stats for numeric columns."""
    distributions = {}
    for col, ctype in col_types.items():
        if ctype not in ('numeric', 'binary'):
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        try:
            counts, bin_edges = np.histogram(series, bins=20)
            distributions[col] = {
                'bins': [round(float(b), 4) for b in bin_edges[:-1]],
                'counts': [int(c) for c in counts],
                'mean': round(float(series.mean()), 4),
                'std': round(float(series.std()), 4),
                'median': round(float(series.median()), 4),
                'min': round(float(series.min()), 4),
                'max': round(float(series.max()), 4),
                'skew': round(float(series.skew()), 4),
                'q25': round(float(series.quantile(0.25)), 4),
                'q75': round(float(series.quantile(0.75)), 4),
            }
        except Exception:
            continue

    return distributions


def compute_correlations(df: pd.DataFrame, col_types: dict) -> dict:
    """Compute correlation matrix for numeric columns."""
    num_cols = [c for c, t in col_types.items() if t in ('numeric', 'binary')]
    if len(num_cols) < 2:
        return {'cols': [], 'matrix': []}

    # Limit to 15 columns for readability
    if len(num_cols) > 15:
        num_cols = num_cols[:15]

    corr = df[num_cols].corr()
    return {
        'cols': [_short_name(c) for c in num_cols],
        'col_names': num_cols,
        'matrix': [[round(float(v), 3) for v in row] for row in corr.values.tolist()]
    }


def compute_categoricals(df: pd.DataFrame, col_types: dict) -> dict:
    """Compute value counts for categorical columns."""
    categoricals = {}
    for col, ctype in col_types.items():
        if ctype != 'categorical':
            continue
        vc = df[col].value_counts().head(10)
        categoricals[col] = {
            'values': {str(k): int(v) for k, v in vc.items()},
            'nunique': int(df[col].nunique()),
            'top': str(vc.index[0]) if len(vc) > 0 else None,
        }
    return categoricals


def compute_quality_flags(df: pd.DataFrame, col_types: dict, distributions: dict) -> list:
    """Detect data quality issues and suspicious patterns."""
    flags = []
    n = len(df)

    # Missing values
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    if total_missing == 0:
        flags.append({
            'type': 'info', 'severity': 'low',
            'message': 'Zero missing values across all columns',
            'detail': 'Unusual for real-world data — may indicate synthetic generation or heavy preprocessing'
        })
    else:
        cols_with_missing = [(c, int(v)) for c, v in missing.items() if v > 0]
        for col, count in cols_with_missing:
            pct = round(count / n * 100, 1)
            sev = 'high' if pct > 30 else 'medium' if pct > 10 else 'low'
            flags.append({
                'type': 'missing', 'severity': sev,
                'message': f'{col}: {count} missing ({pct}%)',
                'detail': f'Consider imputation or exclusion'
            })

    # Duplicates
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        flags.append({
            'type': 'duplicates', 'severity': 'medium',
            'message': f'{n_dupes} duplicate rows ({round(n_dupes/n*100, 1)}%)',
            'detail': 'May indicate data collection issues'
        })

    # Categorical balance (synthetic indicator)
    cats = {c: t for c, t in col_types.items() if t == 'categorical'}
    balanced_cats = 0
    for col in cats:
        vc = df[col].value_counts(normalize=True)
        if len(vc) >= 2 and vc.std() < 0.02:
            balanced_cats += 1
    if balanced_cats >= 2:
        flags.append({
            'type': 'synthetic', 'severity': 'medium',
            'message': f'{balanced_cats} categorical columns have near-perfect balance',
            'detail': 'Unusual in real survey data — may indicate synthetic generation'
        })

    # Near-zero skew (synthetic indicator)
    low_skew = 0
    for col, dist in distributions.items():
        if abs(dist.get('skew', 1)) < 0.1:
            low_skew += 1
    if low_skew >= 3:
        flags.append({
            'type': 'synthetic', 'severity': 'low',
            'message': f'{low_skew} numeric columns have near-zero skew',
            'detail': 'Real-world data typically shows some skewness'
        })

    # Outliers (>3 sigma)
    outlier_cols = []
    for col, ctype in col_types.items():
        if ctype != 'numeric':
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue
        z = np.abs(sp_stats.zscore(series))
        n_outliers = int((z > 3).sum())
        if n_outliers > 0:
            outlier_cols.append(col)
    if outlier_cols:
        flags.append({
            'type': 'outliers', 'severity': 'low',
            'message': f'{len(outlier_cols)} columns contain outliers (>3σ)',
            'detail': f'Columns: {", ".join(outlier_cols[:5])}'
        })

    return flags


def profile_dataset(df: pd.DataFrame) -> dict:
    """Run full profiling pipeline on a dataframe."""
    col_types = detect_column_types(df)
    distributions = compute_distributions(df, col_types)
    correlations = compute_correlations(df, col_types)
    categoricals = compute_categoricals(df, col_types)
    quality_flags = compute_quality_flags(df, col_types, distributions)

    # Summary
    type_counts = {}
    for t in col_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        'summary': {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'type_counts': type_counts,
            'missing_total': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        },
        'column_types': col_types,
        'distributions': distributions,
        'correlations': correlations,
        'categoricals': categoricals,
        'quality_flags': quality_flags,
        'numeric_columns': [c for c, t in col_types.items() if t == 'numeric'],
        'categorical_columns': [c for c, t in col_types.items() if t == 'categorical'],
        'binary_columns': [c for c, t in col_types.items() if t == 'binary'],
        'target_candidates': _suggest_targets(df, col_types),
    }


def _suggest_targets(df: pd.DataFrame, col_types: dict) -> list:
    """Suggest possible target variables for prediction."""
    candidates = []
    for col, ctype in col_types.items():
        if ctype != 'numeric':
            continue
        series = df[col].dropna()
        candidates.append({
            'column': col,
            'type': ctype,
            'range': f'{round(float(series.min()), 1)}–{round(float(series.max()), 1)}',
            'mean': round(float(series.mean()), 2),
            'nunique': int(series.nunique()),
            'missing_pct': round(float(df[col].isnull().sum() / len(df) * 100), 1),
        })

    # Sort: prefer columns with high cardinality, low missing
    candidates.sort(key=lambda x: (-x['nunique'], x['missing_pct']))
    return candidates[:8]


def _short_name(col: str, max_len: int = 10) -> str:
    """Shorten column name for display."""
    col = col.replace('_', ' ').title()
    if len(col) <= max_len:
        return col
    return col[:max_len-1] + '.'
