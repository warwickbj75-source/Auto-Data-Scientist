"""
Interpretation Service
Level 1: Generic data-driven report with actual numbers
Level 2: AI-powered narrative via Claude API (optional)
"""
import os
import json


def generate_generic_report(profile: dict, exploration: dict, training: dict) -> dict:
    """Generate a Level 1 generic interpretation report using actual data."""
    target = training.get('target', 'target')
    best = training.get('best_model', {})
    features = training.get('features', [])
    importance = exploration.get('feature_importance', [])
    segments = exploration.get('segments', {}).get('segments', [])
    flags = profile.get('quality_flags', [])

    # Central finding
    top2 = importance[:2] if len(importance) >= 2 else importance
    top2_names = ' and '.join([f['feature'] for f in top2])
    top2_total = sum(f['importance'] for f in top2)

    central_finding = (
        f"The primary drivers of {target} are {top2_names}, "
        f"together accounting for {top2_total:.0f}% of model importance. "
        f"The best model ({best.get('name', 'Unknown')}) achieves R² = {best.get('r2', 0)/100:.3f} "
        f"with MAE = {best.get('mae', 0):.2f}."
    )

    # Key metrics
    metrics = [
        {'value': f"R² {best.get('r2', 0)/100:.2f}", 'label': 'Model Accuracy',
         'detail': f'{best.get("name", "Unknown")} regression'},
        {'value': f"{best.get('mae', 0):.1f}", 'label': 'Mean Abs. Error',
         'detail': f'Average prediction error'},
    ]
    if segments:
        at_risk = segments[0]
        metrics.append({
            'value': f"{at_risk['pct']}%", 'label': 'Lowest Segment',
            'detail': f"Avg {target}: {at_risk['target_mean']}"
        })
    if len(importance) > 0:
        metrics.append({
            'value': f"{importance[0]['importance']:.0f}%", 'label': 'Top Feature',
            'detail': importance[0]['feature']
        })

    # Findings
    findings = []
    if len(importance) >= 2:
        findings.append({
            'title': f'{top2_names} dominate prediction',
            'text': (
                f"These two features account for {top2_total:.0f}% of model importance. "
                f"Remaining {len(importance) - 2} features contribute {100 - top2_total:.0f}% combined."
            ),
            'implication': 'Focus interventions on these primary drivers for maximum impact.'
        })

    if len(importance) >= 3:
        weak = [f for f in importance if f['importance'] < 5]
        if weak:
            findings.append({
                'title': f'{len(weak)} features have minimal impact',
                'text': (
                    f"Features like {', '.join([f['feature'] for f in weak[:3]])} "
                    f"each contribute less than 5% importance."
                ),
                'implication': 'Interventions targeting these variables are unlikely to move outcomes.'
            })

    if segments and len(segments) >= 2:
        best_seg = segments[-1]
        worst_seg = segments[0]
        gap = best_seg['target_mean'] - worst_seg['target_mean']
        findings.append({
            'title': f'Performance gap of {gap:.1f} between segments',
            'text': (
                f"The highest-performing segment averages {best_seg['target_mean']:.1f} on {target} "
                f"while the lowest averages {worst_seg['target_mean']:.1f}. "
                f"The gap represents the intervention opportunity."
            ),
            'implication': 'Target resources at the lowest-performing segment for maximum uplift.'
        })

    # Recommendations (generic)
    recommendations = []
    for i, feat in enumerate(importance[:4]):
        priority = 'HIGH' if i < 2 else 'MEDIUM' if i < 3 else 'LOW'
        recommendations.append({
            'priority': priority,
            'title': f"Address {feat['feature']}",
            'metric': f"{feat['importance']:.0f}%",
            'metric_label': 'Model importance',
            'description': (
                f"{feat['feature']} accounts for {feat['importance']:.1f}% of prediction importance. "
                f"Changes to this variable have the largest potential impact on {target}."
            ),
        })

    # Risks
    risks = []
    synthetic_flags = [f for f in flags if f.get('type') == 'synthetic']
    if synthetic_flags:
        risks.append({
            'severity': 'HIGH',
            'title': 'Potential synthetic data',
            'body': '. '.join([f['message'] for f in synthetic_flags]),
        })
    risks.append({
        'severity': 'HIGH',
        'title': 'Correlation does not imply causation',
        'body': 'All relationships are correlational. Interventions must be tested with controlled trials.'
    })
    risks.append({
        'severity': 'MEDIUM',
        'title': 'Unobserved confounders',
        'body': f'The model uses {len(features)} features. Important variables may be missing from the dataset.'
    })
    if best.get('r2', 0) > 90:
        risks.append({
            'severity': 'MEDIUM',
            'title': 'Suspiciously high accuracy',
            'body': f"R² of {best['r2']/100:.3f} is unusually high for behavioural data. Check for data leakage."
        })

    # Segment profiles
    segment_profiles = []
    for seg in segments:
        top_features_for_seg = {}
        for feat in importance[:5]:
            fname = feat['feature']
            if fname in seg.get('profile', {}):
                top_features_for_seg[fname] = seg['profile'][fname]
        segment_profiles.append({
            'label': seg.get('label', f"Segment {seg['id']}"),
            'count': seg['count'],
            'pct': seg['pct'],
            'target_mean': seg['target_mean'],
            'top_features': top_features_for_seg,
        })

    return {
        'level': 1,
        'central_finding': central_finding,
        'metrics': metrics,
        'findings': findings,
        'recommendations': recommendations,
        'risks': risks,
        'segment_profiles': segment_profiles,
        'methodology': _build_methodology(profile, training),
    }


def generate_ai_report(profile: dict, exploration: dict, training: dict,
                        api_key: str = None) -> dict:
    """Generate a Level 2 AI-powered narrative report using Claude."""
    if not api_key:
        api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not api_key:
        # Fall back to generic
        report = generate_generic_report(profile, exploration, training)
        report['ai_note'] = 'No API key provided — using generic report. Set ANTHROPIC_API_KEY for AI narrative.'
        return report

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build context for Claude
        context = _build_ai_context(profile, exploration, training)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            system=(
                "You are a senior data scientist writing an interpretation report. "
                "Be specific with numbers. Write in clear, professional prose. "
                "Structure your response as JSON with keys: "
                "central_finding (string), findings (array of {title, text, implication}), "
                "recommendations (array of {priority, title, description}), "
                "risks (array of {severity, title, body}). "
                "Use the actual data provided — do not invent numbers."
            ),
            messages=[{"role": "user", "content": context}]
        )

        # Parse response
        response_text = message.content[0].text
        # Try to extract JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            ai_content = json.loads(json_match.group())
        else:
            ai_content = {'central_finding': response_text}

        # Merge AI narrative with data-driven metrics
        generic = generate_generic_report(profile, exploration, training)
        generic['level'] = 2
        generic['central_finding'] = ai_content.get('central_finding', generic['central_finding'])
        if ai_content.get('findings'):
            generic['findings'] = ai_content['findings']
        if ai_content.get('recommendations'):
            generic['recommendations'] = ai_content['recommendations']
        if ai_content.get('risks'):
            generic['risks'] = ai_content['risks']

        return generic

    except Exception as e:
        report = generate_generic_report(profile, exploration, training)
        report['ai_error'] = str(e)
        return report


def _build_ai_context(profile: dict, exploration: dict, training: dict) -> str:
    """Build a concise context string for the AI."""
    target = training.get('target', 'target')
    best = training.get('best_model', {})
    importance = exploration.get('feature_importance', [])[:8]
    segments = exploration.get('segments', {}).get('segments', [])
    summary = profile.get('summary', {})
    flags = profile.get('quality_flags', [])

    parts = [
        f"Dataset: {summary.get('rows', '?')} rows, {summary.get('columns', '?')} columns.",
        f"Target variable: {target}.",
        f"Best model: {best.get('name', '?')} with R²={best.get('r2', 0)/100:.3f}, MAE={best.get('mae', 0):.2f}.",
        f"Top features by importance: {json.dumps(importance[:6])}.",
    ]
    if segments:
        parts.append(f"Segments: {json.dumps([{k: v for k, v in s.items() if k != 'profile'} for s in segments])}.")
    if flags:
        parts.append(f"Quality flags: {json.dumps(flags[:5])}.")

    return '\n'.join(parts)


def _build_methodology(profile: dict, training: dict) -> list:
    """Build methodology section."""
    summary = profile.get('summary', {})
    best = training.get('best_model', {})
    features = training.get('features', [])
    excluded = training.get('excluded', [])

    return [
        {
            'section': 'Data Pipeline',
            'items': [
                f"Source: {summary.get('rows', '?')} rows × {summary.get('columns', '?')} columns",
                f"Missing values: {summary.get('missing_total', 0)}",
                f"Duplicate rows: {summary.get('duplicate_rows', 0)}",
                f"Features used: {len(features)}",
                f"Excluded: {', '.join(excluded) if excluded else 'None'}",
            ]
        },
        {
            'section': 'Model Training',
            'items': [
                f"Train/test split: 80/20",
                f"Algorithms tested: 9 (Linear, Ridge, Lasso, ElasticNet, DTree, RF, GB, KNN, AdaBoost)",
                f"Evaluation: 5-fold cross-validation",
                f"Winner: {best.get('name', '?')} (R²={best.get('r2', 0)/100:.3f})",
            ]
        },
    ]
