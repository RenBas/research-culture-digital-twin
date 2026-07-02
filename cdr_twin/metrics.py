# ============================================================
# metrics.py – compute research metrics from metadata
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
from .utils import interpret_utilisation_rate

@st.cache_data(show_spinner=False)
def _compute_research_metrics(metadata_df, school_id):
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id]
    metrics = {}
    if school_meta.empty:
        return metrics
    tc = school_meta['theme'].value_counts().reset_index()
    tc.columns = ['Theme', 'Count']
    metrics['theme_counts'] = tc
    metrics['top_theme'] = tc.iloc[0]['Theme'] if not tc.empty else "N/A"
    if 'utilized_by_school' in school_meta.columns:
        tu = school_meta.groupby('theme')['utilized_by_school'].mean().reset_index()
        tu.columns = ['Theme', 'Utilisation Rate']
        metrics['theme_util_df'] = tu
        metrics['top_util_theme'] = tu.loc[tu['Utilisation Rate'].idxmax(), 'Theme'] if not tu.empty else "N/A"
    else:
        metrics['theme_util_df'] = None; metrics['top_util_theme'] = "N/A"
    sc = school_meta['status'].value_counts().reset_index()
    sc.columns = ['Status', 'Count']
    metrics['status_counts'] = sc
    published = sc[sc['Status'] == 'published']['Count'].sum() if not sc.empty else 0
    total = sc['Count'].sum() if not sc.empty else 0
    metrics['pub_rate'] = (published / total * 100) if total > 0 else 0
    metrics['total_outputs'] = total
    if 'upload_date' in school_meta.columns:
        sm = school_meta.copy()
        sm['quarter'] = sm['upload_date'].dt.to_period('Q').astype(str)
        ot = sm.groupby('quarter').size().reset_index(name='count')
        metrics['output_timeline'] = ot if not ot.empty else None
        if 'utilized_by_school' in school_meta.columns:
            ut = sm.groupby('quarter')['utilized_by_school'].mean().reset_index()
            ut.columns = ['quarter', 'utilisation_rate']
            metrics['util_timeline'] = ut if not ut.empty else None
        else:
            metrics['util_timeline'] = None
    else:
        metrics['output_timeline'] = None; metrics['util_timeline'] = None
    utilised = school_meta['utilized_by_school'].sum() if 'utilized_by_school' in school_meta.columns else 0
    total = len(school_meta)
    util_rate = (utilised / total * 100) if total > 0 else 0
    level, desc = interpret_utilisation_rate(util_rate)
    metrics['util_rate'] = util_rate; metrics['util_level'] = level; metrics['util_desc'] = desc
    tc2 = school_meta['teacher_name'].value_counts().reset_index().head(10)
    tc2.columns = ['Teacher', 'Number of Outputs']
    metrics['teacher_counts'] = tc2
    metrics['top_teacher'] = tc2.iloc[0]['Teacher'] if not tc2.empty else "N/A"
    if 'years_of_service' in school_meta.columns and not school_meta['years_of_service'].isna().all():
        ts = school_meta.groupby('teacher_name').agg(
            output_count=('document_type', 'count'), years_of_service=('years_of_service', 'first')
        ).dropna()
        if len(ts) > 1:
            x, y = ts['years_of_service'], ts['output_count']
            z = np.polyfit(x, y, 1)
            trend_x = np.linspace(x.min(), x.max(), 100)
            metrics['service_data'] = {'teacher_summary': ts, 'trend_x': trend_x,
                                       'trend_y': np.poly1d(z)(trend_x), 'slope': z[0],
                                       'avg_service': x.mean(), 'avg_output': y.mean()}
        else:
            metrics['service_data'] = None
    else:
        metrics['service_data'] = None
    if 'teacher_rank' in school_meta.columns and not school_meta['teacher_rank'].isna().all():
        rg = school_meta.groupby('teacher_rank').size().reset_index(name='total_outputs')
        rn = school_meta.groupby('teacher_rank')['teacher_name'].nunique().reset_index(name='num_teachers')
        rs = rg.merge(rn, on='teacher_rank')
        rs['avg_outputs'] = rs['total_outputs'] / rs['num_teachers']
        metrics['rank_summary'] = rs
    else:
        metrics['rank_summary'] = None
    if 'educational_attainment' in school_meta.columns and not school_meta['educational_attainment'].isna().all():
        eg = school_meta.groupby('educational_attainment').size().reset_index(name='total_outputs')
        en = school_meta.groupby('educational_attainment')['teacher_name'].nunique().reset_index(name='num_teachers')
        es = eg.merge(en, on='educational_attainment')
        es['avg_outputs'] = es['total_outputs'] / es['num_teachers']
        metrics['edu_summary'] = es
    else:
        metrics['edu_summary'] = None
    return metrics
