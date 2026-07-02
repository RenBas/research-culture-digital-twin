# ============================================================
# data.py – data loading and validation
# ============================================================

import pandas as pd
import streamlit as st
from typing import Optional
from .constants import REQUIRED_SURVEY_COLS, REQUIRED_META_COLS, OPTIONAL_META_COLS, VARIABLES
from .utils import month_str_to_num

@st.cache_data(show_spinner="Processing survey data...")
def process_survey(_survey_df):
    if _survey_df is None:
        return None, None, "No survey file uploaded."
    try:
        df = _survey_df.copy()
        missing = [c for c in REQUIRED_SURVEY_COLS if c not in df.columns]
        if missing:
            return None, None, f"Missing columns: {', '.join(missing)}"
        if 'school_id_no' in df.columns:
            df['school_id_no'] = df['school_id_no'].astype(int)
        elif 'school_id' in df.columns:
            df['school_id_no'] = df['school_id'].astype(str).apply(lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x))
        else:
            return None, None, "Need 'school_id_no' or 'school_id'."
        if 'school_name' not in df.columns:
            df['school_name'] = df['school_id_no'].apply(lambda x: f"School_{x}")
        else:
            df['school_name'] = df['school_name'].fillna(df['school_id_no'].apply(lambda x: f"School_{x}"))
        df['month_num'] = df['month'].apply(month_str_to_num)
        for v in VARIABLES:
            if not pd.api.types.is_numeric_dtype(df[v]):
                df[v] = pd.to_numeric(df[v], errors='coerce')
            if df[v].isna().any() or (df[v] < 0).any() or (df[v] > 1).any():
                return None, None, f"Column {v} must be numeric between 0 and 1."
        school_info = df[['school_id_no', 'school_name']].drop_duplicates().sort_values('school_id_no')
        return df, school_info, None
    except Exception as e:
        return None, None, f"Survey error: {str(e)}"

@st.cache_data(show_spinner="Processing metadata...")
def process_metadata(_metadata_df):
    if _metadata_df is None:
        return None, "No metadata file uploaded."
    try:
        df = _metadata_df.copy()
        missing = [c for c in REQUIRED_META_COLS if c not in df.columns]
        if missing:
            return None, f"Missing columns: {', '.join(missing)}"
        if 'school_id_no' not in df.columns:
            if 'school' in df.columns:
                df['school_id_no'] = df['school'].astype(str).apply(lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x))
            else:
                return None, "Need 'school_id_no' or 'school'."
        df['school_id_no'] = df['school_id_no'].astype(int)
        for col, default in OPTIONAL_META_COLS.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
        if df['upload_date'].isna().any():
            return None, "Invalid dates in upload_date."
        if df['utilized_by_school'].dtype != bool:
            df['utilized_by_school'] = df['utilized_by_school'].astype(str).str.lower().map(
                {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}).fillna(False)
        return df, None
    except Exception as e:
        return None, f"Metadata error: {str(e)}"

def get_latest_survey(survey_df, school_id):
    sdf = survey_df[survey_df['school_id_no'] == school_id]
    if sdf.empty:
        return None
    return sdf.sort_values('month_num').iloc[-1]
