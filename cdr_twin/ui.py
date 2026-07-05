# ============================================================
# ui.py – main Streamlit user interface (with school map)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import time
import os                                 # <-- moved to top for clarity
from typing import List, Optional

# --- folium for interactive map ---
import folium
from streamlit_folium import st_folium

# Import all modules
from .constants import (
    USTP_DARK_BLUE, USTP_GOLD, DEPED_RED, DEPED_MAROON,
    VARIABLES, VAR_FULL_NAMES, VAR_INTERPRETATION, MILESTONE_NAMES,
    VAR_COLORS, RCSI_LEVELS
)
from .utils import classify_rcsi, interpret_avg_milestone, get_rcsi_interpretation_table, create_glossary, classify_utilisation
from .gauges import create_gauge, create_utilisation_gauge, create_rcsi_gauge, display_gauge_with_interpretation
from .data import process_survey, process_metadata, get_latest_survey
from .metrics import _compute_research_metrics
from .simulation import Simulation, create_empty_history, init_simulation_with_data, record_history, apply_survey_override
from .analysis import generate_baseline_synopsis, baseline_heatmap, cycle_research_correlation, division_level_analysis, school_comparison_gauge, generate_division_baseline_synopsis
from .monte_carlo import calibrate_coefficients, get_agent_params, run_sensitivity, monte_carlo_sim, plot_monte_carlo_bands, causal_analysis

def apply_theme(dark_mode: bool) -> None:
    if dark_mode:
        st.markdown(f"""
        <style>
            .stApp {{ background-color: #1E1E1E !important; color: #FFFFFF !important; }}
            .sidebar .sidebar-content {{ background-color: #2E2E2E !important; border-right: 2px solid #F5A623 !important; }}
            .sidebar .sidebar-content * {{ color: #FFFFFF !important; }}
            h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: #F5A623 !important; }}
            .stMarkdown, .stText, .stCaption, .stDataFrame {{ color: #FFFFFF !important; }}
            .stButton > button {{ background-color: #0D2B5E !important; color: #FFFFFF !important; border: 1px solid #F5A623 !important; }}
            .stButton > button:hover {{ background-color: #F5A623 !important; color: #0D2B5E !important; }}
            .stMetric {{ background-color: #2E2E2E !important; border: 1px solid #F5A623 !important; border-radius: 5px; padding: 10px; }}
            .stMetric label {{ color: #FFFFFF !important; }}
            .dataframe {{ background-color: #2E2E2E !important; color: #FFFFFF !important; }}
            .dataframe thead tr th {{ background-color: #0D2B5E !important; color: #FFFFFF !important; }}
            .dataframe tbody tr {{ background-color: #2E2E2E !important; }}
            .dataframe tbody tr:hover {{ background-color: #3E3E3E !important; }}
            .streamlit-expanderHeader {{ background-color: #2E2E2E !important; color: #FFFFFF !important; border: 1px solid #F5A623 !important; }}
            .streamlit-expanderContent {{ background-color: #1E1E1E !important; color: #FFFFFF !important; }}
            .stAlert {{ background-color: #2E2E2E !important; color: #FFFFFF !important; border: 1px solid #F5A623 !important; }}
            .stSelectbox label, .stNumberInput label, .stCheckbox label {{ color: #FFFFFF !important; }}
            .stRadio label {{ color: #FFFFFF !important; }}
            .stFileUploader {{ background-color: #2E2E2E !important; border: 1px dashed #F5A623 !important; }}
            .stFileUploader label {{ color: #FFFFFF !important; }}
            .stCaption {{ color: #CCCCCC !important; }}
            .main .block-container {{ background-color: #1E1E1E !important; }}
            .css-1y4p8pa {{ background-color: #2E2E2E !important; }}
            div[style*="background-color: #E3F2FD"] {{ background-color: #2E2E2E !important; border-left: 5px solid #F5A623 !important; color: #FFFFFF !important; }}
            div[style*="background-color: #E8F5E9"] {{ background-color: #2E2E2E !important; border-left: 5px solid #F5A623 !important; color: #FFFFFF !important; }}
            table {{ background-color: #2E2E2E !important; color: #FFFFFF !important; border: 1px solid #F5A623 !important; }}
            table th {{ background-color: #0D2B5E !important; color: #FFFFFF !important; }}
            table td {{ background-color: #2E2E2E !important; color: #FFFFFF !important; }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background-color: #FFFFFF; }
            .sidebar .sidebar-content { background-color: #F8F9FA; }
            .stButton > button { background-color: #0D2B5E; color: white; }
            .stButton > button:hover { background-color: #F5A623; color: #0D2B5E; }
        </style>
        """, unsafe_allow_html=True)

def get_figure_download_link(fig, filename="chart.html", link_text="Download chart"):
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    b64 = base64.b64encode(html_str.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():
    st.set_page_config(page_title="CDO Research Culture Sustainability Framework", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #0D2B5E;'>CDO Division Research Culture Sustainability Framework</h1>", unsafe_allow_html=True)

    # Session state initialisation
    for key, default in [('max_schools', 200), ('num_schools', 0), ('total_teachers', 0)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"<h2 style='color: {USTP_DARK_BLUE};'>Controls</h2>", unsafe_allow_html=True)
        dark_mode = st.checkbox("Dark Mode", value=False)
        apply_theme(dark_mode)

        user_role = st.radio("User Role", options=["Principal", "Division Head"], index=0,
                             help="Principal sees only the selected school. Division Head sees division-level aggregates.")

        st.metric("Total Schools Loaded", st.session_state.num_schools)
        st.metric("Total Teachers Recorded", st.session_state.total_teachers)
        if st.session_state.get('total_months', 0) > 0:
            st.metric("Simulation Month", st.session_state.total_months)

        st.markdown("---")
        with st.expander("📚 Glossary of Terms"):
            glossary = create_glossary()
            for term, definition in glossary.items():
                st.markdown(f"**{term}:** {definition}")

        st.markdown("---")
        st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Policy Levers</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            u_train = st.slider("Training freq.", 0.0, 1.0, 0.5, 0.05, help="Frequency of research training workshops per quarter")
            u_mentor = st.slider("Mentorship ratio", 0.0, 1.0, 0.5, 0.05, help="Ratio of experienced-to-novice researcher pairings")
            u_budget = st.slider("Support budget", 0.0, 1.0, 0.5, 0.05, help="Proportion of budget allocated to research support")
        with col2:
            u_lead = st.slider("Leadership commit.", 0.0, 1.0, 0.5, 0.05, help="Degree of school leadership commitment")
            u_collab = st.slider("Collaboration freq.", 0.0, 1.0, 0.5, 0.05, help="Frequency of inter-school collaboration events")
        levers = {'u_train': u_train, 'u_mentor': u_mentor, 'u_budget': u_budget, 'u_lead': u_lead, 'u_collab': u_collab}

        st.markdown("---")
        st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Simulation Parameters</h3>", unsafe_allow_html=True)
        duration = st.selectbox("Run duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=9)
        random_events = st.checkbox("Enable random events", value=False)
        use_survey = st.checkbox("Override with survey data", value=True)

        st.markdown("---")
        st.markdown("#### 🎲 Monte Carlo (Phase 2)")
        mc_enabled = st.checkbox("Enable Monte Carlo", value=False)
        mc_runs = st.number_input("Number of runs", min_value=10, max_value=100, value=30, step=10)

        st.markdown("---")
        st.markdown("#### Simulation Actions")
        col_buttons = st.columns(3)
        with col_buttons[0]:
            run_btn = st.button("Run", use_container_width=True)
        with col_buttons[1]:
            step_btn = st.button("Step (1 month)", use_container_width=True)
        with col_buttons[2]:
            reset_btn = st.button("Reset", use_container_width=True)
        st.caption("Run: full forecast. Step: one month. Reset: clear history.")

        st.markdown("---")
        st.markdown("#### Export Data")
        export_btn = st.button("Export results (CSV)", use_container_width=True)

    # --- File Upload ---
    with st.expander("Step 1: Upload your CSV files", expanded=True):
        st.markdown("""
        **Instructions:**
        - Upload **Quarterly Survey** CSV (columns: `month, school_id_no, R, A, C, S, I, P, M`).
        - Upload **Research Metadata** CSV (columns: `upload_date, teacher_name, school_id_no, ...`).
        """)
        col1, col2 = st.columns(2)
        with col1:
            survey_file = st.file_uploader("Upload quarterly survey (CSV)", type=["csv"], key="survey")
        with col2:
            metadata_file = st.file_uploader("Upload research metadata (CSV)", type=["csv"], key="metadata")
        # School coordinates (optional)
        coord_file = st.file_uploader("School Coordinates CSV (optional)", type=["csv"], key="coordinates")
        st.markdown("---")
        st.markdown("**Need templates?**")
        survey_template = ("month,school_id_no,school_name,R,A,C,S,I,P,M\n"
                           "2026-01,1,School_1,0.32,0.41,0.28,0.15,0.14,0.19,0.08")
        metadata_template = ("upload_date,teacher_name,school_id_no,document_type,title,theme,"
                             "status,publication_link,utilized_by_school,utilization_date,"
                             "year_undertaken,years_of_service,teacher_rank,educational_attainment\n"
                             "2026-03-15,Anna Reyes,1,abstract,Improving Reading,Teaching Strategies,"
                             "published,https://doi.org/10.1234,True,2026-02-10,2025,10,Teacher II,Master's")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Survey Template", survey_template, "quarterly_survey_template.csv", "text/csv")
        with c2:
            st.download_button("Metadata Template", metadata_template, "research_metadata_template.csv", "text/csv")

    # --- Main logic ---
    if survey_file is not None and metadata_file is not None:
        survey_df_raw = pd.read_csv(survey_file)
        metadata_df_raw = pd.read_csv(metadata_file)
        survey_df, school_info, survey_error = process_survey(survey_df_raw)
        metadata_df, meta_error = process_metadata(metadata_df_raw)

        # --- AUTO-LOAD COORDINATES (with feedback) ---
        coord_df = None
        # Try common paths
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'school_coordinates.csv'),  # parent of cdr_twin
            os.path.join(os.path.dirname(__file__), 'school_coordinates.csv'),                  # same as ui.py
            os.path.join(os.getcwd(), 'school_coordinates.csv'),                               # current working directory
        ]
        for path in search_paths:
            if os.path.isfile(path):
                coord_df_raw = pd.read_csv(path)
                coord_df_raw.columns = [c.strip().lower() for c in coord_df_raw.columns]
                if {'school_id_no', 'latitude', 'longitude'}.issubset(coord_df_raw.columns):
                    coord_df = coord_df_raw[['school_id_no', 'latitude', 'longitude']].dropna()
                    st.success(f"✅ Loaded coordinates automatically from {path}")
                    break
                else:
                    st.warning(f"Coordinates file found at {path} but columns are incorrect. Using manual upload if provided.")

        # Fallback to manual upload
        if coord_df is None and coord_file is not None:
            coord_df_raw = pd.read_csv(coord_file)
            coord_df_raw.columns = [c.strip().lower() for c in coord_df_raw.columns]
            if {'school_id_no', 'latitude', 'longitude'}.issubset(coord_df_raw.columns):
                coord_df = coord_df_raw[['school_id_no', 'latitude', 'longitude']].dropna()
            else:
                st.error(f"Coordinates file must contain columns: school_id_no, latitude, longitude. Found: {', '.join(coord_df_raw.columns.tolist())}")

        if survey_error:
            st.error(f"Survey error: {survey_error}")
        elif meta_error:
            st.error(f"Metadata error: {meta_error}")
        else:
            # (rest of the main processing – identical to previous full version, no changes needed)
            # I will not repeat the entire block due to length; it's exactly the same as before,
            # starting from "actual_count = len(school_info)" to the end.
            # Please keep the existing code from that point onward.
            # The map expander uses coord_df as before.
