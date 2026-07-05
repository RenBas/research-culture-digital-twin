# ============================================================
# ui.py – main Streamlit user interface (with school map)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import time
import os
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
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'school_coordinates.csv'),  # parent of cdr_twin (project root)
            os.path.join(os.path.dirname(__file__), 'school_coordinates.csv'),                  # same as ui.py (cdr_twin/)
            os.path.join(os.getcwd(), 'school_coordinates.csv'),                               # current working directory
        ]
        for path in search_paths:
            if os.path.isfile(path):
                coord_df_raw = pd.read_csv(path)
                coord_df_raw.columns = [c.strip().lower() for c in coord_df_raw.columns]
                if {'school_name', 'latitude', 'longitude'}.issubset(coord_df_raw.columns):
                    coord_df = coord_df_raw[['school_name', 'latitude', 'longitude']].dropna()
                    st.success(f"✅ Loaded coordinates automatically from {path}")
                    break
                else:
                    st.warning(f"Coordinates file found at {path} but columns are incorrect. Using manual upload if provided.")

        # Fallback to manual upload
        if coord_df is None and coord_file is not None:
            coord_df_raw = pd.read_csv(coord_file)
            coord_df_raw.columns = [c.strip().lower() for c in coord_df_raw.columns]
            if {'school_name', 'latitude', 'longitude'}.issubset(coord_df_raw.columns):
                coord_df = coord_df_raw[['school_name', 'latitude', 'longitude']].dropna()
            else:
                st.error(f"Coordinates file must contain columns: school_name, latitude, longitude. Found: {', '.join(coord_df_raw.columns.tolist())}")

        if survey_error:
            st.error(f"Survey error: {survey_error}")
        elif meta_error:
            st.error(f"Metadata error: {meta_error}")
        else:
            actual_count = len(school_info)
            total_teachers = metadata_df['teacher_name'].nunique()
            state_changed = False
            if st.session_state.num_schools != actual_count:
                st.session_state.num_schools = actual_count; state_changed = True
            if st.session_state.total_teachers != total_teachers:
                st.session_state.total_teachers = total_teachers; state_changed = True
            if state_changed:
                st.rerun()

            st.success(f"Loaded {actual_count} schools and {total_teachers} teachers.")

            school_ids = school_info['school_id_no'].tolist()
            id_to_label = {sid: f"ID {sid}: {school_info[school_info['school_id_no']==sid]['school_name'].values[0]}" for sid in school_ids}
            label_to_id = {v: k for k, v in id_to_label.items()}
            selected_school_label = st.selectbox("Select school", list(id_to_label.values()), index=0)
            selected_school_id = label_to_id[selected_school_label]
            selected_school_name = id_to_label[selected_school_id].split(": ", 1)[1]

            # Role filtering
            if user_role == "Principal":
                show_div_data = False
            else:
                show_div_data = True

            # --- Baseline ---
            st.markdown("<h2 style='text-align: center;'>Baseline from Uploaded Data</h2>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### Research Outputs (Recent)")
            df_show = metadata_df[metadata_df['school_id_no'] == selected_school_id].sort_values('upload_date', ascending=False)
            if not df_show.empty:
                st.dataframe(df_show[['teacher_name', 'year_undertaken', 'title', 'theme', 'status', 'utilized_by_school']].head(10))
            else:
                st.info("No research outputs for this school.")

            # Compute research metrics once for reuse
            metrics = _compute_research_metrics(metadata_df, selected_school_id)

            latest = get_latest_survey(survey_df, selected_school_id)
            if latest is not None:
                st.markdown("### Current Research Culture Profile (7 Variables)")
                cols = st.columns(4)
                var_cols = ['R', 'A', 'C', 'S', 'I', 'P', 'M']
                for i, var in enumerate(var_cols):
                    if i < 4:
                        with cols[i]:
                            val = latest[var]
                            thresh = 0.8 if var == 'A' else None
                            fig = create_gauge(val, VAR_FULL_NAMES[var], threshold=thresh, dark_mode=dark_mode)
                            display_gauge_with_interpretation(fig, val, VAR_INTERPRETATION[var], dark_mode)
                    else:
                        if i == 4:
                            cols2 = st.columns(3)
                        idx2 = i - 4
                        with cols2[idx2]:
                            val = latest[var]
                            thresh = 0.8 if var == 'A' else None
                            fig = create_gauge(val, VAR_FULL_NAMES[var], threshold=thresh, dark_mode=dark_mode)
                            display_gauge_with_interpretation(fig, val, VAR_INTERPRETATION[var], dark_mode)

                # RCSI comparison
                st.markdown("### School & Division RCSI Comparison")
                rcsi_school = np.mean([latest[v] for v in VARIABLES])
                rcsi_division = np.mean([np.mean([survey_df[survey_df['school_id_no']==sid][v].iloc[-1] if not survey_df[survey_df['school_id_no']==sid].empty else 0.0 for v in VARIABLES]) for sid in school_ids]) if show_div_data else rcsi_school

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    fig1 = create_rcsi_gauge(rcsi_school, f"{selected_school_name}\nRCSI (Average)", dark_mode)
                    st.plotly_chart(fig1, use_container_width=True)
                    st.markdown(f"<div style='text-align: center;'><b>Level: {classify_rcsi(rcsi_school)}</b></div>", unsafe_allow_html=True)
                with col_r2:
                    if show_div_data:
                        fig2 = create_rcsi_gauge(rcsi_division, "Division Average RCSI", dark_mode)
                        st.plotly_chart(fig2, use_container_width=True)
                        st.markdown(f"<div style='text-align: center;'><b>Level: {classify_rcsi(rcsi_division)}</b></div>", unsafe_allow_html=True)
                    else:
                        st.info("Switch to Division Head role to see division average.")

                # Utilisation
                st.markdown("### Research Utilisation Rate")
                school_meta = metadata_df[metadata_df['school_id_no'] == selected_school_id]
                school_util = (school_meta['utilized_by_school'].sum() / len(school_meta) * 100) if len(school_meta) > 0 else 0.0
                div_util = (metadata_df['utilized_by_school'].sum() / len(metadata_df) * 100) if len(metadata_df) > 0 else 0.0

                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    fig_u1 = create_utilisation_gauge(school_util, f"{selected_school_name}\nUtilisation", dark_mode)
                    st.plotly_chart(fig_u1, use_container_width=True)
                    st.markdown(f"<div style='text-align: center;'><b>{classify_utilisation(school_util)}</b></div>", unsafe_allow_html=True)
                with col_u2:
                    if show_div_data:
                        fig_u2 = create_utilisation_gauge(div_util, "Division Utilisation", dark_mode)
                        st.plotly_chart(fig_u2, use_container_width=True)
                        st.markdown(f"<div style='text-align: center;'><b>{classify_utilisation(div_util)}</b></div>", unsafe_allow_html=True)
                    else:
                        st.info("Switch to Division Head role to see division utilisation.")

                # Baseline synopsis (school)
                if 'baseline_synopsis' not in st.session_state:
                    with st.spinner("Generating baseline synopsis..."):
                        time.sleep(0.5)
                        st.session_state.baseline_synopsis = generate_baseline_synopsis(latest, selected_school_name, metadata_df)
                        st.session_state.baseline_survey_row = latest.to_dict()
                        school_survey = survey_df[survey_df['school_id_no'] == selected_school_id]
                        st.session_state.baseline_std_devs = {v: school_survey[v].std() if len(school_survey) > 1 else 0.1 for v in VARIABLES}
                        st.rerun()

                if 'baseline_synopsis' in st.session_state:
                    bs = st.session_state.baseline_synopsis
                    bg_color = '#2E2E2E' if dark_mode else '#E3F2FD'
                    text_col = 'white' if dark_mode else 'inherit'

                    # ---- New: Demographics insight for synopsis ----
                    demo_insight = ""
                    if metrics:
                        # Rank
                        if metrics.get('rank_summary') is not None:
                            rank = metrics['rank_summary']
                            top_rank = rank.loc[rank['total_outputs'].idxmax()]
                            demo_insight += f" The most productive teacher rank is **{top_rank['teacher_rank']}** with {top_rank['total_outputs']} outputs (avg {top_rank['avg_outputs']:.1f} per teacher)."
                        # Education
                        if metrics.get('edu_summary') is not None:
                            edu = metrics['edu_summary']
                            top_edu = edu.loc[edu['total_outputs'].idxmax()]
                            demo_insight += f" Teachers with **{top_edu['educational_attainment']}** produce the most research ({top_edu['total_outputs']} outputs, avg {top_edu['avg_outputs']:.1f})."
                        # Service
                        sd = metrics.get('service_data')
                        if sd is not None:
                            demo_insight += f" Average years of service is {sd['avg_service']:.1f}, with {sd['avg_output']:.1f} outputs per teacher. "
                            if sd['slope'] > 0:
                                demo_insight += "Research output tends to increase with years of experience."
                            else:
                                demo_insight += "Research output does not increase with years of service, suggesting targeted support for mid‑career teachers."
                    if demo_insight:
                        demo_insight = f"<b>Teacher Demographics:</b> {demo_insight}"

                    st.markdown("### Baseline Synopsis (School)")
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                    <b>School: {selected_school_name}</b><br>
                    Baseline RCSI: {bs['baseline_rcsi']:.3f}<br>
                    Strengths (≥0.6): {', '.join(bs['strengths']) if bs['strengths'] else 'None'}<br>
                    Critical Gaps (≤0.3): {', '.join(bs['gaps']) if bs['gaps'] else 'None'}<br>
                    Moderate (0.3–0.6): {', '.join(bs['moderate']) if bs['moderate'] else 'None'}<br>
                    Actionable Recommendations:<br>{'<br>'.join(bs['recommendations'])}<br><br>
                    {demo_insight}
                    </div>
                    """, unsafe_allow_html=True)

                # ---- NEW: Teacher Demographics vs Outputs (baseline) ----
                st.markdown("### 👥 Teacher Demographics vs. Research Outputs")
                st.caption("These charts show how teacher characteristics relate to the number of research outputs produced (based on uploaded metadata only).")
                school_meta_demo = metadata_df[metadata_df['school_id_no'] == selected_school_id]
                if not school_meta_demo.empty:
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        if 'teacher_rank' in school_meta_demo.columns and not school_meta_demo['teacher_rank'].isna().all():
                            rank_counts = school_meta_demo.groupby('teacher_rank').size().reset_index(name='Total Outputs')
                            fig_r = px.bar(rank_counts, x='teacher_rank', y='Total Outputs',
                                           title="By Teacher Rank",
                                           color='teacher_rank', color_discrete_sequence=[USTP_DARK_BLUE, USTP_GOLD, DEPED_RED])
                            fig_r.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', showlegend=False)
                            st.plotly_chart(fig_r, use_container_width=True)
                        else:
                            st.info("No teacher rank data available.")
                    with col_d2:
                        if 'educational_attainment' in school_meta_demo.columns and not school_meta_demo['educational_attainment'].isna().all():
                            edu_counts = school_meta_demo.groupby('educational_attainment').size().reset_index(name='Total Outputs')
                            fig_e = px.bar(edu_counts, x='educational_attainment', y='Total Outputs',
                                           title="By Educational Attainment",
                                           color='educational_attainment',
                                           color_discrete_sequence=[DEPED_RED, USTP_GOLD, USTP_DARK_BLUE])
                            fig_e.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', showlegend=False)
                            st.plotly_chart(fig_e, use_container_width=True)
                        else:
                            st.info("No educational attainment data available.")
                    with col_d3:
                        if 'years_of_service' in school_meta_demo.columns and not school_meta_demo['years_of_service'].isna().all():
                            def service_bracket(y):
                                if y <= 5: return "0-5"
                                elif y <= 10: return "6-10"
                                elif y <= 15: return "11-15"
                                elif y <= 20: return "16-20"
                                else: return "20+"
                            school_meta_demo['service_bracket'] = school_meta_demo['years_of_service'].apply(service_bracket)
                            sv_counts = school_meta_demo.groupby('service_bracket').size().reset_index(name='Total Outputs')
                            order = ["0-5", "6-10", "11-15", "16-20", "20+"]
                            sv_counts['service_bracket'] = pd.Categorical(sv_counts['service_bracket'], categories=order, ordered=True)
                            sv_counts = sv_counts.sort_values('service_bracket')
                            fig_s = px.bar(sv_counts, x='service_bracket', y='Total Outputs',
                                           title="By Years of Service",
                                           color='service_bracket',
                                           color_discrete_sequence=[USTP_GOLD, DEPED_RED, USTP_DARK_BLUE, '#8E44AD', '#2ECC71'])
                            fig_s.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', showlegend=False)
                            st.plotly_chart(fig_s, use_container_width=True)
                        else:
                            st.info("No years of service data available.")
                    st.markdown("""
                    **Interpretation:**
                    - **Teacher Rank**: Highlights which rank (e.g., Teacher I, II, III) contributes the most research outputs. Higher ranks often correlate with greater research activity.
                    - **Educational Attainment**: Shows how teachers with Master's or Doctoral degrees compare to those with only Bachelor's degrees in terms of research productivity.
                    - **Years of Service**: Illustrates whether experienced teachers (20+ years) or early‑career teachers produce more research, helping to target mentoring efforts.
                    """)
                else:
                    st.info("No metadata available for teacher demographics.")

                # Division Baseline Synopsis (if Division Head)
                if show_div_data:
                    if 'division_baseline_synopsis' not in st.session_state:
                        with st.spinner("Generating division baseline synopsis..."):
                            time.sleep(0.5)
                            div_syn = generate_division_baseline_synopsis(survey_df, metadata_df, school_info)
                            if div_syn:
                                st.session_state.division_baseline_synopsis = div_syn
                            else:
                                st.session_state.division_baseline_synopsis = None
                            st.rerun()

                    if 'division_baseline_synopsis' in st.session_state and st.session_state.division_baseline_synopsis is not None:
                        ds = st.session_state.division_baseline_synopsis
                        bg_color = '#2E2E2E' if dark_mode else '#E8F5E9'
                        text_col = 'white' if dark_mode else 'inherit'
                        st.markdown("### Baseline Synopsis (Division)")
                        early_pct = ds['stage_distribution']['early_pct']
                        adv_pct = ds['stage_distribution']['adv_pct']
                        trans_pct = ds['stage_distribution']['trans_pct']
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                        <b>Division‑Level Baseline (all {ds['total_schools']} schools)</b><br>
                        - Average RCSI: <b>{ds['avg_rcsi']:.3f}</b><br>
                        - Average Milestone: <b>{ds['avg_milestone']:.1f}</b><br>
                        - Research Utilisation Rate: <b>{ds['div_util_rate']:.1f}%</b><br>
                        - Strengths (≥0.6): {', '.join(ds['strengths']) if ds['strengths'] else 'None'}<br>
                        - Critical Gaps (≤0.3): {', '.join(ds['gaps']) if ds['gaps'] else 'None'}<br>
                        - Moderate (0.3–0.6): {', '.join(ds['moderate']) if ds['moderate'] else 'None'}<br>
                        - Stage distribution: {early_pct:.1f}% early (M≤2), {trans_pct:.1f}% transitional (M3), {adv_pct:.1f}% advanced (M≥4).<br>
                        Actionable Recommendations:<br>{'<br>'.join(ds['recommendations'])}
                        </div>
                        """, unsafe_allow_html=True)

            else:
                st.info("No survey data for current quarter.")

            # Heatmap
            baseline_heatmap(survey_df, metadata_df, dark_mode)

            # Calibration
            if 'calibrated_coeff' not in st.session_state:
                with st.spinner("Calibrating model coefficients from data..."):
                    time.sleep(0.5)
                    coeff, calib_msg = calibrate_coefficients(survey_df)
                    if coeff is not None:
                        st.success(calib_msg)
                        st.session_state.calibrated_coeff = coeff
                        st.session_state.calibration_status = "custom"
                    else:
                        st.info(calib_msg)
                        st.session_state.calibrated_coeff = None
                        st.session_state.calibration_status = "default"
                    st.rerun()

            if 'calibration_status' in st.session_state:
                if st.session_state.calibration_status == "custom":
                    st.caption("✅ Using data‑calibrated coefficients.")
                else:
                    st.caption("ℹ️ Using default coefficients (insufficient data for calibration).")

            agent_params = get_agent_params(school_ids, survey_df, metadata_df, st.session_state.calibrated_coeff)

            # Simulation init
            if 'sim' not in st.session_state:
                st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events, agent_params)
                st.session_state.current_month = 0
                st.session_state.total_months = 0
                st.session_state.history = create_empty_history(school_ids)

            # Run / Step / Reset
            if run_btn:
                st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events, agent_params)
                st.session_state.current_month = 0; st.session_state.total_months = 0
                st.session_state.history = create_empty_history(school_ids)
                progress = st.progress(0, text="Running simulation...")
                for m in range(1, duration + 1):
                    if use_survey:
                        apply_survey_override(st.session_state.sim.agents, survey_df, m)
                    st.session_state.sim.step(levers, m)
                    st.session_state.current_month = m; st.session_state.total_months = m
                    record_history(st.session_state.history, st.session_state.sim.agents, m)
                    progress.progress(m / duration, text=f"Month {m}/{duration}")
                progress.empty()
                if mc_enabled:
                    with st.spinner(f"Running {mc_runs} Monte Carlo simulations..."):
                        mc_data, mc_info = monte_carlo_sim(mc_runs, Simulation, agent_params, school_ids, levers, duration,
                                                           use_survey, survey_df, metadata_df, selected_school_id)
                        st.session_state.mc_data = mc_data
                        st.session_state.mc_info = mc_info
                st.rerun()

            if step_btn:
                target = st.session_state.current_month + 1
                if use_survey:
                    apply_survey_override(st.session_state.sim.agents, survey_df, target)
                st.session_state.sim.step(levers, target)
                st.session_state.current_month = target; st.session_state.total_months = target
                record_history(st.session_state.history, st.session_state.sim.agents, target)
                st.rerun()

            if reset_btn:
                st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events, agent_params)
                st.session_state.current_month = 0; st.session_state.total_months = 0
                st.session_state.history = create_empty_history(school_ids)
                st.rerun()

            # Simulation results
            if st.session_state.total_months > 0:
                text_col = 'white' if dark_mode else 'inherit'
                st.markdown("<h2 style='text-align: center;'>Simulated Data</h2>", unsafe_allow_html=True)
                st.markdown("---")
                hist = st.session_state.history.get(selected_school_id)
                agent = next((a for a in st.session_state.sim.agents if a.real_id == selected_school_id), None)
                if hist and agent:
                    # Main charts
                    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Variable Evolution", "Milestone Progress",
                                                                         "Research Culture Sustainability Index (RCSI)",
                                                                         "Improvement per Completed Cycle"))
                    for i, var in enumerate(VARIABLES):
                        fig1.add_trace(go.Scatter(x=hist['month'], y=hist[var], mode='lines', name=var,
                                                  line=dict(color=VAR_COLORS[i])), row=1, col=1)
                    fig1.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'], mode='lines', name='Milestone',
                                              line=dict(color=DEPED_RED, width=3)), row=1, col=2)
                    fig1.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'], mode='lines', name='RCSI',
                                              line=dict(color=USTP_GOLD, width=3)), row=2, col=1)
                    if agent.cycle_improvements:
                        cycles = [c.cycle_number for c in agent.cycle_improvements]
                        improvements = [c.total_improvement for c in agent.cycle_improvements]
                        fig1.add_trace(go.Bar(x=cycles, y=improvements, name='RCSI per cycle',
                                              marker_color=USTP_DARK_BLUE), row=2, col=2)
                    else:
                        fig1.add_annotation(text="No cycles completed yet", xref="x2 domain", yref="y2 domain",
                                            x=0.5, y=0.5, showarrow=False, row=2, col=2)
                    template = 'plotly_dark' if dark_mode else 'plotly_white'
                    fig1.update_layout(height=800, showlegend=True, font=dict(color=text_col), template=template,
                                       xaxis_title="Month", yaxis_title="Value",
                                       xaxis2_title="Month", yaxis2_title="Milestone",
                                       xaxis3_title="Month", yaxis3_title="RCSI",
                                       xaxis4_title="Cycle Number", yaxis4_title="RCSI per Cycle")
                    st.plotly_chart(fig1, use_container_width=True)
                    get_figure_download_link(fig1, "simulation_overview.html", "Download Simulation Charts")

                    # Simulated gauges
                    st.markdown("### Simulated Current State (Final Month)")
                    final_vals = [getattr(agent, v) for v in VARIABLES]
                    cols_g = st.columns(4)
                    for i, var in enumerate(VARIABLES):
                        if i < 4:
                            with cols_g[i]:
                                val = final_vals[i]
                                thresh = 0.8 if var == 'A' else None
                                fig = create_gauge(val, VAR_FULL_NAMES[var], threshold=thresh, dark_mode=dark_mode)
                                display_gauge_with_interpretation(fig, val, VAR_INTERPRETATION[var], dark_mode)
                        else:
                            if i == 4:
                                cols_g2 = st.columns(3)
                            idx2 = i - 4
                            with cols_g2[idx2]:
                                val = final_vals[i]
                                thresh = 0.8 if var == 'A' else None
                                fig = create_gauge(val, VAR_FULL_NAMES[var], threshold=thresh, dark_mode=dark_mode)
                                display_gauge_with_interpretation(fig, val, VAR_INTERPRETATION[var], dark_mode)

                    # Final RCSI
                    final_rcsi = agent.running_total_outcome
                    st.markdown("### Final RCSI")
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        fig_rcsi_final = create_rcsi_gauge(final_rcsi, f"{selected_school_name}\nFinal RCSI (Average)", dark_mode)
                        st.plotly_chart(fig_rcsi_final, use_container_width=True)
                        st.markdown(f"<div style='text-align: center;'><b>Level: {classify_rcsi(final_rcsi)}</b></div>", unsafe_allow_html=True)
                    with col_f2:
                        if show_div_data:
                            div_rcsi_final = np.mean([a.running_total_outcome for a in st.session_state.sim.agents])
                            fig_div_rcsi = create_rcsi_gauge(div_rcsi_final, "Division Avg Final RCSI", dark_mode)
                            st.plotly_chart(fig_div_rcsi, use_container_width=True)
                            st.markdown(f"<div style='text-align: center;'><b>Level: {classify_rcsi(div_rcsi_final)}</b></div>", unsafe_allow_html=True)
                        else:
                            st.info("Switch to Division Head role for division average.")

                    # Expanders
                    with st.expander("Cycle vs Research Outputs"):
                        cycle_research_correlation(agent, metadata_df, selected_school_id, dark_mode)

                    with st.expander("Division-Level Analysis"):
                        div_metrics = division_level_analysis(metadata_df, st.session_state.history,
                                                              st.session_state.sim.agents, dark_mode)

                    # Comparative
                    with st.expander("Comparative School Analysis (Gauges)"):
                        comp_options = [sid for sid in school_ids if sid != selected_school_id]
                        comp_options = [selected_school_id] + comp_options
                        selected_comparison = st.multiselect(
                            "Select 2-3 schools to compare",
                            options=comp_options,
                            format_func=lambda x: id_to_label[x],
                            default=comp_options[:2] if len(comp_options) >= 2 else comp_options
                        )
                        if len(selected_comparison) >= 2:
                            school_comparison_gauge(selected_comparison, school_info, st.session_state.history,
                                                    dark_mode)
                        else:
                            st.info("Select at least 2 schools for comparison.")

                    # ---------------------------
                    # Division School Map (only for Division Head)
                    # ---------------------------
                    if show_div_data and coord_df is not None:
                        with st.expander("📍 Division School Map"):
                            # Build name -> coordinate lookup
                            name_to_coord = {}
                            for _, row in coord_df.iterrows():
                                name_to_coord[str(row['school_name']).strip()] = (row['latitude'], row['longitude'])

                            lats = [v[0] for v in name_to_coord.values()]
                            lons = [v[1] for v in name_to_coord.values()]
                            map_center = [np.mean(lats), np.mean(lons)] if lats else [8.48, 124.65]
                            school_map = folium.Map(location=map_center, zoom_start=12)

                            sim_state = {}
                            sim_names = []
                            for ag in st.session_state.sim.agents:
                                sname = school_info[school_info['school_id_no'] == ag.real_id]['school_name'].values[0] if not school_info[school_info['school_id_no'] == ag.real_id].empty else None
                                if sname:
                                    sim_state[sname.strip()] = {
                                        'RCSI': ag.running_total_outcome,
                                        'milestone': ag.current_milestone,
                                        'school_name': sname
                                    }
                                    sim_names.append(sname.strip())

                            milestone_colors = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green',
                                               4: 'lightblue', 5: 'blue', 6: 'purple'}

                            matched = 0
                            for school_name, (lat, lon) in name_to_coord.items():
                                if school_name in sim_state:
                                    state = sim_state[school_name]
                                    color = milestone_colors.get(state['milestone'], 'gray')
                                    popup_text = f"""
                                    <b>{state['school_name']}</b><br>
                                    RCSI: {state['RCSI']:.3f}<br>
                                    Milestone: M{state['milestone']}<br>
                                    """
                                    folium.Marker(
                                        location=[lat, lon],
                                        popup=folium.Popup(popup_text, max_width=250),
                                        icon=folium.Icon(color=color, icon='info-sign')
                                    ).add_to(school_map)
                                    matched += 1

                            st_folium(school_map, width=700, height=500)
                            st.caption("Marker colors: Red=M0, Orange=M1, Yellow=M2, Green=M3, "
                                       "Light blue=M4, Blue=M5, Purple=M6")
                            st.write(f"**Schools displayed on map:** {matched} out of {len(name_to_coord)} coordinate entries.")
                            if matched == 0:
                                st.warning(
                                    "No schools could be matched. The school names in the coordinates file "
                                    "do not match any school name in the simulation data. "
                                    f"\n\nFirst 5 coordinate names: {list(name_to_coord.keys())[:5]}"
                                    f"\nFirst 5 simulation names: {sim_names[:5]}"
                                )

                    # Sensitivity
                    sensitivity_info = ""
                    if not st.session_state.get('sensitivity_fig'):
                        with st.spinner("Computing sensitivity analysis..."):
                            fig_tornado, sensitivity_info = run_sensitivity(Simulation, agent_params, school_ids, levers,
                                                                            duration, use_survey, survey_df, metadata_df,
                                                                            selected_school_id)
                            st.session_state.sensitivity_fig = fig_tornado
                            st.session_state.sensitivity_info = sensitivity_info
                    else:
                        fig_tornado = st.session_state.sensitivity_fig
                        sensitivity_info = st.session_state.sensitivity_info

                    with st.expander("Sensitivity Analysis (Tornado)"):
                        st.plotly_chart(fig_tornado, use_container_width=True)
                        st.caption("Each lever varied ±10% while others fixed at current slider values.")
                        if sensitivity_info:
                            st.markdown(sensitivity_info)

                    # Monte Carlo
                    if 'mc_data' in st.session_state:
                        with st.expander("Monte Carlo Uncertainty Bands"):
                            mc_data = st.session_state.mc_data
                            fig_mc = plot_monte_carlo_bands(mc_data, dark_mode)
                            st.plotly_chart(fig_mc, use_container_width=True)
                            st.caption(f"Shaded area: P10‑P90 range over {mc_runs} simulations.")
                            if st.session_state.get('mc_info'):
                                st.markdown(st.session_state.mc_info)
                                with st.expander("RCSI Interpretation Guide"):
                                    st.markdown(get_rcsi_interpretation_table())
                                st.markdown("""
                                **Graph Interpretation:**  
                                At the school level, the simulated RCSI’s tight interquartile range confirms that internal processes are fully insulated from external shocks, validating the current local administration.  
                                Divergently, at the division level, this uniformity signals a systemic plateau—prompting division leaders to shift focus from risk mitigation to pedagogical innovation, as quantitative variance no longer provides actionable leverage for district‑wide improvement.
                                """)
                            if 'baseline_synopsis' in st.session_state:
                                baseline_vals = st.session_state.baseline_synopsis['values']
                                causal_coeffs = causal_analysis(mc_data['final_rcsi'], baseline_vals)
                                if causal_coeffs:
                                    st.markdown("**Causal Impact (increase final RCSI per unit increase in baseline variable):**")
                                    df_causal = pd.DataFrame(list(causal_coeffs.items()), columns=['Variable', 'Impact'])
                                    st.dataframe(df_causal)
                                else:
                                    st.info("Not enough Monte Carlo runs for causal analysis (need >10).")

                    # School synopsis (simulation)
                    rcsi_val = agent.running_total_outcome
                    rcsi_level = classify_rcsi(rcsi_val)
                    milestone_name = MILESTONE_NAMES.get(agent.current_milestone, f"Milestone {agent.current_milestone}")
                    if agent.cycle_count >= 2:
                        cycle_text = f"has completed {agent.cycle_count} full cycles, indicating a self-sustaining research culture."
                    elif agent.cycle_count == 1:
                        cycle_text = "has completed one full cycle, demonstrating initial sustainability."
                    else:
                        cycle_text = "has not yet completed any full cycle."
                    if agent.current_milestone == 0:
                        milestone_progress = "is at the very beginning of the journey."
                    elif agent.current_milestone <= 2:
                        milestone_progress = "has moved beyond initial readiness but remains in early capacity‑building phases."
                    elif agent.current_milestone <= 4:
                        milestone_progress = "has established structured support and is embedding research into institutional practice."
                    else:
                        milestone_progress = "is realising tangible impact and is approaching or has achieved cyclical sustainability."

                    sens_text = sensitivity_info if sensitivity_info else ""
                    mc_text = st.session_state.get('mc_info', "")
                    avg_rcsi_division = np.mean([a.running_total_outcome for a in st.session_state.sim.agents])
                    gap = rcsi_val - avg_rcsi_division
                    gap_text = f"Compared to the division average of **{avg_rcsi_division:.3f}**, this school is **{gap:+.3f}** points {'above' if gap > 0 else 'below'} the division average."

                    bg_col = '#2E2E2E' if dark_mode else '#E3F2FD'
                    synopsis = f"""
                    After {st.session_state.total_months} months, {selected_school_name} (ID {selected_school_id}) has reached {milestone_name} and {cycle_text}
                    The school's Research Culture Sustainability Index (RCSI) is <b>{rcsi_val:.3f}</b>, which falls into the <b>{rcsi_level}</b> level.
                    {sens_text}
                    {mc_text}
                    {gap_text}
                    Overall, the school is on a path toward research culture sustainability, but further policy support may be needed.
                    """
                    st.markdown(f"""
                    <div style="background-color: {bg_col}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                    <b>School {selected_school_id} ({selected_school_name}) – Simulation Synopsis</b><br>
                    {synopsis}
                    </div>
                    """, unsafe_allow_html=True)

                    # Baseline vs Sim comparison
                    if ('baseline_synopsis' in st.session_state and 'baseline_survey_row' in st.session_state):
                        bs = st.session_state.baseline_synopsis
                        baseline_vals = st.session_state.baseline_survey_row
                        gaps = bs['gaps']
                        if gaps:
                            st.markdown("#### Baseline vs Simulation Comparison (Critical Gaps)")
                            table_data = []
                            baseline_std_devs = st.session_state.get('baseline_std_devs', {})
                            for var in gaps:
                                base_val = baseline_vals[var]
                                sim_val = getattr(agent, var)
                                diff = sim_val - base_val
                                status = "Improving" if diff > 0.01 else ("Regressing" if diff < -0.01 else "Stable")
                                std_dev = baseline_std_devs.get(var, 0.1)
                                if abs(diff) >= 0.10:
                                    significance = "Both statistically and practically significant"
                                elif abs(diff) >= 0.5 * std_dev:
                                    significance = "Statistically significant, but limited practical impact"
                                else:
                                    significance = "Not significant (within normal variability)"
                                table_data.append({
                                    "Critical Gap": VAR_FULL_NAMES[var],
                                    "Baseline": f"{base_val:.2f}",
                                    "Simulation": f"{sim_val:.2f}",
                                    "Status": status,
                                    "Significance": significance
                                })
                            st.table(pd.DataFrame(table_data))

                    # Division synopsis (simulation) – only if show_div_data
                    if show_div_data:
                        total_schools = len(st.session_state.sim.agents)
                        early_stage = sum(1 for a in st.session_state.sim.agents if a.current_milestone <= 2)
                        advanced_stage = sum(1 for a in st.session_state.sim.agents if a.current_milestone >= 4)
                        transitional = total_schools - early_stage - advanced_stage
                        early_percent = (early_stage / total_schools * 100) if total_schools > 0 else 0
                        advanced_percent = (advanced_stage / total_schools * 100) if total_schools > 0 else 0
                        transitional_percent = (transitional / total_schools * 100) if total_schools > 0 else 0
                        early_text = f"{early_percent:.1f}% of schools" if early_percent > 0 else "No schools"
                        advanced_text = f"{advanced_percent:.1f}% of schools" if advanced_percent > 0 else "No schools"
                        if early_percent == 100:
                            sustainability_text = "All schools are in early milestones; foundational capacity‑building is the priority."
                        elif early_percent >= 75:
                            sustainability_text = f"The vast majority ({early_percent:.1f}%) are in early milestones; urgent interventions needed."
                        elif early_percent >= 50:
                            sustainability_text = f"More than half ({early_percent:.1f}%) are in early milestones; targeted policy support may accelerate progress."
                        elif early_percent > 0:
                            sustainability_text = f"{early_percent:.1f}% remain in early milestones; continued efforts are required."
                        else:
                            sustainability_text = "No schools are in early milestones; the division exhibits a strong, advanced research culture."

                        total_outcome = sum(a.running_total_outcome for a in st.session_state.sim.agents)
                        avg_rcsi_div = total_outcome / total_schools if total_schools > 0 else 0
                        level_avg = classify_rcsi(avg_rcsi_div)
                        total_cycles = sum(a.cycle_count for a in st.session_state.sim.agents)
                        avg_milestone = np.mean([a.current_milestone for a in st.session_state.sim.agents])
                        avg_milestone_interp = interpret_avg_milestone(avg_milestone)

                        school_ids_in_sim = [a.real_id for a in st.session_state.sim.agents]
                        div_metadata = metadata_df[metadata_df['school_id_no'].isin(school_ids_in_sim)]
                        total_utilised = div_metadata['utilized_by_school'].sum() if 'utilized_by_school' in div_metadata.columns else 0
                        total_research = len(div_metadata)
                        div_util_rate = (total_utilised / total_research * 100) if total_research > 0 else 0

                        top_div_teacher = div_metrics.get('top_div_teacher', 'N/A')
                        top_div_school = div_metrics.get('top_div_school', 'N/A')
                        top_div_outputs = div_metrics.get('top_div_outputs', 0)
                        bottleneck_milestone = div_metrics.get('bottleneck_milestone', 'N/A')
                        bottleneck_time = div_metrics.get('bottleneck_time', 0)

                        output_trend_div = ""
                        if not metadata_df.empty and 'upload_date' in metadata_df.columns:
                            div_timeline = metadata_df.groupby(metadata_df['upload_date'].dt.to_period('Q')).size()
                            if len(div_timeline) >= 2:
                                if div_timeline.iloc[-1] > div_timeline.iloc[-2]:
                                    output_trend_div = "The division's research output is increasing over time."
                                elif div_timeline.iloc[-1] < div_timeline.iloc[-2]:
                                    output_trend_div = "The division's research output is declining over time."
                                else:
                                    output_trend_div = "The division's research output has remained stable."
                                avg_div_output = div_timeline.mean()
                                output_trend_div += f" On average, the division produces {avg_div_output:.1f} outputs per quarter."

                        full_bottleneck = MILESTONE_NAMES.get(
                            int(bottleneck_milestone.replace('M', '')) if isinstance(bottleneck_milestone, str) and bottleneck_milestone.startswith('M') else 0,
                            bottleneck_milestone
                        )
                        bottleneck_insight = (f"Schools spend the most time on average in {full_bottleneck} ({bottleneck_time:.1f} months). This is the critical bottleneck." if bottleneck_milestone != "N/A" else "")
                        top_teacher_insight = (f"The division's top researcher is {top_div_teacher} from {top_div_school} with {top_div_outputs} outputs." if top_div_teacher != "N/A" else "")

                        div_mc_text = ""
                        if 'mc_data' in st.session_state:
                            mc_finals = st.session_state.mc_data['final_rcsi']
                            div_mc_text = (f"Monte Carlo projections suggest that the division's average RCSI is estimated around "
                                           f"**{np.mean(mc_finals):.3f}** with a P10‑P90 range of "
                                           f"**{np.percentile(mc_finals, 10):.3f}** – **{np.percentile(mc_finals, 90):.3f}**, "
                                           f"indicating that the division as a whole exhibits low variance and stable sustainability.")

                        bg_div = '#2E2E2E' if dark_mode else '#E8F5E9'
                        st.markdown(f"""
                        <div style="background-color: {bg_div}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                        <b>Division‑Level Sustainability Synopsis (all {total_schools} schools)</b><br>
                        - Average milestone = {avg_milestone:.1f} → {avg_milestone_interp}<br>
                        - Total completed cycles = {total_cycles}<br>
                        - Average RCSI = <b>{avg_rcsi_div:.3f}</b> → <b>{level_avg}</b> level.<br>
                        - Average research utilisation rate = <b>{div_util_rate:.1f}%</b>.<br>
                        - Stage distribution: {early_text} are in early stages (M≤2), {transitional_percent:.1f}% transitional (M3), and {advanced_text} are advanced (M≥4).<br>
                        <i>Division‑wide sustainability assessment:</i> {sustainability_text}<br><br>
                        <b>Productivity:</b> {output_trend_div}<br>
                        <b>Bottleneck:</b> {bottleneck_insight}<br>
                        <b>Top Division Researcher:</b> {top_teacher_insight}<br>
                        {div_mc_text}
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("Graph Interpretations"):
                        st.markdown("""
                        - **Variable Evolution:** How R, A, C, S, I, P, M change over time. Higher values (closer to 1) mean stronger readiness, awareness, capacity, etc.
                        - **Milestone Progress:** The school moves through milestones 0-6. Reaching milestone 6 and cycling back indicates a full sustainable cycle.
                        - **RCSI:** Cumulative strength of the research ecosystem, derived from Impact Realization (M) and Collaboration (P).
                        - **Improvement per Cycle:** Each bar shows the RCSI contributed by one cycle. Higher bars in later cycles indicate increasing effectiveness.
                        - **Gauges:** Each variable is shown as a speedometer-style gauge with a thick coloured bar (needle) and a red threshold line at the current value. The Awareness gauge also has a red marker at 0.8 (M0→M1).
                        - **Research Outputs Dashboard:** Tracks themes, publication status, utilisation, teacher productivity, experience vs output, timeline, top teachers, and breakdown by rank and attainment.
                        - **Sensitivity Tornado:** Shows which policy lever most influences the final RCSI when varied ±10%.
                        - **Monte Carlo Bands:** Depicts the uncertainty range (P10‑P90) of RCSI and milestone trajectories over multiple simulation runs.
                        - **Division‑Level Analysis:** Milestone transition bottlenecks and teacher leaderboard.
                        - **Comparative Analysis:** Compares up to 3 schools using RCSI gauges.
                        """)

                # Export
                if export_btn:
                    all_data = []
                    for a in st.session_state.sim.agents:
                        h = st.session_state.history[a.real_id]
                        for t in range(len(h['month'])):
                            row = {'school_id': a.real_id, 'month': h['month'][t], 'milestone': h['milestone'][t], 'running_outcome': h['running_outcome'][t]}
                            for v in VARIABLES:
                                row[v] = h[v][t]
                            all_data.append(row)
                    df_hist = pd.DataFrame(all_data)
                    cycle_records = []
                    for a in st.session_state.sim.agents:
                        for rec in a.cycle_improvements:
                            cycle_records.append({'school_id': a.real_id, 'cycle_number': rec.cycle_number,
                                                  'total_improvement': rec.total_improvement, 'completion_month': rec.completion_month})
                    df_cycles = pd.DataFrame(cycle_records)
                    st.download_button("Download simulation history", df_hist.to_csv(index=False).encode('utf-8'), "simulation_history.csv", "text/csv")
                    st.download_button("Download cycle improvements", df_cycles.to_csv(index=False).encode('utf-8'), "cycle_improvements.csv", "text/csv")
    else:
        st.info("Please upload quarterly survey and research metadata CSV files to begin.")
