import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict
import math

# ============================================================
# CDO Division Colour Palette (USTP + DepEd)
# ============================================================
USTP_DARK_BLUE = "#0D2B5E"
USTP_GOLD = "#F5A623"
DEPED_RED = "#D32F2F"
DEPED_MAROON = "#8B0000"
LIGHT_BG = "#F8F9FA"
DARK_BG = "#1E1E1E"
DARK_TEXT = "#FFFFFF"
LIGHT_TEXT = "#000000"

# ------------------------------------------------------------
# Apply Dark Mode CSS if enabled
# ------------------------------------------------------------
def apply_theme(dark_mode):
    if dark_mode:
        st.markdown(f"""
        <style>
            .stApp {{ background-color: {DARK_BG} !important; color: {DARK_TEXT} !important; }}
            .sidebar .sidebar-content {{ background-color: #2E2E2E !important; border-right: 2px solid {USTP_GOLD} !important; }}
            .sidebar .sidebar-content * {{ color: {DARK_TEXT} !important; }}
            h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {USTP_GOLD} !important; }}
            .stMarkdown, .stText, .stCaption, .stDataFrame {{ color: {DARK_TEXT} !important; }}
            .stButton > button {{ background-color: {USTP_DARK_BLUE} !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            .stButton > button:hover {{ background-color: {USTP_GOLD} !important; color: {USTP_DARK_BLUE} !important; }}
            .stMetric {{ background-color: #2E2E2E !important; border: 1px solid {USTP_GOLD} !important; border-radius: 5px; padding: 10px; }}
            .stMetric label {{ color: {DARK_TEXT} !important; }}
            .dataframe {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; }}
            .dataframe thead tr th {{ background-color: {USTP_DARK_BLUE} !important; color: {DARK_TEXT} !important; }}
            .dataframe tbody tr {{ background-color: #2E2E2E !important; }}
            .dataframe tbody tr:hover {{ background-color: #3E3E3E !important; }}
            .streamlit-expanderHeader {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            .streamlit-expanderContent {{ background-color: #1E1E1E !important; color: {DARK_TEXT} !important; }}
            .stAlert {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            .stSelectbox label, .stNumberInput label, .stCheckbox label {{ color: {DARK_TEXT} !important; }}
            .stRadio label {{ color: {DARK_TEXT} !important; }}
            .stFileUploader {{ background-color: #2E2E2E !important; border: 1px dashed {USTP_GOLD} !important; }}
            .stFileUploader label {{ color: {DARK_TEXT} !important; }}
            .stCaption {{ color: #CCCCCC !important; }}
            .main .block-container {{ background-color: {DARK_BG} !important; }}
            .css-1y4p8pa {{ background-color: #2E2E2E !important; }}
            div[style*="background-color: #E3F2FD"] {{ background-color: #2E2E2E !important; border-left: 5px solid {USTP_GOLD} !important; color: {DARK_TEXT} !important; }}
            div[style*="background-color: #E8F5E9"] {{ background-color: #2E2E2E !important; border-left: 5px solid {USTP_GOLD} !important; color: {DARK_TEXT} !important; }}
            table {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            table th {{ background-color: {USTP_DARK_BLUE} !important; color: {DARK_TEXT} !important; }}
            table td {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; }}
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

# ------------------------------------------------------------
# Core simulation engine (unchanged)
# ------------------------------------------------------------
@dataclass
class CycleRecord:
    cycle_number: int
    total_improvement: float
    completion_month: int

class SchoolAgent:
    def __init__(self, unique_id,
                 initial_R=0.3, initial_A=0.2, initial_C=0.2,
                 initial_S=0.1, initial_I=0.1, initial_P=0.1, initial_M=0.0,
                 random_events_enabled=False):
        self.id = unique_id
        self.R = initial_R
        self.A = initial_A
        self.C = initial_C
        self.S = initial_S
        self.I = initial_I
        self.P = initial_P
        self.M = initial_M
        self.current_milestone = 0
        self.months_in_milestone = 0
        self.current_cycle_accumulator = 0.0
        self.cycle_improvements: List[CycleRecord] = []
        self.cycle_count = 0
        self.running_total_outcome = 0.0
        self.min_value = 0.1
        self.random_events_enabled = random_events_enabled
        self.model_time = 0
        self.random = np.random.RandomState()

    def apply_random_event(self):
        if not self.random_events_enabled:
            return
        if self.random.rand() < 0.00417:
            event_type = self.random.choice(["loss_champion", "funding", "leadership_change"])
            if event_type == "loss_champion":
                for var in ['R','A','C','S','I','P','M']:
                    setattr(self, var, max(self.min_value, getattr(self, var) - 0.1))
            elif event_type == "funding":
                self.S = min(1.0, self.S + 0.15)
            elif event_type == "leadership_change":
                self.I = max(self.min_value, self.I - 0.2)

    def step(self, levers: Dict[str, float]):
        u_train, u_mentor, u_budget, u_lead, u_collab = levers.values()
        u_lead_eff = min(1.0, u_lead + 0.05 * self.M)
        self.R += 0.02 * self.M - 0.01 * (1 - u_lead_eff)
        self.A += 0.04 * self.R + 0.02 * u_train + 0.01 * self.M - 0.005
        self.C += 0.03 * u_train + 0.02 * u_mentor - 0.01
        self.S += 0.04 * u_budget + 0.02 * u_mentor - 0.01 * (1 - u_lead_eff)
        self.I += 0.03 * u_lead_eff + 0.02 * self.S - 0.005
        self.P += 0.04 * u_collab + 0.02 * self.I - 0.01
        self.M += 0.02 * self.C + 0.02 * self.P - 0.005
        for var in ['R','A','C','S','I','P','M']:
            setattr(self, var, max(self.min_value, min(1.0, getattr(self, var))))
        monthly_gain = 0.001 * self.M * (1 + self.P)
        self.running_total_outcome += monthly_gain
        self.current_cycle_accumulator += monthly_gain
        self._update_milestone()
        self.apply_random_event()

    def _update_milestone(self):
        self.months_in_milestone += 1
        next_milestone = self.current_milestone
        if self.current_milestone == 0 and self.A >= 0.8:
            next_milestone = 1
        elif self.current_milestone == 1 and self.C >= 0.7:
            next_milestone = 2
        elif self.current_milestone == 2 and self.S >= 0.7:
            next_milestone = 3
        elif self.current_milestone == 3 and self.I >= 0.8:
            next_milestone = 4
        elif self.current_milestone == 4 and self.P >= 0.8:
            next_milestone = 5
        elif self.current_milestone == 5 and self.M >= 0.7:
            next_milestone = 6
        elif self.current_milestone == 6 and self.M >= 0.9 and self.R >= 0.8:
            self._complete_cycle()
            next_milestone = 0
        if next_milestone != self.current_milestone and self.months_in_milestone >= 6:
            self.current_milestone = next_milestone
            self.months_in_milestone = 0

    def _complete_cycle(self):
        old_M = self.M
        self.R = min(1.0, self.R + 0.10)
        self.M = max(0.2, self.M * 0.5)
        bonus = 0.03 + 0.03 * old_M
        self.running_total_outcome += bonus
        self.current_cycle_accumulator += bonus
        self.cycle_count += 1
        self.cycle_improvements.append(CycleRecord(cycle_number=self.cycle_count,
                                                   total_improvement=self.current_cycle_accumulator,
                                                   completion_month=self.model_time))
        self.current_cycle_accumulator = 0.0

class Simulation:
    def __init__(self, num_schools=1, random_events=False):
        self.agents = [SchoolAgent(i, random_events_enabled=random_events) for i in range(num_schools)]
    def step(self, levers, month):
        for agent in self.agents:
            agent.model_time = month
            agent.step(levers)
    def get_agent(self, idx=0):
        return self.agents[idx]

# ------------------------------------------------------------
# Data processing functions
# ------------------------------------------------------------
def process_survey(survey_df):
    if survey_df is None:
        return None, None, "No survey file uploaded."
    try:
        if 'school_id_no' not in survey_df.columns:
            if 'school_id' in survey_df.columns:
                survey_df['school_id_no'] = survey_df['school_id'].astype(str).apply(lambda x: int(x.split('_')[-1]) if '_' in x else int(x))
            else:
                return None, None, "Survey file must contain 'school_id_no' or 'school_id' column."
        if 'school_name' not in survey_df.columns:
            survey_df['school_name'] = survey_df['school_id_no'].apply(lambda x: f"School_{x}")
        def month_str_to_num(month_str):
            try:
                year = int(month_str[:4])
                month = int(month_str[5:])
                return (year - 2026) * 12 + month
            except:
                return 0
        survey_df['month_num'] = survey_df['month'].apply(month_str_to_num)
        school_info = survey_df[['school_id_no', 'school_name']].drop_duplicates().sort_values('school_id_no')
        return survey_df, school_info, None
    except Exception as e:
        return None, None, f"Error processing survey: {str(e)}"

def process_metadata(metadata_df):
    if metadata_df is None:
        return None, "No metadata file uploaded."
    try:
        if 'school_id_no' not in metadata_df.columns:
            if 'school' in metadata_df.columns:
                metadata_df['school_id_no'] = metadata_df['school'].astype(str).apply(lambda x: int(x.split('_')[-1]) if '_' in x else int(x))
            else:
                return None, "Metadata file must contain 'school_id_no' or 'school' column."
        if 'year_undertaken' not in metadata_df.columns:
            metadata_df['year_undertaken'] = 2025
        if 'utilization_date' not in metadata_df.columns:
            metadata_df['utilization_date'] = ''
        if 'publication_link' not in metadata_df.columns:
            metadata_df['publication_link'] = ''
        if 'years_of_service' not in metadata_df.columns:
            metadata_df['years_of_service'] = None
        if 'teacher_rank' not in metadata_df.columns:
            metadata_df['teacher_rank'] = None
        if 'educational_attainment' not in metadata_df.columns:
            metadata_df['educational_attainment'] = None
        metadata_df['upload_date'] = pd.to_datetime(metadata_df['upload_date'])
        return metadata_df, None
    except Exception as e:
        return None, f"Error processing metadata: {str(e)}"

def get_latest_survey(survey_df, school_id):
    school_data = survey_df[survey_df['school_id_no'] == school_id]
    if school_data.empty:
        return None
    return school_data.sort_values('month_num').iloc[-1]

# ---------- Radar chart (clean version with bottom annotation) ----------
def radar_chart(survey_row, school_name, dark_mode):
    variables = ['R (M0)', 'A (M1)', 'C (M2)', 'S (M3)', 'I (M4)', 'P (M5)', 'M (M6)']
    value_map = {
        'R (M0)': survey_row['R'],
        'A (M1)': survey_row['A'],
        'C (M2)': survey_row['C'],
        'S (M3)': survey_row['S'],
        'I (M4)': survey_row['I'],
        'P (M5)': survey_row['P'],
        'M (M6)': survey_row['M']
    }
    values = [value_map[v] for v in variables]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=variables,
        fill='toself',
        name=school_name,
        line_color=USTP_GOLD,
        fillcolor=f"rgba(245, 166, 35, 0.3)"
    ))

    template = 'plotly_dark' if dark_mode else 'plotly_white'
    fig.update_layout(
        template=template,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                color=USTP_GOLD if dark_mode else USTP_DARK_BLUE
            ),
            angularaxis=dict(
                direction="clockwise",
                tickfont=dict(size=11, color=USTP_GOLD if dark_mode else USTP_DARK_BLUE)
            )
        ),
        title=f"Current Research Culture Profile (latest quarter)<br>{school_name}",
        showlegend=False,
        font=dict(color=USTP_GOLD if dark_mode else USTP_DARK_BLUE),
        annotations=[
            dict(
                text="↻ Milestone cycle direction (clockwise)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.12,
                showarrow=False,
                font=dict(size=13, color=USTP_GOLD if dark_mode else USTP_DARK_BLUE),
                bgcolor="rgba(255,255,255,0.0)",
                bordercolor="rgba(0,0,0,0)"
            )
        ],
        height=500,
        margin=dict(l=60, r=80, t=80, b=100)
    )
    return fig

# ---------- Research Outputs Dashboard (with new rank & attainment charts) ----------
def research_outputs_dashboard(metadata_df, school_id, school_name, dark_mode):
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id]
    if school_meta.empty:
        st.info(f"No research outputs for {school_name}.")
        return None

    st.caption("📝 This dashboard displays all research outputs from the uploaded metadata, independent of simulation duration.")

    # Theme Distribution
    theme_counts = school_meta['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    fig_theme = px.bar(theme_counts, x='Theme', y='Count', title=f"Theme Distribution – {school_name}", color='Theme', color_discrete_sequence=[USTP_GOLD, DEPED_RED, USTP_DARK_BLUE])
    fig_theme.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_theme, use_container_width=True)
    top_theme = theme_counts.iloc[0]['Theme'] if not theme_counts.empty else "N/A"
    st.caption(f"📝 Research outputs are most concentrated in '{top_theme}'. This suggests the school’s research focus area.")

    # Theme Utilisation Rate
    if 'utilized_by_school' in school_meta.columns:
        theme_util = school_meta.groupby('theme')['utilized_by_school'].mean().reset_index()
        theme_util.columns = ['Theme', 'Utilisation Rate']
        fig_theme_util = px.bar(theme_util, x='Theme', y='Utilisation Rate', title=f"Theme Utilisation Rate – {school_name}", color='Utilisation Rate', color_continuous_scale=['#F5A623', '#0D2B5E'])
        fig_theme_util.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
        st.plotly_chart(fig_theme_util, use_container_width=True)
        top_util_theme = theme_util.loc[theme_util['Utilisation Rate'].idxmax(), 'Theme'] if not theme_util.empty else "N/A"
        st.caption(f"📝 The theme with the highest utilisation rate is '{top_util_theme}'. This indicates that research in this area is most likely to be translated into practice.")
    else:
        top_util_theme = "N/A"

    # Publication Status
    status_counts = school_meta['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig_status = px.bar(status_counts, x='Status', y='Count', title=f"Publication Status – {school_name}", color='Status', color_discrete_sequence=[USTP_DARK_BLUE, USTP_GOLD, DEPED_MAROON])
    fig_status.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_status, use_container_width=True)
    published = status_counts[status_counts['Status']=='published']['Count'].sum() if not status_counts.empty else 0
    total = status_counts['Count'].sum() if not status_counts.empty else 0
    pub_rate = (published/total*100) if total>0 else 0
    st.caption(f"📝 {pub_rate:.1f}% of research outputs are published. A higher publication rate often correlates with greater institutional recognition.")

    # Research Output Timeline
    output_timeline = None
    if 'upload_date' in school_meta.columns:
        school_meta['quarter'] = school_meta['upload_date'].dt.to_period('Q').astype(str)
        output_timeline = school_meta.groupby('quarter').size().reset_index(name='count')
        if not output_timeline.empty:
            fig_timeline = px.line(output_timeline, x='quarter', y='count', title=f"Research Output Timeline – {school_name}", markers=True)
            fig_timeline.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', xaxis_title='Quarter', yaxis_title='Number of Outputs')
            st.plotly_chart(fig_timeline, use_container_width=True)
            latest_count = output_timeline.iloc[-1]['count'] if not output_timeline.empty else 0
            st.caption(f"📝 In the latest quarter, {latest_count} research outputs were produced. A rising trend indicates growing research productivity.")

    # Utilisation Over Time
    util_timeline = None
    if 'upload_date' in school_meta.columns and 'utilized_by_school' in school_meta.columns:
        school_meta['quarter'] = school_meta['upload_date'].dt.to_period('Q').astype(str)
        util_timeline = school_meta.groupby('quarter')['utilized_by_school'].mean().reset_index()
        util_timeline.columns = ['quarter', 'utilisation_rate']
        if not util_timeline.empty:
            fig_util_time = px.line(util_timeline, x='quarter', y='utilisation_rate', title=f"Utilisation Rate Over Time – {school_name}", markers=True)
            fig_util_time.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', xaxis_title='Quarter', yaxis_title='Utilisation Rate')
            st.plotly_chart(fig_util_time, use_container_width=True)
            latest_util = util_timeline.iloc[-1]['utilisation_rate'] if not util_timeline.empty else 0
            st.caption(f"📝 In the latest quarter, the utilisation rate is {latest_util:.1%}. A stable or increasing rate indicates effective translation of research into practice.")

    # Utilisation Rate (school-level)
    utilised = school_meta['utilized_by_school'].sum() if 'utilized_by_school' in school_meta.columns else 0
    total = len(school_meta)
    util_rate = (utilised / total * 100) if total > 0 else 0
    st.metric("📘 School‑level Research Utilisation Rate", f"{util_rate:.1f}%",
              help="Percentage of research outputs from this school that have been adopted into practice (e.g., new teaching strategies, policy changes).")
    st.caption(f"📝 {'High utilisation indicates strong translation of research into practice.' if util_rate > 70 else 'Moderate or low utilisation suggests a gap between research production and practical adoption.'}")

    # Teacher Productivity (Top 10)
    teacher_counts = school_meta['teacher_name'].value_counts().reset_index().head(10)
    teacher_counts.columns = ['Teacher', 'Number of Outputs']
    fig_teacher = px.bar(teacher_counts, x='Number of Outputs', y='Teacher', orientation='h', title=f"Teacher Productivity (Top 10) – {school_name}", color='Number of Outputs', color_continuous_scale=['#F5A623', '#0D2B5E'])
    fig_teacher.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_teacher, use_container_width=True)
    top_teacher = teacher_counts.iloc[0]['Teacher'] if not teacher_counts.empty else "N/A"
    st.caption(f"📝 The most productive teacher has {teacher_counts.iloc[0]['Number of Outputs'] if not teacher_counts.empty else 0} research outputs. Encouraging collaborative research could further strengthen culture.")

    # Years of Service vs Output
    if 'years_of_service' in school_meta.columns and not school_meta['years_of_service'].isna().all():
        teacher_summary = school_meta.groupby('teacher_name').agg(
            output_count=('document_type', 'count'),
            years_of_service=('years_of_service', 'first')
        ).reset_index()
        teacher_summary = teacher_summary.dropna(subset=['years_of_service'])
        if len(teacher_summary) > 1:
            x = teacher_summary['years_of_service']
            y = teacher_summary['output_count']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            trend_x = np.linspace(x.min(), x.max(), 100)
            trend_y = p(trend_x)
            fig_service = go.Figure()
            fig_service.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=12, color=USTP_GOLD, line=dict(color=USTP_DARK_BLUE, width=1)), text=teacher_summary['teacher_name'], hoverinfo='text+x+y', name='Teachers'))
            fig_service.add_trace(go.Scatter(x=trend_x, y=trend_y, mode='lines', line=dict(color=USTP_DARK_BLUE, width=2, dash='dash'), name='Trend'))
            fig_service.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', title=f"Years of Service vs Research Outputs – {school_name}", xaxis_title="Years of Service", yaxis_title="Number of Research Outputs", font=dict(color=USTP_GOLD if dark_mode else USTP_DARK_BLUE), showlegend=True, height=400)
            st.plotly_chart(fig_service, use_container_width=True)
            avg_output = teacher_summary['output_count'].mean()
            avg_service = teacher_summary['years_of_service'].mean()
            slope = z[0]
            direction = "increases" if slope > 0.1 else "decreases" if slope < -0.1 else "stays relatively stable"
            st.caption(f"📝 On average, teachers have {avg_service:.1f} years of service and produce {avg_output:.1f} outputs. The trend line suggests that research output {direction} with years of experience.")
        else:
            st.info("Insufficient data for a meaningful scatter plot (need at least 2 teachers).")
    else:
        st.info("📝 'years_of_service' column not found or all values are missing in metadata. To enable experience vs output analysis, add this column to your CSV file.")

    # ---- NEW: Research Outputs by Teacher Rank ----
    if 'teacher_rank' in school_meta.columns and not school_meta['teacher_rank'].isna().all():
        # Group by rank
        rank_group = school_meta.groupby('teacher_rank').size().reset_index(name='total_outputs')
        # Compute average outputs per teacher within each rank (we need to count teachers per rank)
        # We'll compute count of teachers per rank and then average
        teacher_rank_counts = school_meta.groupby('teacher_rank')['teacher_name'].nunique().reset_index(name='num_teachers')
        rank_summary = rank_group.merge(teacher_rank_counts, on='teacher_rank')
        rank_summary['avg_outputs'] = rank_summary['total_outputs'] / rank_summary['num_teachers']
        # Sort by rank (optional)
        fig_rank = px.bar(rank_summary, x='teacher_rank', y='total_outputs',
                          title=f"Research Outputs by Teacher Rank – {school_name}",
                          labels={'total_outputs': 'Total Outputs', 'teacher_rank': 'Teacher Rank'},
                          color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E'])
        fig_rank.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
        st.plotly_chart(fig_rank, use_container_width=True)
        # Add caption with average
        top_rank = rank_summary.loc[rank_summary['total_outputs'].idxmax(), 'teacher_rank'] if not rank_summary.empty else None
        if top_rank:
            st.caption(f"📝 The rank with the most outputs is '{top_rank}'. On average, teachers in this rank produce {rank_summary.loc[rank_summary['teacher_rank']==top_rank, 'avg_outputs'].values[0]:.1f} outputs per teacher.")
    else:
        st.info("📝 'teacher_rank' column not found or all values are missing. To enable rank analysis, add this column to your CSV file.")

    # ---- NEW: Research Outputs by Educational Attainment ----
    if 'educational_attainment' in school_meta.columns and not school_meta['educational_attainment'].isna().all():
        edu_group = school_meta.groupby('educational_attainment').size().reset_index(name='total_outputs')
        # Compute average outputs per teacher per attainment
        teacher_edu_counts = school_meta.groupby('educational_attainment')['teacher_name'].nunique().reset_index(name='num_teachers')
        edu_summary = edu_group.merge(teacher_edu_counts, on='educational_attainment')
        edu_summary['avg_outputs'] = edu_summary['total_outputs'] / edu_summary['num_teachers']
        fig_edu = px.bar(edu_summary, x='educational_attainment', y='total_outputs',
                         title=f"Research Outputs by Educational Attainment – {school_name}",
                         labels={'total_outputs': 'Total Outputs', 'educational_attainment': 'Educational Attainment'},
                         color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E'])
        fig_edu.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
        st.plotly_chart(fig_edu, use_container_width=True)
        top_edu = edu_summary.loc[edu_summary['total_outputs'].idxmax(), 'educational_attainment'] if not edu_summary.empty else None
        if top_edu:
            st.caption(f"📝 The attainment level with the most outputs is '{top_edu}'. On average, teachers with this attainment produce {edu_summary.loc[edu_summary['educational_attainment']==top_edu, 'avg_outputs'].values[0]:.1f} outputs per teacher.")
    else:
        st.info("📝 'educational_attainment' column not found or all values are missing. To enable educational attainment analysis, add this column to your CSV file.")

    # Top Teacher by Category
    st.markdown("#### 🏆 Top Teacher by Category")
    col_rank, col_service, col_edu = st.columns(3)
    top_rank_name = "N/A"
    top_service_name = "N/A"
    top_edu_name = "N/A"
    if 'teacher_rank' in school_meta.columns and not school_meta['teacher_rank'].isna().all():
        rank_group = school_meta.groupby(['teacher_rank', 'teacher_name']).size().reset_index(name='count')
        top_rank = rank_group.loc[rank_group.groupby('teacher_rank')['count'].idxmax()]
        if not top_rank.empty:
            top_rank_name = top_rank.iloc[0]['teacher_name']
            with col_rank:
                st.metric(label="Top by Rank", value=top_rank_name, help=f"Rank: {top_rank.iloc[0]['teacher_rank']} | Outputs: {top_rank.iloc[0]['count']}")
    else:
        with col_rank:
            st.info("Rank data not provided.")

    if 'years_of_service' in school_meta.columns and not school_meta['years_of_service'].isna().all():
        def service_bracket(years):
            if years <= 5: return "0-5"
            elif years <= 10: return "6-10"
            elif years <= 15: return "11-15"
            elif years <= 20: return "16-20"
            else: return "20+"
        school_meta['service_bracket'] = school_meta['years_of_service'].apply(service_bracket)
        bracket_group = school_meta.groupby(['service_bracket', 'teacher_name']).size().reset_index(name='count')
        top_bracket = bracket_group.loc[bracket_group.groupby('service_bracket')['count'].idxmax()]
        if not top_bracket.empty:
            top_service_name = top_bracket.iloc[0]['teacher_name']
            with col_service:
                st.metric(label="Top by Service Bracket", value=top_service_name, help=f"Bracket: {top_bracket.iloc[0]['service_bracket']} | Outputs: {top_bracket.iloc[0]['count']}")
    else:
        with col_service:
            st.info("Years of service data not provided.")

    if 'educational_attainment' in school_meta.columns and not school_meta['educational_attainment'].isna().all():
        edu_group = school_meta.groupby(['educational_attainment', 'teacher_name']).size().reset_index(name='count')
        top_edu = edu_group.loc[edu_group.groupby('educational_attainment')['count'].idxmax()]
        if not top_edu.empty:
            top_edu_name = top_edu.iloc[0]['teacher_name']
            with col_edu:
                st.metric(label="Top by Education", value=top_edu_name, help=f"Education: {top_edu.iloc[0]['educational_attainment']} | Outputs: {top_edu.iloc[0]['count']}")
    else:
        with col_edu:
            st.info("Educational attainment data not provided.")

    return {
        'top_theme': top_theme,
        'top_util_theme': top_util_theme,
        'top_teacher': top_teacher,
        'top_rank_name': top_rank_name,
        'top_service_name': top_service_name,
        'top_edu_name': top_edu_name,
        'output_timeline': output_timeline,
        'util_timeline': util_timeline,
        'theme_util': theme_util if 'theme_util' in locals() else None
    }

def cycle_research_correlation(agent, metadata_df, school_id, dark_mode):
    if not agent.cycle_improvements:
        st.info("No cycles completed yet for this school.")
        return
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id]
    def date_to_month_num(d):
        return (d.year - 2026) * 12 + d.month
    school_meta['month_num'] = school_meta['upload_date'].apply(date_to_month_num)
    cumulative_outputs = []
    for rec in agent.cycle_improvements:
        num_outputs = len(school_meta[school_meta['month_num'] <= rec.completion_month])
        cumulative_outputs.append(num_outputs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c.cycle_number for c in agent.cycle_improvements], y=cumulative_outputs,
                             mode='markers+lines', marker=dict(size=10, color=USTP_GOLD),
                             line=dict(color=USTP_DARK_BLUE), name='Research outputs'))
    fig.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', title="Cycle vs Cumulative Research Outputs", xaxis_title="Cycle Number",
                      yaxis_title="Number of Research Outputs (cumulative)", showlegend=False, font=dict(color=USTP_GOLD if dark_mode else USTP_DARK_BLUE))
    st.plotly_chart(fig, use_container_width=True)
    if len(cumulative_outputs) >= 2:
        increase = cumulative_outputs[-1] - cumulative_outputs[-2]
        if increase > 0:
            st.caption(f"📝 Research output accumulation increases with each cycle (+{increase} outputs from previous cycle). This suggests a growing research culture.")
        else:
            st.caption("📝 Research output growth has plateaued across cycles. Consider policies to revitalise research engagement.")
    elif len(cumulative_outputs) == 1:
        st.caption("📝 First cycle completed. Continued research output will be needed to build sustainability.")

# ---------- Division-Level Analysis ----------
def division_level_analysis(survey_df, metadata_df, history_per_school, sim_agents, dark_mode):
    st.markdown("### 🔍 Division‑Level Analysis")

    # ---- 1. Teacher Productivity Leaderboard ----
    st.markdown("#### 🏆 Teacher Productivity Leaderboard (Division‑Wide)")
    if not metadata_df.empty:
        teacher_summary = metadata_df.groupby(['teacher_name', 'school_id_no']).size().reset_index(name='total_outputs')
        school_names = survey_df[['school_id_no', 'school_name']].drop_duplicates()
        teacher_summary = teacher_summary.merge(school_names, on='school_id_no', how='left')
        if 'teacher_rank' in metadata_df.columns:
            rank_info = metadata_df.groupby('teacher_name')['teacher_rank'].first().reset_index()
            teacher_summary = teacher_summary.merge(rank_info, on='teacher_name', how='left')
        if 'educational_attainment' in metadata_df.columns:
            edu_info = metadata_df.groupby('teacher_name')['educational_attainment'].first().reset_index()
            teacher_summary = teacher_summary.merge(edu_info, on='teacher_name', how='left')
        if 'years_of_service' in metadata_df.columns:
            service_info = metadata_df.groupby('teacher_name')['years_of_service'].first().reset_index()
            teacher_summary = teacher_summary.merge(service_info, on='teacher_name', how='left')
        teacher_summary = teacher_summary.sort_values('total_outputs', ascending=False).head(20)
        st.dataframe(teacher_summary[['teacher_name', 'school_name', 'total_outputs', 'teacher_rank', 'educational_attainment', 'years_of_service']])
        st.caption("📝 Top 20 teachers across the division by research output count. Use this to identify research champions.")
        top_div_teacher = teacher_summary.iloc[0]['teacher_name'] if not teacher_summary.empty else "N/A"
        top_div_school = teacher_summary.iloc[0]['school_name'] if not teacher_summary.empty else "N/A"
        top_div_outputs = teacher_summary.iloc[0]['total_outputs'] if not teacher_summary.empty else 0
    else:
        top_div_teacher = top_div_school = "N/A"
        top_div_outputs = 0

    # ---- 2. Correlation Heatmap ----
    st.markdown("#### 📊 Correlation Heatmap: Survey Variables vs Research Output Count")
    top_corr_var = "N/A"
    top_corr_val = 0
    if survey_df is not None and not metadata_df.empty:
        survey_agg = survey_df.groupby(['school_id_no', 'month_num'])[['R','A','C','S','I','P','M']].mean().reset_index()
        meta = metadata_df.copy()
        meta['month_num'] = meta['upload_date'].apply(lambda d: (d.year - 2026)*12 + d.month)
        output_counts = meta.groupby(['school_id_no', 'month_num']).size().reset_index(name='output_count')
        merged = survey_agg.merge(output_counts, on=['school_id_no', 'month_num'], how='inner')
        if not merged.empty:
            corr = merged[['R','A','C','S','I','P','M','output_count']].corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale='Blues', aspect='auto')
            fig_corr.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)
            corr_vals = corr['output_count'].drop('output_count')
            if not corr_vals.empty:
                top_corr_var = corr_vals.abs().idxmax()
                top_corr_val = corr_vals[top_corr_var]
                st.caption(f"📝 The variable most strongly correlated with research output is '{top_corr_var}' (r = {top_corr_val:.2f}). {'Positive' if top_corr_val>0 else 'Negative'} correlation suggests that {'higher' if top_corr_val>0 else 'lower'} {top_corr_var} is associated with more research outputs.")
        else:
            st.info("Insufficient data to compute correlation (need survey and metadata for the same quarters).")

    # ---- 3. Milestone Transition Analysis ----
    st.markdown("#### ⏱️ Milestone Transition Analysis (Average Months per Milestone)")
    all_durations = {m: [] for m in range(7)}
    for agent in sim_agents:
        hist = history_per_school.get(agent.real_id)
        if hist and 'milestone' in hist:
            milestones = hist['milestone']
            for i in range(1, len(milestones)):
                if milestones[i] != milestones[i-1]:
                    start = milestones.index(milestones[i-1], 0, i) if milestones[i-1] in milestones[:i] else i-1
                    duration = i - start
                    all_durations[milestones[i-1]].append(duration)
            if milestones:
                last_milestone = milestones[-1]
                start = milestones.index(last_milestone, 0, len(milestones)) if last_milestone in milestones else len(milestones)-1
                duration = len(milestones) - start
                all_durations[last_milestone].append(duration)
    avg_durations = {m: np.mean(v) if v else np.nan for m, v in all_durations.items()}
    durations_df = pd.DataFrame({
        'Milestone': [f'M{i}' for i in range(7)],
        'Avg Months': [avg_durations.get(i, np.nan) for i in range(7)]
    }).dropna()
    bottleneck_milestone = "N/A"
    bottleneck_time = 0
    if not durations_df.empty:
        fig_dur = px.bar(durations_df, x='Milestone', y='Avg Months', title="Average Months Spent per Milestone", color='Avg Months', color_continuous_scale=['#F5A623', '#0D2B5E'])
        fig_dur.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', xaxis_title='Milestone', yaxis_title='Average Months')
        st.plotly_chart(fig_dur, use_container_width=True)
        max_row = durations_df.loc[durations_df['Avg Months'].idxmax()]
        bottleneck_milestone = max_row['Milestone']
        bottleneck_time = max_row['Avg Months']
        st.caption(f"📝 Schools spend the most time on average in Milestone {bottleneck_milestone} ({bottleneck_time:.1f} months). This indicates a potential bottleneck for research culture progression.")
    else:
        st.info("Not enough transition data to compute milestone durations.")

    return {
        'top_div_teacher': top_div_teacher,
        'top_div_school': top_div_school,
        'top_div_outputs': top_div_outputs,
        'top_corr_var': top_corr_var,
        'top_corr_val': top_corr_val,
        'bottleneck_milestone': bottleneck_milestone,
        'bottleneck_time': bottleneck_time
    }

def school_comparison_dashboard(survey_df, history_per_school, school_info, selected_school_ids, dark_mode):
    st.markdown("### 📊 Comparative School Analysis")
    if len(selected_school_ids) < 2:
        st.info("Please select at least two schools to compare.")
        return

    histories = {}
    for sid in selected_school_ids:
        hist = history_per_school.get(sid)
        if hist:
            histories[sid] = hist

    if not histories:
        st.info("No simulation history for selected schools. Run the simulation first.")
        return

    fig_comp = make_subplots(rows=2, cols=1, subplot_titles=("RCSI Comparison", "Milestone Comparison"))
    for sid, hist in histories.items():
        school_name = school_info[school_info['school_id_no']==sid]['school_name'].values[0] if sid in school_info['school_id_no'].values else f"School {sid}"
        fig_comp.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'], mode='lines', name=f"{school_name} RCSI"), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'], mode='lines', name=f"{school_name} Milestone"), row=2, col=1)
    fig_comp.update_layout(height=600, template='plotly_dark' if dark_mode else 'plotly_white')
    fig_comp.update_xaxes(title_text="Month", row=1, col=1)
    fig_comp.update_yaxes(title_text="RCSI", row=1, col=1)
    fig_comp.update_xaxes(title_text="Month", row=2, col=1)
    fig_comp.update_yaxes(title_text="Milestone", row=2, col=1)
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption("📝 Overlay of RCSI and Milestone progress for selected schools. Compare which schools are advancing faster and which are lagging.")

# ------------------------------------------------------------
# Helper to interpret average milestone
# ------------------------------------------------------------
def interpret_avg_milestone(avg_milestone):
    if avg_milestone < 0.5:
        return f"{avg_milestone:.1f} → between M0 and M1, approaching M1"
    elif avg_milestone < 1.5:
        return f"{avg_milestone:.1f} → between M1 and M2"
    elif avg_milestone < 2.5:
        return f"{avg_milestone:.1f} → between M2 and M3"
    elif avg_milestone < 3.5:
        return f"{avg_milestone:.1f} → between M3 and M4"
    elif avg_milestone < 4.5:
        return f"{avg_milestone:.1f} → between M4 and M5"
    elif avg_milestone < 5.5:
        return f"{avg_milestone:.1f} → between M5 and M6"
    else:
        return f"{avg_milestone:.1f} → at or beyond M6 (Impact Realization)"

# ------------------------------------------------------------
# Baseline Synopsis Generator
# ------------------------------------------------------------
def generate_baseline_synopsis(survey_row, school_name, metadata_df):
    variables = ['R','A','C','S','I','P','M']
    values = {v: survey_row[v] for v in variables}
    
    strengths = [v for v in variables if values[v] >= 0.6]
    gaps = [v for v in variables if values[v] <= 0.3]
    moderate = [v for v in variables if 0.3 < values[v] < 0.6]
    
    baseline_rcsi = np.mean([values[v] for v in variables])
    
    recommendations = []
    if 'C' in gaps:
        recommendations.append("🔹 **Priority 1: Build Teacher Capacity (C).** Conduct training workshops on research methods and data analysis.")
    if 'S' in gaps:
        recommendations.append("🔹 **Priority 2: Improve Structured Support (S).** Allocate budget and time for research activities.")
    if 'I' in gaps:
        recommendations.append("🔹 **Priority 3: Institutional Anchoring (I).** Embed research into school plans and regular meetings.")
    if 'P' in gaps:
        recommendations.append("🔹 **Priority 4: Strengthen Community of Practice (P).** Establish regular research sharing forums and peer mentoring.")
    if 'M' in gaps:
        recommendations.append("🔹 **Priority 5: Enhance Impact Realization (M).** Document and share evidence of research impact.")
    if not recommendations:
        recommendations.append("✅ All variables are at moderate or high levels. Maintain current policies and focus on continuous improvement.")
    
    return {
        'strengths': strengths,
        'gaps': gaps,
        'moderate': moderate,
        'baseline_rcsi': baseline_rcsi,
        'recommendations': recommendations,
        'values': values
    }

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="CDO Division Research Culture Sustainability Framework", layout="wide")
st.markdown("<h1 style='text-align: center; color: #0D2B5E;'>CDO Division Research Culture Sustainability Framework</h1>", unsafe_allow_html=True)

# Initialize session state variables
if 'max_schools' not in st.session_state:
    st.session_state.max_schools = 200
if 'num_schools' not in st.session_state:
    st.session_state.num_schools = 0  # changed from 20 to 0

with st.sidebar:
    st.markdown(f"<h2 style='color: {USTP_DARK_BLUE};'>Controls</h2>", unsafe_allow_html=True)
    dark_mode = st.checkbox("🌙 Dark Mode", value=False)
    apply_theme(dark_mode)

    st.metric(label="🏫 Total Schools Loaded", value=st.session_state.num_schools, help="Number of schools detected in the uploaded survey data.")

    st.markdown("---")
    st.markdown("#### 📊 Baseline Analysis")
    baseline_btn = st.button("🔍 Analyze Baseline", use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Policy Levers</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        u_train = st.slider("Training freq.", 0.0, 1.0, 0.5, 0.05)
        u_mentor = st.slider("Mentorship ratio", 0.0, 1.0, 0.5, 0.05)
        u_budget = st.slider("Support budget", 0.0, 1.0, 0.5, 0.05)
    with col2:
        u_lead = st.slider("Leadership commit.", 0.0, 1.0, 0.5, 0.05)
        u_collab = st.slider("Collaboration freq.", 0.0, 1.0, 0.5, 0.05)
    levers = {'u_train': u_train, 'u_mentor': u_mentor, 'u_budget': u_budget, 'u_lead': u_lead, 'u_collab': u_collab}

    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Simulation Parameters</h3>", unsafe_allow_html=True)
    duration = st.selectbox("Run duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=9)
    random_events = st.checkbox("Enable random events", value=False)
    use_survey = st.checkbox("Override with survey data", value=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Simulation Actions")
    col_buttons = st.columns(3)
    with col_buttons[0]:
        run_btn = st.button("🚀 Run", use_container_width=True)
    with col_buttons[1]:
        step_btn = st.button("⏭️ Step (1 month)", use_container_width=True)
    with col_buttons[2]:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    st.caption("**Run:** Full forecast for selected duration (resets history). **Step:** Advance one month without resetting (observe gradual changes).")

    st.markdown("---")
    st.markdown("#### 📥 Export Data")
    export_btn = st.button("📊 Export results (CSV)", use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>📄 Download Templates</h3>", unsafe_allow_html=True)
    st.caption("Download blank CSV templates to fill with your data.")
    survey_template = """month,school_id_no,school_name,R,A,C,S,I,P,M
2026-01,1,School_1,0.32,0.41,0.28,0.15,0.14,0.19,0.08"""
    metadata_template = """upload_date,teacher_name,school_id_no,document_type,title,theme,status,publication_link,utilized_by_school,utilization_date,year_undertaken,years_of_service,teacher_rank,educational_attainment
2026-03-15,Anna Reyes,1,abstract,Improving Reading,Teaching Strategies,published,https://doi.org/10.1234,True,2026-02-10,2025,10,Teacher II,Master's"""
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.download_button(label="📄 Survey Template (CSV)", data=survey_template, file_name="quarterly_survey_template.csv", mime="text/csv", use_container_width=True)
    with col_t2:
        st.download_button(label="📄 Metadata Template (CSV)", data=metadata_template, file_name="research_metadata_template.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>📂 Data Upload</h3>", unsafe_allow_html=True)
    st.caption("Upload your filled CSV files below:")
    survey_file = st.file_uploader("Upload quarterly survey (CSV)", type=["csv"], key="survey")
    metadata_file = st.file_uploader("Upload research metadata (CSV)", type=["csv"], key="metadata")

# ------------------------------------------------------------
# Main area
# ------------------------------------------------------------
if survey_file is not None and metadata_file is not None:
    try:
        survey_df = pd.read_csv(survey_file)
        metadata_df = pd.read_csv(metadata_file)
        survey_df, school_info, survey_error = process_survey(survey_df)
        metadata_df, meta_error = process_metadata(metadata_df)
        if survey_error:
            st.error(f"Survey error: {survey_error}")
        elif meta_error:
            st.error(f"Metadata error: {meta_error}")
        else:
            actual_count = len(school_info)
            if st.session_state.num_schools != actual_count:
                st.session_state.num_schools = actual_count
                st.rerun()

            st.success(f"Loaded {actual_count} schools.")
            
            # ---------- ALWAYS VISIBLE (BASELINE DATA) ----------
            school_ids = school_info['school_id_no'].tolist()
            school_options = [f"ID {sid}: {school_info[school_info['school_id_no']==sid]['school_name'].values[0]}" for sid in school_ids]
            selected_school_label = st.selectbox("Select school", school_options, index=0)
            selected_school_id = int(selected_school_label.split(":")[0].split()[1])
            selected_school_name = school_info[school_info['school_id_no']==selected_school_id]['school_name'].values[0]
            
            # Centered header for Baseline
            st.markdown("<h2 style='text-align: center;'>📋 Baseline from Uploaded Data</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown("### Research Outputs (Recent)")
            df_show = metadata_df[metadata_df['school_id_no'] == selected_school_id].copy()
            if not df_show.empty:
                df_show_sorted = df_show.sort_values('upload_date', ascending=False)
                st.dataframe(df_show_sorted[['teacher_name', 'year_undertaken', 'title', 'theme', 'status', 'utilized_by_school']].head(10))
            else:
                st.info("No research outputs for this school.")
            
            # Radar Chart with Legend (adjusted ratio for smaller legend)
            latest = get_latest_survey(survey_df, selected_school_id)
            if latest is not None:
                col_left, col_right = st.columns([1, 5])  # smaller left column for legend
                with col_left:
                    st.markdown("**📌 Legend:**")
                    legend_text = """
                    - **R (M0)** → Readiness & Relevance
                    - **A (M1)** → Awareness to Action
                    - **C (M2)** → Capacity Spark
                    - **S (M3)** → Structured Support
                    - **I (M4)** → Institutional Anchoring
                    - **P (M5)** → Community of Practice
                    - **M (M6)** → Impact Realization
                    """
                    # Use smaller font size for legend
                    st.markdown(f'<div style="font-size: 12px;">{legend_text}</div>', unsafe_allow_html=True)
                with col_right:
                    latest_dict = latest.to_dict()
                    st.plotly_chart(radar_chart(latest_dict, selected_school_name, dark_mode), use_container_width=True)
            else:
                st.info("No survey data for current quarter.")
            
            with st.expander("📚 Research Outputs Dashboard (for selected school)"):
                school_metrics = research_outputs_dashboard(metadata_df, selected_school_id, selected_school_name, dark_mode)
            
            # ---------- BASELINE SYNOPSIS ----------
            if baseline_btn:
                latest_row = get_latest_survey(survey_df, selected_school_id)
                if latest_row is not None:
                    baseline_synopsis = generate_baseline_synopsis(latest_row, selected_school_name, metadata_df)
                    st.session_state.baseline_synopsis = baseline_synopsis
                    st.session_state.baseline_survey_row = latest_row.to_dict()
                else:
                    st.warning("No survey data available to generate baseline.")
            
            if 'baseline_synopsis' in st.session_state:
                bs = st.session_state.baseline_synopsis
                st.markdown("### 📊 Baseline Synopsis")
                st.markdown(f"""
                <div style="background-color: {'#2E2E2E' if dark_mode else '#E3F2FD'}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {DARK_TEXT if dark_mode else 'inherit'};">
                <b>School: {selected_school_name}</b><br>
                <b>Baseline RCSI:</b> {bs['baseline_rcsi']:.3f}<br>
                <b>Strengths (≥0.6):</b> {', '.join(bs['strengths']) if bs['strengths'] else 'None'}<br>
                <b>Critical Gaps (≤0.3):</b> {', '.join(bs['gaps']) if bs['gaps'] else 'None'}<br>
                <b>Moderate (0.3–0.6):</b> {', '.join(bs['moderate']) if bs['moderate'] else 'None'}<br>
                <b>Actionable Recommendations:</b><br>
                {'<br>'.join(bs['recommendations'])}
                </div>
                """, unsafe_allow_html=True)

            # ---------- SIMULATION DEPENDENT ----------
            if 'sim' not in st.session_state:
                st.session_state.sim = Simulation(num_schools=actual_count, random_events=random_events)
                st.session_state.current_month = 0
                st.session_state.total_months = 0
                st.session_state.history = {sid: {'R':[],'A':[],'C':[],'S':[],'I':[],'P':[],'M':[],'month':[],'milestone':[],'running_outcome':[]} for sid in school_ids}
                for idx, agent in enumerate(st.session_state.sim.agents):
                    agent.real_id = school_ids[idx]
                for agent in st.session_state.sim.agents:
                    school_metadata = metadata_df[metadata_df['school_id_no'] == agent.real_id]
                    agent.A = min(1.0, agent.A + len(school_metadata[school_metadata['document_type']=='abstract'])*0.01)
                    agent.M = min(1.0, agent.M + len(school_metadata[school_metadata['status']=='published'])*0.02)
                    agent.C = min(1.0, agent.C + len(school_metadata[school_metadata['document_type']=='full_paper'])*0.005)
                    agent.P = min(1.0, agent.P + school_metadata['theme'].nunique()*0.01)

            if run_btn:
                st.session_state.sim = Simulation(num_schools=actual_count, random_events=random_events)
                for idx, agent in enumerate(st.session_state.sim.agents):
                    agent.real_id = school_ids[idx]
                for agent in st.session_state.sim.agents:
                    school_metadata = metadata_df[metadata_df['school_id_no'] == agent.real_id]
                    agent.A = min(1.0, agent.A + len(school_metadata[school_metadata['document_type']=='abstract'])*0.01)
                    agent.M = min(1.0, agent.M + len(school_metadata[school_metadata['status']=='published'])*0.02)
                    agent.C = min(1.0, agent.C + len(school_metadata[school_metadata['document_type']=='full_paper'])*0.005)
                    agent.P = min(1.0, agent.P + school_metadata['theme'].nunique()*0.01)
                st.session_state.current_month = 0
                st.session_state.total_months = 0
                st.session_state.history = {sid: {'R':[],'A':[],'C':[],'S':[],'I':[],'P':[],'M':[],'month':[],'milestone':[],'running_outcome':[]} for sid in school_ids}
                for m in range(1, duration+1):
                    if use_survey:
                        for agent in st.session_state.sim.agents:
                            row = survey_df[(survey_df['school_id_no'] == agent.real_id) & (survey_df['month_num'] == st.session_state.current_month + m)]
                            if not row.empty:
                                r = row.iloc[0]
                                agent.R, agent.A, agent.C, agent.S, agent.I, agent.P, agent.M = r[['R','A','C','S','I','P','M']]
                    st.session_state.sim.step(levers, st.session_state.current_month + m)
                    st.session_state.current_month += 1
                    st.session_state.total_months += 1
                    for agent in st.session_state.sim.agents:
                        h = st.session_state.history[agent.real_id]
                        h['month'].append(st.session_state.total_months)
                        for var in ['R','A','C','S','I','P','M']:
                            h[var].append(getattr(agent, var))
                        h['milestone'].append(agent.current_milestone)
                        h['running_outcome'].append(agent.running_total_outcome)
                st.rerun()

            if step_btn:
                m = 1
                if use_survey:
                    for agent in st.session_state.sim.agents:
                        row = survey_df[(survey_df['school_id_no'] == agent.real_id) & (survey_df['month_num'] == st.session_state.current_month + m)]
                        if not row.empty:
                            r = row.iloc[0]
                            agent.R, agent.A, agent.C, agent.S, agent.I, agent.P, agent.M = r[['R','A','C','S','I','P','M']]
                st.session_state.sim.step(levers, st.session_state.current_month + m)
                st.session_state.current_month += 1
                st.session_state.total_months += 1
                for agent in st.session_state.sim.agents:
                    h = st.session_state.history[agent.real_id]
                    h['month'].append(st.session_state.total_months)
                    for var in ['R','A','C','S','I','P','M']:
                        h[var].append(getattr(agent, var))
                    h['milestone'].append(agent.current_milestone)
                    h['running_outcome'].append(agent.running_total_outcome)
                st.rerun()

            if reset_btn:
                st.session_state.sim = Simulation(num_schools=actual_count, random_events=random_events)
                for idx, agent in enumerate(st.session_state.sim.agents):
                    agent.real_id = school_ids[idx]
                for agent in st.session_state.sim.agents:
                    school_metadata = metadata_df[metadata_df['school_id_no'] == agent.real_id]
                    agent.A = min(1.0, agent.A + len(school_metadata[school_metadata['document_type']=='abstract'])*0.01)
                    agent.M = min(1.0, agent.M + len(school_metadata[school_metadata['status']=='published'])*0.02)
                    agent.C = min(1.0, agent.C + len(school_metadata[school_metadata['document_type']=='full_paper'])*0.005)
                    agent.P = min(1.0, agent.P + school_metadata['theme'].nunique()*0.01)
                st.session_state.current_month = 0
                st.session_state.total_months = 0
                st.session_state.history = {sid: {'R':[],'A':[],'C':[],'S':[],'I':[],'P':[],'M':[],'month':[],'milestone':[],'running_outcome':[]} for sid in school_ids}
                st.rerun()

            # ---------- Simulation-dependent outputs ----------
            if st.session_state.total_months > 0:
                st.markdown("<h2 style='text-align: center;'>⚙️ Simulated Data</h2>", unsafe_allow_html=True)
                st.markdown("---")
                
                hist = st.session_state.history.get(selected_school_id, None)
                agent = next((a for a in st.session_state.sim.agents if a.real_id == selected_school_id), None)
                if hist and agent:
                    # Main plots
                    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Variable Evolution", "Milestone Progress", "Research Culture Sustainability Index (RCSI)", "Improvement per Completed Cycle"))
                    colors = ['#1E88E5', USTP_GOLD, '#8E44AD', '#2ECC71', '#E67E22', DEPED_RED, '#1ABC9C']
                    vars_ = ['R','A','C','S','I','P','M']
                    for i, var in enumerate(vars_):
                        fig1.add_trace(go.Scatter(x=hist['month'], y=hist[var], mode='lines', name=var, line=dict(color=colors[i])), row=1, col=1)
                    fig1.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'], mode='lines', name='Milestone', line=dict(color=DEPED_RED, width=3)), row=1, col=2)
                    fig1.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'], mode='lines', name='RCSI', line=dict(color=USTP_GOLD, width=3)), row=2, col=1)
                    if agent.cycle_improvements:
                        cycles = [c.cycle_number for c in agent.cycle_improvements]
                        improvements = [c.total_improvement for c in agent.cycle_improvements]
                        fig1.add_trace(go.Bar(x=cycles, y=improvements, name='RCSI per cycle', marker_color=USTP_DARK_BLUE), row=2, col=2)
                    else:
                        fig1.add_annotation(text="No cycles completed yet", xref="x2 domain", yref="y2 domain", x=0.5, y=0.5, showarrow=False, row=2, col=2)
                    template = 'plotly_dark' if dark_mode else 'plotly_white'
                    fig1.update_layout(height=800, showlegend=True, font=dict(color=USTP_GOLD if dark_mode else USTP_DARK_BLUE), template=template)
                    fig1.update_xaxes(title_text="Month", row=1, col=1)
                    fig1.update_yaxes(title_text="Value (0-1)", row=1, col=1)
                    fig1.update_xaxes(title_text="Month", row=1, col=2)
                    fig1.update_yaxes(title_text="Milestone", row=1, col=2)
                    fig1.update_xaxes(title_text="Month", row=2, col=1)
                    fig1.update_yaxes(title_text="RCSI", row=2, col=1)
                    fig1.update_xaxes(title_text="Cycle Number", row=2, col=2)
                    fig1.update_yaxes(title_text="RCSI", row=2, col=2)
                    st.plotly_chart(fig1, use_container_width=True)

                    with st.expander("🔄 Cycle vs Research Outputs"):
                        cycle_research_correlation(agent, metadata_df, selected_school_id, dark_mode)

                    with st.expander("🏢 Division‑Level Analysis"):
                        div_metrics = division_level_analysis(survey_df, metadata_df, st.session_state.history, st.session_state.sim.agents, dark_mode)

                    # Comparative School Analysis
                    with st.expander("📊 Comparative School Analysis"):
                        all_schools = school_info['school_id_no'].tolist()
                        selected_comparison = st.multiselect(
                            "Select schools to compare (choose at least two)",
                            options=all_schools,
                            default=[],
                            format_func=lambda x: f"ID {x}: {school_info[school_info['school_id_no']==x]['school_name'].values[0]}"
                        )
                        school_comparison_dashboard(survey_df, st.session_state.history, school_info, selected_comparison, dark_mode)

                    # RCSI interpretation table
                    st.markdown("### 📈 Research Culture Sustainability Index (RCSI) Interpretation Table")
                    st.markdown("""
                    | RCSI Range | Level | Description |
                    |------------|-------|-------------|
                    | 0.0 – 0.2 | Very Low | Little to no accumulated research culture strength. |
                    | 0.2 – 0.4 | Low | Minimal ecosystem vitality; research culture still weak. |
                    | 0.4 – 0.6 | Moderate | Noticeable strength; research culture developing. |
                    | 0.6 – 0.8 | High | Strong ecosystem; research culture becoming sustainable. |
                    | 0.8 – 1.0 | Very High | Excellent vitality; research culture fully embedded. |
                    """)

                    # ===================== SIMULATION SYNOPSIS =====================
                    rcsi_val = agent.running_total_outcome
                    rcsi_level = "Exceptional"
                    for low,high,lev in [(0.0,0.2,"Very Low"), (0.2,0.4,"Low"), (0.4,0.6,"Moderate"), (0.6,0.8,"High"), (0.8,1.0,"Very High")]:
                        if low <= rcsi_val < high:
                            rcsi_level = lev
                            break
                    milestone_names = {0:"Milestone 0 (Readiness and Relevance)",1:"Milestone 1 (Awareness to Action)",
                                       2:"Milestone 2 (Capacity Spark)",3:"Milestone 3 (Structured Support)",
                                       4:"Milestone 4 (Institutional Anchoring)",5:"Milestone 5 (Community of Practice)",
                                       6:"Milestone 6 (Impact Realization)"}
                    milestone_name = milestone_names.get(agent.current_milestone, f"Milestone {agent.current_milestone}")

                    if agent.cycle_count >= 2:
                        cycle_text = f"has completed {agent.cycle_count} full cycles, indicating a self‑sustaining research culture where cyclical improvement is institutionalized."
                    elif agent.cycle_count == 1:
                        cycle_text = "has completed one full cycle, demonstrating initial sustainability but may need further reinforcement."
                    else:
                        cycle_text = "has not yet completed any full cycle, meaning the research culture is still in early formation and has not achieved cyclical momentum."

                    if agent.current_milestone == 0:
                        milestone_progress = "is at the very beginning of the journey."
                    elif agent.current_milestone <= 2:
                        milestone_progress = "has moved beyond initial readiness but remains in early capacity‑building phases."
                    elif agent.current_milestone <= 4:
                        milestone_progress = "has established structured support and is embedding research into institutional practice."
                    else:
                        milestone_progress = "is realising tangible impact and is approaching or has achieved cyclical sustainability."

                    key_R = hist['R'][-1] if hist['R'] else 0
                    key_M = hist['M'][-1] if hist['M'] else 0

                    # Additional insights from school_metrics
                    output_trend_text = ""
                    if school_metrics and school_metrics.get('output_timeline') is not None:
                        tl = school_metrics['output_timeline']
                        if len(tl) >= 2:
                            if tl.iloc[-1]['count'] > tl.iloc[-2]['count']:
                                output_trend_text = "Research output is increasing over time."
                            elif tl.iloc[-1]['count'] < tl.iloc[-2]['count']:
                                output_trend_text = "Research output is declining over time."
                            else:
                                output_trend_text = "Research output has remained stable over time."
                            avg_output = tl['count'].mean()
                            output_trend_text += f" On average, the school produces {avg_output:.1f} outputs per quarter."
                        else:
                            output_trend_text = "Research output data is limited; continuing to monitor will help identify trends."

                    theme_util_text = ""
                    if school_metrics and school_metrics.get('theme_util') is not None:
                        tu = school_metrics['theme_util']
                        if not tu.empty:
                            max_util = tu.loc[tu['Utilisation Rate'].idxmax()]
                            min_util = tu.loc[tu['Utilisation Rate'].idxmin()]
                            theme_util_text = f"The most utilised theme is '{max_util['Theme']}' ({max_util['Utilisation Rate']:.0%} utilisation), while '{min_util['Theme']}' has the lowest adoption ({min_util['Utilisation Rate']:.0%})."

                    top_teacher_text = ""
                    if school_metrics and school_metrics.get('top_teacher') != "N/A":
                        top_teacher_text = f"The school's top researcher is {school_metrics['top_teacher']}. "
                        if school_metrics.get('top_rank_name') != "N/A":
                            top_teacher_text += f"Rank: {school_metrics['top_rank_name']}. "
                        if school_metrics.get('top_edu_name') != "N/A":
                            top_teacher_text += f"Education: {school_metrics['top_edu_name']}."

                    coherent_text = f"""
                    After {st.session_state.total_months} months, {selected_school_name} (ID {selected_school_id}) has reached {milestone_name} and {cycle_text} 
                    The school’s Research Culture Sustainability Index (RCSI) is <b>{rcsi_val:.3f}</b>, which falls into the <b>{rcsi_level}</b> level. 
                    Key indicators: Readiness (R) = {key_R:.2f}, Impact (M) = {key_M:.2f}, and current Milestone = {agent.current_milestone}. 
                    This combination suggests that {milestone_progress} 
                    The RCSI level <b>{rcsi_level.lower()}</b> reinforces this assessment: a {rcsi_level.lower()} score indicates the overall health of the research ecosystem.
                    📊 {output_trend_text} 
                    📘 {theme_util_text} 
                    🏆 {top_teacher_text}
                    Overall, the school is on a path toward research culture sustainability, but further policy support may be needed to accelerate cycle completion.
                    """
                    st.markdown(f"""
                    <div style="background-color: {'#2E2E2E' if dark_mode else '#E3F2FD'}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {DARK_TEXT if dark_mode else 'inherit'};">
                    <b>📌 School {selected_school_id} ({selected_school_name}) – Simulation Synopsis</b><br>
                    {coherent_text}
                    </div>
                    """, unsafe_allow_html=True)

                    # ===================== BASELINE vs SIMULATION COMPARISON (TABLE) =====================
                    if 'baseline_synopsis' in st.session_state and 'baseline_survey_row' in st.session_state:
                        bs = st.session_state.baseline_synopsis
                        baseline_vals = st.session_state.baseline_survey_row
                        gaps = bs['gaps']
                        if gaps:
                            st.markdown("#### 🔍 Baseline vs Simulation Comparison (Critical Gaps)")
                            table_data = []
                            var_names = {
                                'R': 'Readiness',
                                'A': 'Awareness',
                                'C': 'Capacity',
                                'S': 'Structured Support',
                                'I': 'Institutional Anchoring',
                                'P': 'Community of Practice',
                                'M': 'Impact Realization'
                            }
                            for var in gaps:
                                base_val = baseline_vals[var]
                                sim_val = getattr(agent, var)
                                diff = sim_val - base_val
                                if diff > 0.01:
                                    status = "↑ Improving"
                                elif diff < -0.01:
                                    status = "↓ Regressing"
                                else:
                                    status = "→ Stable"
                                table_data.append({
                                    "Critical Gap": f"{var_names[var]} ({var})",
                                    "Baseline Value": f"{base_val:.2f}",
                                    "Simulation Value": f"{sim_val:.2f}",
                                    "Status": status
                                })
                            df_compare = pd.DataFrame(table_data)
                            st.table(df_compare)
                            improving = sum(1 for v in gaps if getattr(agent, v) - baseline_vals[v] > 0.01)
                            regressing = sum(1 for v in gaps if getattr(agent, v) - baseline_vals[v] < -0.01)
                            if improving > regressing:
                                summary = "Overall, the simulation indicates that most critical gaps are improving. The policy levers appear to be effective."
                            elif regressing > improving:
                                summary = "Overall, the simulation indicates that several critical gaps are regressing. Consider adjusting policy levers."
                            else:
                                summary = "Overall, the simulation shows mixed or stable results for critical gaps. Further analysis may be needed."
                            st.markdown(f"**Interpretation:** {summary}")

                    # ===================== DIVISION SYNOPSIS =====================
                    total_schools = len(st.session_state.sim.agents)
                    early_stage_count = sum(1 for a in st.session_state.sim.agents if a.current_milestone <= 2)
                    advanced_stage_count = sum(1 for a in st.session_state.sim.agents if a.current_milestone >= 4)
                    transitional_count = total_schools - early_stage_count - advanced_stage_count
                    early_percent = (early_stage_count / total_schools) * 100 if total_schools > 0 else 0
                    advanced_percent = (advanced_stage_count / total_schools) * 100 if total_schools > 0 else 0
                    transitional_percent = (transitional_count / total_schools) * 100 if total_schools > 0 else 0

                    if early_percent == 100:
                        early_text = "All schools"
                    elif early_percent >= 75:
                        early_text = f"The vast majority of schools ({early_percent:.1f}%)"
                    elif early_percent >= 50:
                        early_text = f"More than half of schools ({early_percent:.1f}%)"
                    elif early_percent > 0:
                        early_text = f"{early_percent:.1f}% of schools"
                    else:
                        early_text = "No schools"

                    if advanced_percent == 100:
                        advanced_text = "All schools"
                    elif advanced_percent >= 75:
                        advanced_text = f"The vast majority of schools ({advanced_percent:.1f}%)"
                    elif advanced_percent >= 50:
                        advanced_text = f"More than half of schools ({advanced_percent:.1f}%)"
                    elif advanced_percent > 0:
                        advanced_text = f"{advanced_percent:.1f}% of schools"
                    else:
                        advanced_text = "No schools"

                    if early_percent == 100:
                        sustainability_text = "All schools are still in early milestones (M0–M2); foundational capacity‑building is the priority to advance the division’s research culture."
                    elif early_percent >= 75:
                        sustainability_text = f"The vast majority ({early_percent:.1f}%) of schools are in early milestones (M0–M2); urgent interventions are needed to move them into higher stages."
                    elif early_percent >= 50:
                        sustainability_text = f"More than half ({early_percent:.1f}%) of schools are in early milestones (M0–M2); targeted policy support may accelerate progress."
                    elif early_percent > 0:
                        sustainability_text = f"{early_percent:.1f}% of schools remain in early milestones; continued efforts are required to reach sustainability."
                    else:
                        sustainability_text = "No schools are in early milestones; the division exhibits a strong, advanced research culture across most schools."

                    total_outcome = sum(a.running_total_outcome for a in st.session_state.sim.agents)
                    avg_rcsi = total_outcome / total_schools
                    level_avg = "Exceptional"
                    for low,high,lev in [(0.0,0.2,"Very Low"), (0.2,0.4,"Low"), (0.4,0.6,"Moderate"), (0.6,0.8,"High"), (0.8,1.0,"Very High")]:
                        if low <= avg_rcsi < high:
                            level_avg = lev
                            break
                    total_cycles = sum(a.cycle_count for a in st.session_state.sim.agents)
                    avg_milestone = np.mean([a.current_milestone for a in st.session_state.sim.agents])
                    avg_milestone_interpretation = interpret_avg_milestone(avg_milestone)
                    school_ids_in_sim = [agent.real_id for agent in st.session_state.sim.agents]
                    div_metadata = metadata_df[metadata_df['school_id_no'].isin(school_ids_in_sim)]
                    total_utilised = div_metadata['utilized_by_school'].sum() if 'utilized_by_school' in div_metadata.columns else 0
                    total_research_outputs = len(div_metadata)
                    div_util_rate = (total_utilised / total_research_outputs * 100) if total_research_outputs > 0 else 0

                    div_insights = div_metrics if 'div_metrics' in locals() else {}
                    top_div_teacher = div_insights.get('top_div_teacher', 'N/A')
                    top_div_school = div_insights.get('top_div_school', 'N/A')
                    top_div_outputs = div_insights.get('top_div_outputs', 0)
                    top_corr_var = div_insights.get('top_corr_var', 'N/A')
                    top_corr_val = div_insights.get('top_corr_val', 0)
                    bottleneck_milestone = div_insights.get('bottleneck_milestone', 'N/A')
                    bottleneck_time = div_insights.get('bottleneck_time', 0)

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
                        else:
                            output_trend_div = "Division output trend data is limited; continued monitoring is recommended."

                    corr_insight = ""
                    if top_corr_var != "N/A":
                        corr_insight = f"The heatmap shows that '{top_corr_var}' has the strongest correlation with research output (r = {top_corr_val:.2f}). Investing in {top_corr_var} may yield the highest return."

                    bottleneck_insight = ""
                    if bottleneck_milestone != "N/A":
                        bottleneck_insight = f"Schools spend the most time on average in {bottleneck_milestone} ({bottleneck_time:.1f} months). This is the critical bottleneck holding back division‑wide progress."

                    top_teacher_insight = ""
                    if top_div_teacher != "N/A":
                        top_teacher_insight = f"The division's top researcher is {top_div_teacher} from {top_div_school} with {top_div_outputs} outputs."

                    division_synopsis = f"""
                    <div style="background-color: {'#2E2E2E' if dark_mode else '#E8F5E9'}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {DARK_TEXT if dark_mode else 'inherit'};">
                    <b>🏢 Division‑Level Sustainability Synopsis (all {total_schools} schools)</b><br>
                    • Average milestone = {avg_milestone:.1f} → {avg_milestone_interpretation}<br>
                    • Total completed cycles across all schools = {total_cycles}<br>
                    • Average Research Culture Sustainability Index (RCSI) = <b>{avg_rcsi:.3f}</b> → <b>{level_avg}</b> level.<br>
                    • Average research utilisation rate = <b>{div_util_rate:.1f}%</b> (research adopted into practice).<br>
                    • Stage distribution: {early_text} are in early stages (milestone ≤2), {transitional_percent:.1f}% are at milestone 3 (transitional), and {advanced_text} are in advanced stages (milestone ≥4).<br>
                    <i>Division‑wide sustainability assessment:</i> {sustainability_text}<br><br>
                    📊 <b>Productivity:</b> {output_trend_div}<br>
                    🔍 <b>Key Driver:</b> {corr_insight}<br>
                    ⏱️ <b>Bottleneck:</b> {bottleneck_insight}<br>
                    🏆 <b>Top Division Researcher:</b> {top_teacher_insight}
                    </div>
                    """
                    st.markdown(division_synopsis, unsafe_allow_html=True)

                    with st.expander("📊 Graph Interpretations"):
                        st.markdown("""
                        - **Variable Evolution:** Shows how R, A, C, S, I, P, M change over time. Higher values (closer to 1) mean stronger readiness, awareness, capacity, etc.
                        - **Milestone Progress:** The school moves through milestones 0–6. Reaching milestone 6 and cycling back indicates a full sustainable cycle.
                        - **Research Culture Sustainability Index (RCSI):** Cumulative strength of the research ecosystem, derived from Impact Realization (M) and Collaboration (P).
                        - **Improvement per Completed Cycle:** Each bar shows the RCSI contributed by one cycle. Higher bars in later cycles indicate increasing effectiveness.
                        - **Radar Chart:** Current snapshot of the seven milestone‑linked variables – the ideal is a balanced, high‑value shape.
                        - **Research Outputs Dashboard:** Tracks themes, publication status, utilisation, teacher productivity, experience vs output, timeline, top teachers, and breakdown by rank and attainment.
                        - **Division‑Level Analysis:** Correlation heatmap, milestone transition bottlenecks, and teacher leaderboard.
                        - **Comparative Analysis:** Overlay multiple schools' RCSI and milestone progress.
                        - **Cycle vs Research Outputs:** Shows how research output accumulation relates to cycle progression.
                        """)

            # Export button
            if export_btn:
                all_data = []
                for agent in st.session_state.sim.agents:
                    h = st.session_state.history[agent.real_id]
                    for t in range(len(h['month'])):
                        row = {'school_id': agent.real_id, 'month': h['month'][t], 'milestone': h['milestone'][t], 'running_outcome': h['running_outcome'][t]}
                        for var in ['R','A','C','S','I','P','M']:
                            row[var] = h[var][t]
                        all_data.append(row)
                df_hist = pd.DataFrame(all_data)
                cycle_records = []
                for agent in st.session_state.sim.agents:
                    for rec in agent.cycle_improvements:
                        cycle_records.append({'school_id': agent.real_id, 'cycle_number': rec.cycle_number,
                                              'total_improvement': rec.total_improvement, 'completion_month': rec.completion_month})
                df_cycles = pd.DataFrame(cycle_records)
                st.download_button("Download simulation history", df_hist.to_csv(index=False).encode('utf-8'), "simulation_history.csv", "text/csv")
                st.download_button("Download cycle improvements", df_cycles.to_csv(index=False).encode('utf-8'), "cycle_improvements.csv", "text/csv")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload quarterly survey and research metadata CSV files to begin.")
