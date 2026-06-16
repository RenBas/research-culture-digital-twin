import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

# ============================================================
# USTP + DepEd Colour Palette
# ============================================================
USTP_DARK_BLUE = "#0D2B5E"
USTP_GOLD = "#F5A623"
DEPED_RED = "#D32F2F"
DEPED_MAROON = "#8B0000"
LIGHT_BG = "#F8F9FA"
DIVISION_GREEN = "#2E7D32"   # for division synopsis (kept from earlier)

# Custom CSS for Streamlit
st.markdown(f"""
<style>
    .reportview-container .main .block-container {{
        padding-top: 2rem;
    }}
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {{
        color: {USTP_DARK_BLUE};
    }}
    .sidebar .sidebar-content {{
        background-color: {LIGHT_BG};
        border-right: 2px solid {USTP_GOLD};
    }}
    .stButton > button {{
        background-color: {USTP_DARK_BLUE};
        color: white;
        border-radius: 5px;
        border: none;
        transition: 0.3s;
    }}
    .stButton > button:hover {{
        background-color: {USTP_GOLD};
        color: {USTP_DARK_BLUE};
    }}
    .stButton > button:focus {{
        box-shadow: none;
    }}
    /* Secondary buttons (Step, Reset) */
    div[data-testid="column"]:nth-of-type(2) .stButton > button,
    div[data-testid="column"]:nth-of-type(3) .stButton > button {{
        background-color: #6C757D;
    }}
    div[data-testid="column"]:nth-of-type(2) .stButton > button:hover,
    div[data-testid="column"]:nth-of-type(3) .stButton > button:hover {{
        background-color: {USTP_GOLD};
        color: {USTP_DARK_BLUE};
    }}
    .stSelectbox label, .stNumberInput label, .stCheckbox label {{
        font-weight: 500;
        color: {USTP_DARK_BLUE};
    }}
    .stDataFrame {{
        border: 1px solid #ddd;
    }}
    .css-1y4p8pa {{
        background-color: {LIGHT_BG};
    }}
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
        metadata_df['upload_date'] = pd.to_datetime(metadata_df['upload_date'])
        return metadata_df, None
    except Exception as e:
        return None, f"Error processing metadata: {str(e)}"

def get_latest_survey(survey_df, school_id):
    school_data = survey_df[survey_df['school_id_no'] == school_id]
    if school_data.empty:
        return None
    return school_data.sort_values('month_num').iloc[-1]

def radar_chart(survey_row, school_name):
    variables = ['R', 'A', 'C', 'S', 'I', 'P', 'M']
    values = [survey_row[v] for v in variables]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=variables,
        fill='toself',
        name=school_name,
        line_color=USTP_GOLD,
        fillcolor=f"rgba(245, 166, 35, 0.3)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color=USTP_DARK_BLUE)
        ),
        title=f"Current Research Culture Profile (latest quarter)<br>{school_name}",
        showlegend=False,
        font=dict(color=USTP_DARK_BLUE)
    )
    return fig

def research_outputs_dashboard(metadata_df, school_id, school_name):
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id]
    if school_meta.empty:
        st.info(f"No research outputs for {school_name}.")
        return
    theme_counts = school_meta['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    fig_theme = px.bar(theme_counts, x='Theme', y='Count', title=f"Theme Distribution – {school_name}", color='Theme', color_discrete_sequence=[USTP_GOLD, DEPED_RED, USTP_DARK_BLUE])
    status_counts = school_meta['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig_status = px.bar(status_counts, x='Status', y='Count', title=f"Publication Status – {school_name}", color='Status', color_discrete_sequence=[USTP_DARK_BLUE, USTP_GOLD, DEPED_MAROON])
    utilised = school_meta['utilized_by_school'].sum() if 'utilized_by_school' in school_meta.columns else 0
    total = len(school_meta)
    util_rate = (utilised / total * 100) if total > 0 else 0
    st.metric("Research Utilisation Rate", f"{util_rate:.1f}%")
    teacher_counts = school_meta['teacher_name'].value_counts().reset_index().head(10)
    teacher_counts.columns = ['Teacher', 'Number of Outputs']
    fig_teacher = px.bar(teacher_counts, x='Number of Outputs', y='Teacher', orientation='h', title=f"Teacher Productivity (Top 10) – {school_name}", color='Number of Outputs', color_continuous_scale=['#F5A623', '#0D2B5E'])
    st.plotly_chart(fig_theme, use_container_width=True)
    st.plotly_chart(fig_status, use_container_width=True)
    st.plotly_chart(fig_teacher, use_container_width=True)

def cycle_research_correlation(agent, metadata_df, school_id):
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
    fig.add_trace(go.Scatter(
        x=[c.cycle_number for c in agent.cycle_improvements],
        y=cumulative_outputs,
        mode='markers+lines',
        marker=dict(size=10, color=USTP_GOLD),
        line=dict(color=USTP_DARK_BLUE),
        name='Research outputs'
    ))
    fig.update_layout(
        title="Cycle vs Cumulative Research Outputs",
        xaxis_title="Cycle Number",
        yaxis_title="Number of Research Outputs (cumulative)",
        showlegend=False,
        font=dict(color=USTP_DARK_BLUE)
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Streamlit UI (colored)
# ------------------------------------------------------------
st.set_page_config(page_title="Research Culture Digital Twin", layout="wide")
st.markdown(f"<h1 style='text-align: center; color: {USTP_DARK_BLUE};'>7‑Milestone Research Culture Digital Twin</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"<h2 style='color: {USTP_DARK_BLUE};'>Policy Levers & Simulation Controls</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        u_train = st.slider("Training freq.", 0.0, 1.0, 0.5, 0.05)
        u_mentor = st.slider("Mentorship ratio", 0.0, 1.0, 0.5, 0.05)
        u_budget = st.slider("Support budget", 0.0, 1.0, 0.5, 0.05)
    with col2:
        u_lead = st.slider("Leadership commit.", 0.0, 1.0, 0.5, 0.05)
        u_collab = st.slider("Collaboration freq.", 0.0, 1.0, 0.5, 0.05)
    
    levers = {
        'u_train': u_train,
        'u_mentor': u_mentor,
        'u_budget': u_budget,
        'u_lead': u_lead,
        'u_collab': u_collab
    }
    
    max_schools_allowed = 200
    num_schools = st.number_input("Number of schools", min_value=1, max_value=max_schools_allowed, value=20, step=1)
    duration = st.selectbox("Run duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=9)
    random_events = st.checkbox("Enable random events", value=False)
    use_survey = st.checkbox("Override with survey data", value=True)
    
    col_buttons = st.columns(3)
    with col_buttons[0]:
        run_btn = st.button("Run", use_container_width=True)
    with col_buttons[1]:
        step_btn = st.button("Step (1 month)", use_container_width=True)
    with col_buttons[2]:
        reset_btn = st.button("Reset", use_container_width=True)
    
    export_btn = st.button("Export results (CSV)", use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Data Upload</h3>", unsafe_allow_html=True)
    survey_file = st.file_uploader("Upload quarterly survey (CSV)", type=["csv"], key="survey")
    metadata_file = st.file_uploader("Upload research metadata (CSV)", type=["csv"], key="metadata")

# Main area
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
            st.success(f"Loaded {len(school_info)} schools.")
            
            if 'sim' not in st.session_state:
                st.session_state.sim = Simulation(num_schools=num_schools, random_events=random_events)
                st.session_state.current_month = 0
                st.session_state.total_months = 0
                school_ids = school_info['school_id_no'].head(num_schools).tolist()
                st.session_state.history = {sid: {'R':[],'A':[],'C':[],'S':[],'I':[],'P':[],'M':[],'month':[],'milestone':[],'running_outcome':[]}
                                            for sid in school_ids}
                for idx, agent in enumerate(st.session_state.sim.agents):
                    agent.real_id = school_ids[idx]
                for agent in st.session_state.sim.agents:
                    school_metadata = metadata_df[metadata_df['school_id_no'] == agent.real_id]
                    agent.A = min(1.0, agent.A + len(school_metadata[school_metadata['document_type']=='abstract'])*0.01)
                    agent.M = min(1.0, agent.M + len(school_metadata[school_metadata['status']=='published'])*0.02)
                    agent.C = min(1.0, agent.C + len(school_metadata[school_metadata['document_type']=='full_paper'])*0.005)
                    agent.P = min(1.0, agent.P + school_metadata['theme'].nunique()*0.01)
            
            school_ids = school_info['school_id_no'].head(num_schools).tolist()
            school_options = [f"ID {sid}: {school_info[school_info['school_id_no']==sid]['school_name'].values[0]}" for sid in school_ids]
            selected_school_label = st.selectbox("Select school", school_options, index=0)
            selected_school_id = int(selected_school_label.split(":")[0].split()[1])
            selected_school_name = school_info[school_info['school_id_no']==selected_school_id]['school_name'].values[0]
            
            st.markdown("### Research Outputs (Recent)")
            df_show = metadata_df[metadata_df['school_id_no'] == selected_school_id].copy()
            if not df_show.empty:
                df_show_sorted = df_show.sort_values('upload_date', ascending=False)
                st.dataframe(df_show_sorted[['teacher_name', 'year_undertaken', 'title', 'theme', 'status', 'utilized_by_school']].head(10))
            else:
                st.info("No research outputs for this school.")
            
            # Simulation actions
            if run_btn:
                st.session_state.sim = Simulation(num_schools=num_schools, random_events=random_events)
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
                st.session_state.history = {sid: {'R':[],'A':[],'C':[],'S':[],'I':[],'P':[],'M':[],'month':[],'milestone':[],'running_outcome':[]}
                                            for sid in school_ids}
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
                st.session_state.sim = Simulation(num_schools=num_schools, random_events=random_events)
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
                st.session_state.history = {sid: {'R':[],'A':[],'C':[],'S':[],'I':[],'P':[],'M':[],'month':[],'milestone':[],'running_outcome':[]}
                                            for sid in school_ids}
                st.rerun()
            
            if st.session_state.total_months > 0:
                hist = st.session_state.history.get(selected_school_id, None)
                agent = next((a for a in st.session_state.sim.agents if a.real_id == selected_school_id), None)
                if hist and agent:
                    # Main plots with colors
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
                    fig1.update_layout(height=800, showlegend=True, font=dict(color=USTP_DARK_BLUE))
                    fig1.update_xaxes(title_text="Month", row=1, col=1)
                    fig1.update_yaxes(title_text="Value (0-1)", row=1, col=1)
                    fig1.update_xaxes(title_text="Month", row=1, col=2)
                    fig1.update_yaxes(title_text="Milestone", row=1, col=2)
                    fig1.update_xaxes(title_text="Month", row=2, col=1)
                    fig1.update_yaxes(title_text="RCSI", row=2, col=1)
                    fig1.update_xaxes(title_text="Cycle Number", row=2, col=2)
                    fig1.update_yaxes(title_text="RCSI", row=2, col=2)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    latest = get_latest_survey(survey_df, selected_school_id)
                    if latest is not None:
                        radar = radar_chart(latest, selected_school_name)
                        st.plotly_chart(radar, use_container_width=True)
                    else:
                        st.info("No survey data for current quarter.")
                    
                    with st.expander("📚 Research Outputs Dashboard (for selected school)"):
                        research_outputs_dashboard(metadata_df, selected_school_id, selected_school_name)
                    
                    with st.expander("🔄 Cycle vs Research Outputs"):
                        cycle_research_correlation(agent, metadata_df, selected_school_id)
                    
                    st.markdown(f"### 📈 Research Culture Sustainability Index (RCSI) Interpretation Table")
                    outcome_table_html = f"""
                    <table style="width:100%; border-collapse: collapse; margin-bottom: 20px; border: 1px solid {USTP_DARK_BLUE};">
                    <tr style="background-color: {USTP_DARK_BLUE}; color: white;">
                        <th>RCSI Range</th><th>Level</th><th>Description</th>
                    </tr>
                    <tr><td>0.0 – 0.2</td><td>Very Low</td><td>Little to no accumulated research culture strength.</td></tr>
                    <tr><td>0.2 – 0.4</td><td>Low</td><td>Minimal ecosystem vitality; research culture still weak.</td></tr>
                    <tr><td>0.4 – 0.6</td><td>Moderate</td><td>Noticeable strength; research culture developing.</td></tr>
                    <tr><td>0.6 – 0.8</td><td>High</td><td>Strong ecosystem; research culture becoming sustainable.</td></tr>
                    <tr><td>0.8 – 1.0</td><td>Very High</td><td>Excellent vitality; research culture fully embedded.</td></tr>
                    </table>
                    """
                    st.markdown(outcome_table_html, unsafe_allow_html=True)
                    
                    rcsi_val = agent.running_total_outcome
                    intervals = [(0.0,0.2,"Very Low"), (0.2,0.4,"Low"), (0.4,0.6,"Moderate"), (0.6,0.8,"High"), (0.8,1.0,"Very High")]
                    level = "Exceptional"
                    for low,high,lev in intervals:
                        if low <= rcsi_val < high:
                            level = lev
                            break
                    if agent.cycle_count >= 2:
                        sustainability = "The school has reached a self‑sustaining research culture (multiple cycles)."
                    elif agent.cycle_count == 1:
                        sustainability = "The school has completed one full cycle, showing initial sustainability."
                    elif agent.current_milestone >= 4:
                        sustainability = "The school is approaching sustainability but has not yet completed a full cycle."
                    else:
                        sustainability = "The school is still in early stages of research culture development."
                    
                    per_school_html = f"""
                    <div style="background-color: #E3F2FD; border-left: 5px solid {USTP_DARK_BLUE}; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <b>📌 School {selected_school_id} ({selected_school_name}) Synopsis:</b><br>
                    After {st.session_state.total_months} months: Milestone = {agent.current_milestone} | Completed cycles = {agent.cycle_count}<br>
                    Research Culture Sustainability Index (RCSI) = <b>{rcsi_val:.3f}</b> → <b>{level}</b> level.<br>
                    <i>Research Sustainability Culture:</i> {sustainability}
                    </div>
                    """
                    st.markdown(per_school_html, unsafe_allow_html=True)
                    
                    total_schools = len(st.session_state.sim.agents)
                    early_count = sum(1 for a in st.session_state.sim.agents if a.current_milestone <= 2 or a.cycle_count == 0)
                    advanced_count = sum(1 for a in st.session_state.sim.agents if a.current_milestone >= 4 or a.cycle_count >= 1)
                    early_percent = (early_count / total_schools) * 100
                    advanced_percent = (advanced_count / total_schools) * 100
                    total_outcome = sum(a.running_total_outcome for a in st.session_state.sim.agents)
                    avg_rcsi = total_outcome / total_schools
                    level_avg = "Exceptional"
                    for low,high,lev in intervals:
                        if low <= avg_rcsi < high:
                            level_avg = lev
                            break
                    total_cycles = sum(a.cycle_count for a in st.session_state.sim.agents)
                    avg_milestone = np.mean([a.current_milestone for a in st.session_state.sim.agents])
                    
                    division_html = f"""
                    <div style="background-color: #E8F5E9; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <b>🏢 Division‑Level Synopsis (all {total_schools} schools):</b><br>
                    Average milestone = {avg_milestone:.1f} | Total completed cycles across all schools = {total_cycles}<br>
                    Average RCSI = {avg_rcsi:.3f} → <b>{level_avg}</b> level.<br>
                    <i>Stage distribution:</i> {early_percent:.1f}% of schools are in early stages (milestone ≤2 or no cycle).<br>
                    {advanced_percent:.1f}% have reached advanced stages (milestone ≥4 or at least one cycle).<br>
                    <i>Division‑wide sustainability:</i> {
                        "The division is showing strong research culture with multiple cycles and high impact." if total_cycles > total_schools else
                        "The division has a moderate research culture; policy adjustments may accelerate progress." if avg_milestone >= 4 else
                        "Most schools are still in early stages of research culture development."
                    }
                    </div>
                    """
                    st.markdown(division_html, unsafe_allow_html=True)
                    
                    with st.expander("📊 Graph Interpretations"):
                        st.markdown(f"""
                        - **Variable Evolution:** Shows how R, A, C, S, I, P, M change over time. Higher values (closer to 1) mean stronger readiness, awareness, capacity, etc.
                        - **Milestone Progress:** The school moves through milestones 0–6. Reaching milestone 6 and cycling back indicates a full sustainable cycle.
                        - **Research Culture Sustainability Index (RCSI):** Cumulative strength of the research ecosystem, derived from Impact Realization (M) and Collaboration (P).
                        - **Improvement per Completed Cycle:** Each bar shows the RCSI contributed by one cycle. Higher bars in later cycles indicate increasing effectiveness.
                        - **Radar Chart:** Current snapshot of the seven variables – the ideal is a balanced, high‑value shape.
                        - **Research Outputs Dashboard:** Tracks themes, publication status, utilisation, and teacher productivity.
                        - **Cycle vs Research Outputs:** Shows how research output accumulation relates to cycle progression.
                        """, unsafe_allow_html=True)
            
            if export_btn:
                all_data = []
                for agent in st.session_state.sim.agents:
                    h = st.session_state.history[agent.real_id]
                    for t in range(len(h['month'])):
                        row = {'school_id': agent.real_id, 'month': h['month'][t], 'milestone': h['milestone'][t],
                               'running_outcome': h['running_outcome'][t]}
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
                csv1 = df_hist.to_csv(index=False).encode('utf-8')
                csv2 = df_cycles.to_csv(index=False).encode('utf-8')
                st.download_button("Download simulation history", csv1, "simulation_history.csv", "text/csv")
                st.download_button("Download cycle improvements", csv2, "cycle_improvements.csv", "text/csv")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload quarterly survey and research metadata CSV files to begin.")
