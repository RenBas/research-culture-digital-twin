import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict

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
# Data processing functions (corrected)
# ------------------------------------------------------------
def process_survey(survey_df):
    """Returns (survey_df, school_info, error_message)"""
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
    """Returns (metadata_df, error_message)"""
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

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Research Culture Digital Twin", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>7‑Milestone Research Culture Digital Twin</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Policy Levers & Simulation Controls")
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
    
    # Maximum schools set to 200
    max_schools_allowed = 200
    num_schools = st.number_input("Number of schools", min_value=1, max_value=max_schools_allowed, value=20, step=1)
    duration = st.selectbox("Run duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=9)
    random_events = st.checkbox("Enable random events", value=False)
    use_survey = st.checkbox("Override with survey data", value=True)
    
    col_buttons = st.columns(3)
    with col_buttons[0]:
        run_btn = st.button("Run", use_container_width=True, type="primary")
    with col_buttons[1]:
        step_btn = st.button("Step (1 month)", use_container_width=True)
    with col_buttons[2]:
        reset_btn = st.button("Reset", use_container_width=True)
    
    export_btn = st.button("Export results (CSV)", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Data Upload")
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
            
            # Initialize session state
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
            
            # School selector
            school_ids = school_info['school_id_no'].head(num_schools).tolist()
            school_options = [f"ID {sid}: {school_info[school_info['school_id_no']==sid]['school_name'].values[0]}" for sid in school_ids]
            selected_school_label = st.selectbox("Select school", school_options, index=0)
            selected_school_id = int(selected_school_label.split(":")[0].split()[1])
            
            # Research outputs table
            st.markdown("### Research Outputs")
            df_show = metadata_df[metadata_df['school_id_no'] == selected_school_id].copy()
            if not df_show.empty:
                df_show_sorted = df_show.sort_values('upload_date')
                df_show_sorted['cumulative_by_teacher'] = df_show_sorted.groupby('teacher_name').cumcount() + 1
                st.dataframe(df_show_sorted[['teacher_name', 'year_undertaken', 'title', 'theme', 'status', 'cumulative_by_teacher']])
            else:
                st.info("No research outputs for this school.")
            
            # Simulation actions
            if run_btn:
                # Reset and run
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
                # Reinitialize
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
            
            # Display plots if history exists
            if st.session_state.total_months > 0:
                hist = st.session_state.history.get(selected_school_id, None)
                agent = next((a for a in st.session_state.sim.agents if a.real_id == selected_school_id), None)
                if hist and agent:
                    # Create Plotly subplots
                    fig = make_subplots(rows=2, cols=2, subplot_titles=("Variable Evolution", "Milestone Progress", "Student Learning Outcome (Running Total)", "Improvement per Completed Cycle"))
                    colors = ['#1E88E5', '#FFB74D', '#8E44AD', '#2ECC71', '#E67E22', '#E74C3C', '#1ABC9C']
                    vars_ = ['R','A','C','S','I','P','M']
                    for i, var in enumerate(vars_):
                        fig.add_trace(go.Scatter(x=hist['month'], y=hist[var], mode='lines', name=var, line=dict(color=colors[i])), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'], mode='lines', name='Milestone', line=dict(color='#D32F2F')), row=1, col=2)
                    fig.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'], mode='lines', name='Outcome', line=dict(color='#2E7D32')), row=2, col=1)
                    if agent.cycle_improvements:
                        cycles = [c.cycle_number for c in agent.cycle_improvements]
                        improvements = [c.total_improvement for c in agent.cycle_improvements]
                        fig.add_trace(go.Bar(x=cycles, y=improvements, name='Improvement', marker_color='#F39C12'), row=2, col=2)
                    else:
                        fig.add_annotation(text="No cycles completed yet", xref="x2 domain", yref="y2 domain", x=0.5, y=0.5, showarrow=False, row=2, col=2)
                    fig.update_layout(height=800, showlegend=True)
                    fig.update_xaxes(title_text="Month", row=1, col=1)
                    fig.update_yaxes(title_text="Value (0-1)", row=1, col=1)
                    fig.update_xaxes(title_text="Month", row=1, col=2)
                    fig.update_yaxes(title_text="Milestone", row=1, col=2)
                    fig.update_xaxes(title_text="Month", row=2, col=1)
                    fig.update_yaxes(title_text="Cumulative Improvement", row=2, col=1)
                    fig.update_xaxes(title_text="Cycle Number", row=2, col=2)
                    fig.update_yaxes(title_text="Improvement", row=2, col=2)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ---------------------------
                    # Interpretation Table for Cumulative Student Outcome
                    # ---------------------------
                    st.markdown("### 📈 Cumulative Student Outcome Interpretation Table")
                    outcome_table_html = """
                    <table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr style="background-color: #ddd;">
                        <th>Range</th><th>Level</th><th>Description</th>
                    </tr>
                    <tr><td>0.0 – 0.2</td><td>Very Low</td><td>Little to no improvement in student learning outcomes.</td></tr>
                    <tr><td>0.2 – 0.4</td><td>Low</td><td>Minimal improvement; research culture still weak.</td></tr>
                    <tr><td>0.4 – 0.6</td><td>Moderate</td><td>Noticeable improvement; research culture developing.</td></tr>
                    <tr><td>0.6 – 0.8</td><td>High</td><td>Strong improvement; research culture becoming sustainable.</td></tr>
                    <tr><td>0.8 – 1.0</td><td>Very High</td><td>Excellent improvement; research culture fully embedded and impactful.</td></tr>
                    </table>
                    """
                    st.markdown(outcome_table_html, unsafe_allow_html=True)
                    
                    # Synopses using the same intervals
                    outcome_val = agent.running_total_outcome
                    intervals = [(0.0,0.2,"Very Low"), (0.2,0.4,"Low"), (0.4,0.6,"Moderate"), (0.6,0.8,"High"), (0.8,1.0,"Very High")]
                    level = "Exceptional"
                    for low,high,lev in intervals:
                        if low <= outcome_val < high:
                            level = lev
                            break
                    st.markdown(f"""
                    <div style="background-color: #E3F2FD; border-left: 5px solid #1E88E5; padding: 10px; border-radius: 5px;">
                    <b>📌 School {selected_school_id} Synopsis:</b><br>
                    After {st.session_state.total_months} months: Milestone = {agent.current_milestone} | Completed cycles = {agent.cycle_count}<br>
                    Cumulative student outcome improvement = <b>{outcome_val:.3f}</b> → <b>{level}</b> level.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    total_outcome = sum(a.running_total_outcome for a in st.session_state.sim.agents)
                    avg_outcome = total_outcome / len(st.session_state.sim.agents)
                    level_avg = "Exceptional"
                    for low,high,lev in intervals:
                        if low <= avg_outcome < high:
                            level_avg = lev
                            break
                    total_cycles = sum(a.cycle_count for a in st.session_state.sim.agents)
                    avg_milestone = np.mean([a.current_milestone for a in st.session_state.sim.agents])
                    st.markdown(f"""
                    <div style="background-color: #E8F5E9; border-left: 5px solid #2E7D32; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <b>🏢 Division-Level Synopsis:</b><br>
                    Average milestone = {avg_milestone:.1f} | Total completed cycles = {total_cycles}<br>
                    Average cumulative outcome = {avg_outcome:.3f} → <b>{level_avg}</b> level.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📊 Graph Interpretations"):
                        st.markdown("""
                        - **Variable Evolution:** Shows how R, A, C, S, I, P, M change over time. Higher values (closer to 1) mean stronger readiness, awareness, capacity, etc.
                        - **Milestone Progress:** The school moves through milestones 0–6. Reaching milestone 6 and cycling back indicates a full sustainable cycle.
                        - **Student Learning Outcome (Running Total):** Cumulative improvement in learner outcomes.
                        - **Improvement per Completed Cycle:** Each bar shows the improvement contributed by one cycle. Higher bars in later cycles indicate increasing effectiveness.
                        """)
            
            if export_btn:
                # Prepare dataframes
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
