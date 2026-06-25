# ============================================================
# Phase 1 Enhanced Digital Twin - CDO Research Culture Framework
# Refactored: bug-fixes, deduplication, performance, UX improvements
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import math
import base64

# ============================================================
# CONSTANTS
# ============================================================

# --- Colour Palette ---
USTP_DARK_BLUE = "#0D2B5E"
USTP_GOLD = "#F5A623"
DEPED_RED = "#D32F2F"
DEPED_MAROON = "#8B0000"
LIGHT_BG = "#F8F9FA"
DARK_BG = "#1E1E1E"
DARK_TEXT = "#FFFFFF"
LIGHT_TEXT = "#000000"

# --- Simulation Model Constants ---
BASE_YEAR = 2026
RANDOM_EVENT_PROB = 0.00417          # ~1 event per 240 months
LOSS_CHAMPION_PENALTY = 0.10
FUNDING_BOOST = 0.15
LEADERSHIP_PENALTY = 0.20
CYCLE_R_BONUS = 0.10
CYCLE_M_DECAY_FACTOR = 0.50
CYCLE_M_MIN_AFTER_DECAY = 0.20
CYCLE_BONUS_BASE = 0.03
CYCLE_BONUS_M_SCALE = 0.03
MONTHLY_OUTCOME_BASE = 0.001
MILESTONE_MIN_MONTHS = 6
VALUE_FLOOR = 0.1
VALUE_CEIL = 1.0

# --- Variable names ---
VARIABLES = ['R', 'A', 'C', 'S', 'I', 'P', 'M']

# --- Milestone definitions ---
MILESTONE_NAMES = {
    0: "Milestone 0 (Readiness and Relevance)",
    1: "Milestone 1 (Awareness to Action)",
    2: "Milestone 2 (Capacity Spark)",
    3: "Milestone 3 (Structured Support)",
    4: "Milestone 4 (Institutional Anchoring)",
    5: "Milestone 5 (Community of Practice)",
    6: "Milestone 6 (Impact Realization)",
}

MILESTONE_SHORT = {k: f"M{k}" for k in range(7)}

VAR_FULL_NAMES = {
    'R': 'Readiness (R)',
    'A': 'Awareness (A)',
    'C': 'Capacity (C)',
    'S': 'Structured Support (S)',
    'I': 'Institutional Anchoring (I)',
    'P': 'Community of Practice (P)',
    'M': 'Impact Realization (M)',
}

# --- Milestone transition thresholds ---
MILESTONE_THRESHOLDS = {
    0: ('A', 0.8, 1),
    1: ('C', 0.7, 2),
    2: ('S', 0.7, 3),
    3: ('I', 0.8, 4),
    4: ('P', 0.8, 5),
    5: ('M', 0.7, 6),
}

# --- RCSI Classification ---
RCSI_LEVELS = [
    (0.0, 0.2, "Very Low"),
    (0.2, 0.4, "Low"),
    (0.4, 0.6, "Moderate"),
    (0.6, 0.8, "High"),
    (0.8, 1.0, "Very High"),
]

# --- Data processing ---
REQUIRED_SURVEY_COLS = ['month', 'school_id_no'] + VARIABLES
OPTIONAL_SURVEY_COLS = ['school_name']

REQUIRED_META_COLS = ['upload_date', 'teacher_name', 'school_id_no']
OPTIONAL_META_COLS = {
    'document_type': 'abstract',
    'title': '',
    'theme': 'Uncategorized',
    'status': 'unpublished',
    'publication_link': '',
    'utilized_by_school': False,
    'utilization_date': '',
    'year_undertaken': 2025,
    'years_of_service': None,
    'teacher_rank': None,
    'educational_attainment': None,
}

# --- Plot colours for the 7 variables ---
VAR_COLORS = ['#1E88E5', USTP_GOLD, '#8E44AD', '#2ECC71', '#E67E22', DEPED_RED, '#1ABC9C']


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def classify_rcsi(value: float) -> str:
    """Classify an RCSI value into a level string."""
    for low, high, lev in RCSI_LEVELS:
        if low <= value < high:
            return lev
    return "Very High"


def interpret_avg_milestone(avg_milestone: float) -> str:
    """Return a human-readable interpretation of the average milestone."""
    milestone_desc = [
        (0.5, "between Milestone 0 (Readiness and Relevance) and Milestone 1 (Awareness to Action), approaching M1"),
        (1.5, "between Milestone 1 (Awareness to Action) and Milestone 2 (Capacity Spark)"),
        (2.5, "between Milestone 2 (Capacity Spark) and Milestone 3 (Structured Support)"),
        (3.5, "between Milestone 3 (Structured Support) and Milestone 4 (Institutional Anchoring)"),
        (4.5, "between Milestone 4 (Institutional Anchoring) and Milestone 5 (Community of Practice)"),
        (5.5, "between Milestone 5 (Community of Practice) and Milestone 6 (Impact Realization)"),
        (float('inf'), "at or beyond Milestone 6 (Impact Realization)"),
    ]
    for threshold, desc in milestone_desc:
        if avg_milestone < threshold:
            return f"{avg_milestone:.1f} -> {desc}"
    return f"{avg_milestone:.1f} -> at or beyond Milestone 6 (Impact Realization)"


def interpret_utilisation_rate(rate: float) -> Tuple[str, str]:
    """Return (level, description) for a utilisation rate percentage."""
    if rate < 20:
        return "Very Low", "Research is rarely adopted into practice; significant gap between production and use."
    elif rate < 40:
        return "Low", "Limited adoption; most research outputs are not utilised."
    elif rate < 60:
        return "Moderate", "Roughly half of research outputs are adopted; room for improvement."
    elif rate < 80:
        return "High", "Strong translation of research into practice; research is valued."
    else:
        return "Very High", "Excellent utilisation; research is consistently applied to improve practice."


def month_str_to_num(month_str: Any) -> int:
    """Convert a 'YYYY-MM' string to a month number relative to BASE_YEAR."""
    try:
        parts = str(month_str).strip().split('-')
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            return (year - BASE_YEAR) * 12 + month
    except (ValueError, TypeError, AttributeError):
        pass
    return 0


def date_to_month_num(d: Any) -> int:
    """Convert a pandas Timestamp / datetime to month number relative to BASE_YEAR."""
    try:
        return (d.year - BASE_YEAR) * 12 + d.month
    except (ValueError, TypeError, AttributeError):
        return 0


# ============================================================
# THEME
# ============================================================

def apply_theme(dark_mode: bool) -> None:
    """Apply light or dark mode CSS to the Streamlit app."""
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


# ============================================================
# CHART DOWNLOAD HELPER
# ============================================================

def get_figure_download_link(fig, filename: str = "chart.html",
                             link_text: str = "Download chart (interactive HTML)") -> None:
    """Generate a download link for an interactive Plotly figure as a self-contained HTML file."""
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    b64 = base64.b64encode(html_str.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CycleRecord:
    cycle_number: int
    total_improvement: float
    completion_month: int


class SchoolAgent:
    """Represents a single school in the simulation with 7 milestone-linked variables."""

    def __init__(self, unique_id: int,
                 initial_R: float = 0.3, initial_A: float = 0.2,
                 initial_C: float = 0.2, initial_S: float = 0.1,
                 initial_I: float = 0.1, initial_P: float = 0.1,
                 initial_M: float = 0.0,
                 random_events_enabled: bool = False):
        self.id = unique_id
        self.real_id: int = unique_id
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
        self.random_events_enabled = random_events_enabled
        self.model_time = 0
        self._rng = np.random.RandomState()

    def apply_random_event(self) -> None:
        if not self.random_events_enabled:
            return
        if self._rng.rand() < RANDOM_EVENT_PROB:
            event_type = self._rng.choice(["loss_champion", "funding", "leadership_change"])
            if event_type == "loss_champion":
                for var in VARIABLES:
                    setattr(self, var, max(VALUE_FLOOR, getattr(self, var) - LOSS_CHAMPION_PENALTY))
            elif event_type == "funding":
                self.S = min(VALUE_CEIL, self.S + FUNDING_BOOST)
            elif event_type == "leadership_change":
                self.I = max(VALUE_FLOOR, self.I - LEADERSHIP_PENALTY)

    def _update_milestone(self) -> None:
        self.months_in_milestone += 1
        next_milestone = self.current_milestone

        if self.current_milestone in MILESTONE_THRESHOLDS:
            var_name, threshold, target = MILESTONE_THRESHOLDS[self.current_milestone]
            if getattr(self, var_name) >= threshold:
                next_milestone = target
        elif self.current_milestone == 6:
            if self.M >= 0.9 and self.R >= 0.8:
                self._complete_cycle()
                next_milestone = 0

        if next_milestone != self.current_milestone and self.months_in_milestone >= MILESTONE_MIN_MONTHS:
            self.current_milestone = next_milestone
            self.months_in_milestone = 0

    def _complete_cycle(self) -> None:
        old_M = self.M
        self.R = min(VALUE_CEIL, self.R + CYCLE_R_BONUS)
        self.M = max(CYCLE_M_MIN_AFTER_DECAY, self.M * CYCLE_M_DECAY_FACTOR)
        bonus = CYCLE_BONUS_BASE + CYCLE_BONUS_M_SCALE * old_M
        self.running_total_outcome += bonus
        self.current_cycle_accumulator += bonus
        self.cycle_count += 1
        self.cycle_improvements.append(
            CycleRecord(
                cycle_number=self.cycle_count,
                total_improvement=self.current_cycle_accumulator,
                completion_month=self.model_time,
            )
        )
        self.current_cycle_accumulator = 0.0


# ============================================================
# SIMULATION ENGINE
# ============================================================

class Simulation:
    """Manages a population of SchoolAgents and runs the monthly step."""

    def __init__(self, num_schools: int = 1, random_events: bool = False):
        self.agents: List[SchoolAgent] = [
            SchoolAgent(i, random_events_enabled=random_events) for i in range(num_schools)
        ]

    def step(self, levers: Dict[str, float], month: int) -> None:
        """Advance all agents by one month (vectorized math + per-agent bookkeeping)."""
        u_train = levers['u_train']
        u_mentor = levers['u_mentor']
        u_budget = levers['u_budget']
        u_lead = levers['u_lead']
        u_collab = levers['u_collab']

        R = np.array([a.R for a in self.agents])
        A = np.array([a.A for a in self.agents])
        C = np.array([a.C for a in self.agents])
        S = np.array([a.S for a in self.agents])
        I = np.array([a.I for a in self.agents])
        P = np.array([a.P for a in self.agents])
        M = np.array([a.M for a in self.agents])

        u_lead_eff = np.minimum(1.0, u_lead + 0.05 * M)

        R_new = np.clip(R + 0.02 * M - 0.01 * (1 - u_lead_eff), VALUE_FLOOR, VALUE_CEIL)
        A_new = np.clip(A + 0.04 * R + 0.02 * u_train + 0.01 * M - 0.005, VALUE_FLOOR, VALUE_CEIL)
        C_new = np.clip(C + 0.03 * u_train + 0.02 * u_mentor - 0.01, VALUE_FLOOR, VALUE_CEIL)
        S_new = np.clip(S + 0.04 * u_budget + 0.02 * u_mentor - 0.01 * (1 - u_lead_eff), VALUE_FLOOR, VALUE_CEIL)
        I_new = np.clip(I + 0.03 * u_lead_eff + 0.02 * S - 0.005, VALUE_FLOOR, VALUE_CEIL)
        P_new = np.clip(P + 0.04 * u_collab + 0.02 * I - 0.01, VALUE_FLOOR, VALUE_CEIL)
        M_new = np.clip(M + 0.02 * C + 0.02 * P - 0.005, VALUE_FLOOR, VALUE_CEIL)

        for i, agent in enumerate(self.agents):
            agent.R, agent.A, agent.C = float(R_new[i]), float(A_new[i]), float(C_new[i])
            agent.S, agent.I, agent.P = float(S_new[i]), float(I_new[i]), float(P_new[i])
            agent.M = float(M_new[i])

            monthly_gain = MONTHLY_OUTCOME_BASE * agent.M * (1 + agent.P)
            agent.running_total_outcome += monthly_gain
            agent.current_cycle_accumulator += monthly_gain
            agent.model_time = month
            agent._update_milestone()
            agent.apply_random_event()

    def get_agent(self, idx: int = 0) -> SchoolAgent:
        return self.agents[idx]


# ============================================================
# SIMULATION HELPERS (eliminates duplicated init/record logic)
# ============================================================

def create_empty_history(school_ids: List[int]) -> Dict[int, Dict[str, list]]:
    """Create an empty history dictionary for all schools."""
    return {
        sid: {var: [] for var in VARIABLES + ['month', 'milestone', 'running_outcome']}
        for sid in school_ids
    }


def init_simulation_with_data(school_ids: List[int], metadata_df: pd.DataFrame,
                               random_events: bool) -> Simulation:
    """Create and seed a Simulation from school IDs and metadata."""
    sim = Simulation(num_schools=len(school_ids), random_events=random_events)
    seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
    return sim


def seed_agents_from_metadata(agents: List[SchoolAgent], school_ids: List[int],
                                metadata_df: pd.DataFrame) -> None:
    """Apply metadata-based boosts to newly created agents."""
    for idx, agent in enumerate(agents):
        agent.real_id = school_ids[idx]
        sm = metadata_df[metadata_df['school_id_no'] == agent.real_id]
        if sm.empty:
            continue
        agent.A = min(VALUE_CEIL, agent.A + len(sm[sm['document_type'] == 'abstract']) * 0.01)
        agent.M = min(VALUE_CEIL, agent.M + len(sm[sm['status'] == 'published']) * 0.02)
        agent.C = min(VALUE_CEIL, agent.C + len(sm[sm['document_type'] == 'full_paper']) * 0.005)
        agent.P = min(VALUE_CEIL, agent.P + sm['theme'].nunique() * 0.01)


def record_history(history: Dict[int, Dict[str, list]], agents: List[SchoolAgent],
                   total_months: int) -> None:
    """Snapshot all agent states into the history dict."""
    for agent in agents:
        h = history[agent.real_id]
        h['month'].append(total_months)
        for var in VARIABLES:
            h[var].append(getattr(agent, var))
        h['milestone'].append(agent.current_milestone)
        h['running_outcome'].append(agent.running_total_outcome)


def apply_survey_override(agents: List[SchoolAgent], survey_df: pd.DataFrame,
                           target_month: int) -> None:
    """Override agent variables with real survey data if available for the target month."""
    for agent in agents:
        row = survey_df[
            (survey_df['school_id_no'] == agent.real_id) &
            (survey_df['month_num'] == target_month)
        ]
        if not row.empty:
            r = row.iloc[0]
            for var in VARIABLES:
                setattr(agent, var, r[var])


# ============================================================
# DATA PROCESSING (cached)
# ============================================================

@st.cache_data(show_spinner="Processing survey data...")
def process_survey(_survey_df: pd.DataFrame):
    """Validate and process survey CSV. Returns (valid_df, school_info, error_msg)."""
    if _survey_df is None:
        return None, None, "No survey file uploaded."
    try:
        df = _survey_df.copy()

        missing_req = [col for col in REQUIRED_SURVEY_COLS if col not in df.columns]
        if missing_req:
            return None, None, f"Missing required survey columns: {', '.join(missing_req)}"

        if 'school_id_no' in df.columns:
            df['school_id_no'] = df['school_id_no'].astype(int)
        elif 'school_id' in df.columns:
            df['school_id_no'] = df['school_id'].astype(str).apply(
                lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x)
            )
        else:
            return None, None, "Survey file must contain 'school_id_no' or 'school_id' column."

        if 'school_name' not in df.columns:
            df['school_name'] = df['school_id_no'].apply(lambda x: f"School_{x}")
        else:
            df['school_name'] = df['school_name'].fillna(
                df['school_id_no'].apply(lambda x: f"School_{x}")
            )

        df['month_num'] = df['month'].apply(month_str_to_num)

        for v in VARIABLES:
            if not pd.api.types.is_numeric_dtype(df[v]):
                df[v] = pd.to_numeric(df[v], errors='coerce')
                if df[v].isna().any():
                    return None, None, f"Column {v} must be numeric."
            if df[v].isna().any():
                return None, None, f"Column {v} contains missing values."
            if (df[v] < 0).any() or (df[v] > 1).any():
                return None, None, f"Column {v} values must be between 0 and 1."

        school_info = df[['school_id_no', 'school_name']].drop_duplicates().sort_values('school_id_no')
        return df, school_info, None

    except Exception as e:
        return None, None, f"Error processing survey: {str(e)}"


@st.cache_data(show_spinner="Processing metadata...")
def process_metadata(_metadata_df: pd.DataFrame):
    """Validate and process metadata CSV. Returns (valid_df, error_msg)."""
    if _metadata_df is None:
        return None, "No metadata file uploaded."
    try:
        df = _metadata_df.copy()

        missing_req = [col for col in REQUIRED_META_COLS if col not in df.columns]
        if missing_req:
            return None, f"Missing required metadata columns: {', '.join(missing_req)}"

        if 'school_id_no' not in df.columns:
            if 'school' in df.columns:
                df['school_id_no'] = df['school'].astype(str).apply(
                    lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x)
                )
            else:
                return None, "Metadata must have 'school_id_no' or 'school' column."
        df['school_id_no'] = df['school_id_no'].astype(int)

        for col, default in OPTIONAL_META_COLS.items():
            if col not in df.columns:
                df[col] = default
                st.warning(f"Optional column '{col}' missing in metadata. Using default: {default}")
            else:
                df[col] = df[col].fillna(default)

        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
        if df['upload_date'].isna().any():
            return None, "Invalid dates in 'upload_date' column."

        if df['utilized_by_school'].dtype != bool:
            df['utilized_by_school'] = (
                df['utilized_by_school']
                .astype(str)
                .str.lower()
                .map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False})
                .fillna(False)
            )

        return df, None
    except Exception as e:
        return None, f"Error processing metadata: {str(e)}"


def get_latest_survey(survey_df: pd.DataFrame, school_id: int) -> Optional[pd.Series]:
    """Get the most recent survey row for a given school."""
    school_data = survey_df[survey_df['school_id_no'] == school_id]
    if school_data.empty:
        return None
    return school_data.sort_values('month_num').iloc[-1]


# ============================================================
# CHART BUILDERS
# ============================================================

def add_circular_arrow(fig: go.Figure, cx: float = 0.5, cy: float = 0.5,
                        radius: float = 0.48, start_deg: int = 0, end_deg: int = 350,
                        arrow_length: float = 0.04, arrow_width: float = 0.02) -> None:
    """Add a clockwise circular arrow around the radar using paper coordinates."""
    def deg_to_xy(deg):
        rad = math.radians(deg)
        return cx + radius * math.sin(rad), cy - radius * math.cos(rad)

    x_start, y_start = deg_to_xy(start_deg)
    x_end, y_end = deg_to_xy(end_deg)
    arc_path = (f"M {x_start:.4f},{y_start:.4f} "
                f"A {radius:.4f},{radius:.4f} 0 1 1 {x_end:.4f},{y_end:.4f}")
    fig.add_shape(type="path", path=arc_path, line=dict(color=USTP_GOLD, width=2),
                  xref="paper", yref="paper")

    rx, ry = x_end - cx, y_end - cy
    tx, ty = -ry, rx
    norm = math.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    tip_x, tip_y = x_end, y_end
    bx, by = tip_x - arrow_length * tx, tip_y - arrow_length * ty
    px, py = -ty * arrow_width, tx * arrow_width
    path_head = (f"M {tip_x:.4f},{tip_y:.4f} "
                 f"L {bx + px:.4f},{by + py:.4f} "
                 f"L {bx - px:.4f},{by - py:.4f} Z")
    fig.add_shape(type="path", path=path_head, fillcolor=USTP_GOLD,
                  line=dict(color=USTP_GOLD), xref="paper", yref="paper")


@st.cache_data(show_spinner=False)
def build_radar_chart(survey_values_tuple: tuple, school_name: str, dark_mode: bool) -> go.Figure:
    """Cacheable radar chart builder."""
    labels = [f"{v} ({MILESTONE_SHORT[i]})" for i, v in enumerate(VARIABLES)]
    values = list(survey_values_tuple)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill='toself', name=school_name,
        line_color=USTP_GOLD, fillcolor="rgba(245, 166, 35, 0.3)",
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>'
    ))
    text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
    fig.update_layout(
        template='plotly_dark' if dark_mode else 'plotly_white',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0],
                            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                            color=text_color),
            angularaxis=dict(direction="clockwise", tickfont=dict(size=11, color=text_color))
        ),
        title=f"Current Research Culture Profile (latest quarter)<br>{school_name}",
        showlegend=False, font=dict(color=text_color),
        annotations=[
            dict(text="Milestone cycle direction (clockwise)", xref="paper", yref="paper",
                 x=0.5, y=-0.12, showarrow=False,
                 font=dict(size=13, color=text_color),
                 bgcolor="rgba(255,255,255,0.0)", bordercolor="rgba(0,0,0,0)")
        ],
        height=500, margin=dict(l=60, r=80, t=80, b=100)
    )
        
    return fig


# ============================================================
# RESEARCH OUTPUTS DASHBOARD (data + rendering separated for caching)
# ============================================================

@st.cache_data(show_spinner=False)
def _compute_research_metrics(metadata_df: pd.DataFrame, school_id: int) -> Dict[str, Any]:
    """Pure data computation -- no dark_mode dependency, no Plotly figures."""
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id]
    metrics: Dict[str, Any] = {}

    if school_meta.empty:
        return metrics

    # 1) Theme Distribution
    theme_counts = school_meta['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    metrics['theme_counts'] = theme_counts
    metrics['top_theme'] = theme_counts.iloc[0]['Theme'] if not theme_counts.empty else "N/A"

    # 2) Theme Utilisation Rate
    if 'utilized_by_school' in school_meta.columns:
        theme_util = school_meta.groupby('theme')['utilized_by_school'].mean().reset_index()
        theme_util.columns = ['Theme', 'Utilisation Rate']
        metrics['theme_util_df'] = theme_util
        metrics['top_util_theme'] = (
            theme_util.loc[theme_util['Utilisation Rate'].idxmax(), 'Theme']
            if not theme_util.empty else "N/A"
        )
    else:
        metrics['theme_util_df'] = None
        metrics['top_util_theme'] = "N/A"

    # 3) Publication Status
    status_counts = school_meta['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    metrics['status_counts'] = status_counts
    published = (status_counts[status_counts['Status'] == 'published']['Count'].sum()
                 if not status_counts.empty else 0)
    total = status_counts['Count'].sum() if not status_counts.empty else 0
    metrics['pub_rate'] = (published / total * 100) if total > 0 else 0
    metrics['total_outputs'] = total

    # 4) Research Output Timeline
    if 'upload_date' in school_meta.columns:
        school_meta_copy = school_meta.copy()
        school_meta_copy['quarter'] = school_meta_copy['upload_date'].dt.to_period('Q').astype(str)
        output_timeline = school_meta_copy.groupby('quarter').size().reset_index(name='count')
        metrics['output_timeline'] = output_timeline if not output_timeline.empty else None
    else:
        metrics['output_timeline'] = None

    # 5) Utilisation Over Time
    if 'upload_date' in school_meta.columns and 'utilized_by_school' in school_meta.columns:
        school_meta_copy = school_meta.copy()
        school_meta_copy['quarter'] = school_meta_copy['upload_date'].dt.to_period('Q').astype(str)
        util_timeline = school_meta_copy.groupby('quarter')['utilized_by_school'].mean().reset_index()
        util_timeline.columns = ['quarter', 'utilisation_rate']
        metrics['util_timeline'] = util_timeline if not util_timeline.empty else None
    else:
        metrics['util_timeline'] = None

    # 6) School-level Utilisation Rate
    utilised = school_meta['utilized_by_school'].sum() if 'utilized_by_school' in school_meta.columns else 0
    total = len(school_meta)
    util_rate = (utilised / total * 100) if total > 0 else 0
    level, desc = interpret_utilisation_rate(util_rate)
    metrics['util_rate'] = util_rate
    metrics['util_level'] = level
    metrics['util_desc'] = desc

    # 7) Teacher Productivity (Top 10)
    teacher_counts = school_meta['teacher_name'].value_counts().reset_index().head(10)
    teacher_counts.columns = ['Teacher', 'Number of Outputs']
    metrics['teacher_counts'] = teacher_counts
    metrics['top_teacher'] = teacher_counts.iloc[0]['Teacher'] if not teacher_counts.empty else "N/A"

    # 8) Years of Service vs Output
    if 'years_of_service' in school_meta.columns and not school_meta['years_of_service'].isna().all():
        teacher_summary = school_meta.groupby('teacher_name').agg(
            output_count=('document_type', 'count'),
            years_of_service=('years_of_service', 'first')
        ).reset_index().dropna(subset=['years_of_service'])
        if len(teacher_summary) > 1:
            x = teacher_summary['years_of_service']
            y = teacher_summary['output_count']
            z = np.polyfit(x, y, 1)
            trend_x = np.linspace(x.min(), x.max(), 100)
            metrics['service_data'] = {
                'teacher_summary': teacher_summary,
                'trend_x': trend_x, 'trend_y': np.poly1d(z)(trend_x),
                'slope': z[0],
                'avg_service': teacher_summary['years_of_service'].mean(),
                'avg_output': teacher_summary['output_count'].mean(),
            }
        else:
            metrics['service_data'] = None
    else:
        metrics['service_data'] = None

    # 9) Teacher Rank breakdown
    if 'teacher_rank' in school_meta.columns and not school_meta['teacher_rank'].isna().all():
        rank_group = school_meta.groupby('teacher_rank').size().reset_index(name='total_outputs')
        teacher_rank_counts = school_meta.groupby('teacher_rank')['teacher_name'].nunique().reset_index(name='num_teachers')
        rank_summary = rank_group.merge(teacher_rank_counts, on='teacher_rank')
        rank_summary['avg_outputs'] = rank_summary['total_outputs'] / rank_summary['num_teachers']
        metrics['rank_summary'] = rank_summary
    else:
        metrics['rank_summary'] = None

    # 10) Educational Attainment breakdown
    if 'educational_attainment' in school_meta.columns and not school_meta['educational_attainment'].isna().all():
        edu_group = school_meta.groupby('educational_attainment').size().reset_index(name='total_outputs')
        teacher_edu_counts = school_meta.groupby('educational_attainment')['teacher_name'].nunique().reset_index(name='num_teachers')
        edu_summary = edu_group.merge(teacher_edu_counts, on='educational_attainment')
        edu_summary['avg_outputs'] = edu_summary['total_outputs'] / edu_summary['num_teachers']
        metrics['edu_summary'] = edu_summary
    else:
        metrics['edu_summary'] = None

    return metrics


def render_research_dashboard(metrics: Dict[str, Any], school_name: str,
                               dark_mode: bool) -> Dict[str, go.Figure]:
    """Build Plotly figures from cached metrics. Returns {name: fig} dict."""
    figs = {}
    template = 'plotly_dark' if dark_mode else 'plotly_white'

    if 'theme_counts' in metrics:
        tc = metrics['theme_counts']
        figs['theme_distribution'] = px.bar(
            tc, x='Theme', y='Count', title=f"Theme Distribution - {school_name}",
            color='Theme', color_discrete_sequence=[USTP_GOLD, DEPED_RED, USTP_DARK_BLUE]
        ).update_layout(template=template)

    if metrics.get('theme_util_df') is not None:
        tu = metrics['theme_util_df']
        figs['theme_utilisation'] = px.bar(
            tu, x='Theme', y='Utilisation Rate', title=f"Theme Utilisation Rate - {school_name}",
            color='Utilisation Rate', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    if 'status_counts' in metrics:
        sc = metrics['status_counts']
        figs['publication_status'] = px.bar(
            sc, x='Status', y='Count', title=f"Publication Status - {school_name}",
            color='Status', color_discrete_sequence=[USTP_DARK_BLUE, USTP_GOLD, DEPED_MAROON]
        ).update_layout(template=template)

    if metrics.get('output_timeline') is not None:
        tl = metrics['output_timeline']
        figs['output_timeline'] = px.line(
            tl, x='quarter', y='count', title=f"Research Output Timeline - {school_name}",
            markers=True
        ).update_layout(template=template, xaxis_title='Quarter', yaxis_title='Number of Outputs')

    if metrics.get('util_timeline') is not None:
        ut = metrics['util_timeline']
        figs['util_timeline'] = px.line(
            ut, x='quarter', y='utilisation_rate',
            title=f"Utilisation Rate Over Time - {school_name}", markers=True
        ).update_layout(template=template, xaxis_title='Quarter', yaxis_title='Utilisation Rate')

    if 'teacher_counts' in metrics:
        tp = metrics['teacher_counts']
        figs['teacher_productivity'] = px.bar(
            tp, x='Number of Outputs', y='Teacher', orientation='h',
            title=f"Teacher Productivity (Top 10) - {school_name}",
            color='Number of Outputs', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    sd = metrics.get('service_data')
    if sd is not None:
        ts = sd['teacher_summary']
        text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts['years_of_service'], y=ts['output_count'], mode='markers',
            marker=dict(size=12, color=USTP_GOLD, line=dict(color=USTP_DARK_BLUE, width=1)),
            text=ts['teacher_name'], hoverinfo='text+x+y', name='Teachers'
        ))
        fig.add_trace(go.Scatter(
            x=sd['trend_x'], y=sd['trend_y'], mode='lines',
            line=dict(color=USTP_DARK_BLUE, width=2, dash='dash'), name='Trend'
        ))
        fig.update_layout(
            template=template, title=f"Years of Service vs Research Outputs - {school_name}",
            xaxis_title="Years of Service", yaxis_title="Number of Research Outputs",
            font=dict(color=text_color), showlegend=True, height=400
        )
        figs['service_vs_output'] = fig

    if metrics.get('rank_summary') is not None:
        rs = metrics['rank_summary']
        figs['rank_breakdown'] = px.bar(
            rs, x='teacher_rank', y='total_outputs',
            title=f"Research Outputs by Teacher Rank - {school_name}",
            labels={'total_outputs': 'Total Outputs', 'teacher_rank': 'Teacher Rank'},
            color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    if metrics.get('edu_summary') is not None:
        es = metrics['edu_summary']
        figs['edu_breakdown'] = px.bar(
            es, x='educational_attainment', y='total_outputs',
            title=f"Research Outputs by Educational Attainment - {school_name}",
            labels={'total_outputs': 'Total Outputs', 'educational_attainment': 'Educational Attainment'},
            color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    return figs


# ============================================================
# BASELINE ANALYSIS
# ============================================================

def generate_baseline_synopsis(survey_row: pd.Series, school_name: str,
                                _metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate baseline synopsis from the latest survey row."""
    values = {v: survey_row[v] for v in VARIABLES}
    strengths = [v for v in VARIABLES if values[v] >= 0.6]
    gaps = [v for v in VARIABLES if values[v] <= 0.3]
    moderate = [v for v in VARIABLES if 0.3 < values[v] < 0.6]
    baseline_rcsi = np.mean(list(values.values()))

    gap_actions = {
        'C': "Build Teacher Capacity (C). Conduct training workshops on research methods and data analysis.",
        'S': "Improve Structured Support (S). Allocate budget and time for research activities.",
        'I': "Institutional Anchoring (I). Embed research into school plans and regular meetings.",
        'P': "Strengthen Community of Practice (P). Establish regular research sharing forums and peer mentoring.",
        'M': "Enhance Impact Realization (M). Document and share evidence of research impact.",
    }
    recommendations = [f"**Priority: {gap_actions[var]}**" for var in gaps if var in gap_actions]
    if not recommendations:
        recommendations.append("All variables are at moderate or high levels. Maintain current policies and focus on continuous improvement.")

    return {
        'strengths': strengths, 'gaps': gaps, 'moderate': moderate,
        'baseline_rcsi': baseline_rcsi, 'recommendations': recommendations, 'values': values,
    }


def baseline_heatmap(survey_df: Optional[pd.DataFrame], metadata_df: Optional[pd.DataFrame],
                      dark_mode: bool) -> None:
    """Display historical correlation matrix."""
    st.markdown("### Historical Correlation Matrix (Diagnostic)")
    if survey_df is None or metadata_df is None:
        st.info("Insufficient data to compute correlation (need survey and metadata).")
        return
    survey_agg = survey_df.groupby(['school_id_no', 'month_num'])[VARIABLES].mean().reset_index()
    meta = metadata_df.copy()
    meta['month_num'] = meta['upload_date'].apply(date_to_month_num)
    output_counts = meta.groupby(['school_id_no', 'month_num']).size().reset_index(name='output_count')
    merged = survey_agg.merge(output_counts, on=['school_id_no', 'month_num'], how='inner')
    if merged.empty:
        st.info("Insufficient data to compute correlation.")
        return
    corr = merged[VARIABLES + ['output_count']].corr()
    fig_corr = px.imshow(
        corr, text_auto=True, title="Correlation Matrix",
        color_continuous_scale='Blues', aspect='auto',
        labels=dict(color="Correlation Coefficient (r)")
    )
    fig_corr.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("This heatmap shows the correlation between the seven variables and research output count in the uploaded historical data.")


# ============================================================
# CYCLE vs RESEARCH OUTPUTS
# ============================================================

def cycle_research_correlation(agent: SchoolAgent, metadata_df: pd.DataFrame,
                                school_id: int, dark_mode: bool) -> None:
    """Show how cycle progression relates to cumulative research outputs."""
    if not agent.cycle_improvements:
        st.info("No cycles completed yet for this school.")
        return
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id].copy()
    school_meta['month_num'] = school_meta['upload_date'].apply(date_to_month_num)
    cumulative_outputs = [
        len(school_meta[school_meta['month_num'] <= rec.completion_month])
        for rec in agent.cycle_improvements
    ]
    text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[c.cycle_number for c in agent.cycle_improvements], y=cumulative_outputs,
        mode='markers+lines', marker=dict(size=10, color=USTP_GOLD),
        line=dict(color=USTP_DARK_BLUE), name='Research outputs'
    ))
    fig.update_layout(
        template='plotly_dark' if dark_mode else 'plotly_white',
        title="Cycle vs Cumulative Research Outputs",
        xaxis_title="Cycle Number", yaxis_title="Number of Research Outputs (cumulative)",
        showlegend=False, font=dict(color=text_color)
    )
    st.plotly_chart(fig, use_container_width=True)
    if len(cumulative_outputs) >= 2:
        increase = cumulative_outputs[-1] - cumulative_outputs[-2]
        if increase > 0:
            st.caption(f"Research output accumulation increases with each cycle (+{increase} outputs from previous cycle).")
        else:
            st.caption("Research output growth has plateaued across cycles.")
    elif len(cumulative_outputs) == 1:
        st.caption("First cycle completed. Continued output needed.")


# ============================================================
# DIVISION-LEVEL ANALYSIS (with caching)
# ============================================================

@st.cache_data(show_spinner=False)
def _compute_division_metrics(metadata_df: pd.DataFrame, school_ids_tuple: tuple) -> Dict[str, Any]:
    """Cached pure computation for division-level insights. school_ids_tuple for hashability."""
    school_ids = list(school_ids_tuple)
    result: Dict[str, Any] = {}

    if metadata_df.empty:
        result.update(top_div_teacher='N/A', top_div_school='N/A', top_div_outputs=0,
                      bottleneck_milestone='N/A', bottleneck_time=0)
        return result

    # Teacher Leaderboard
    teacher_summary = metadata_df.groupby(['teacher_name', 'school_id_no']).size().reset_index(name='total_outputs')
    for col in ['teacher_rank', 'educational_attainment', 'years_of_service']:
        if col in metadata_df.columns:
            info = metadata_df.groupby('teacher_name')[col].first().reset_index()
            teacher_summary = teacher_summary.merge(info, on='teacher_name', how='left')
    teacher_summary = teacher_summary.sort_values('total_outputs', ascending=False).head(20)
    result['leaderboard'] = teacher_summary
    if not teacher_summary.empty:
        result['top_div_teacher'] = teacher_summary.iloc[0]['teacher_name']
        result['top_div_school'] = teacher_summary.iloc[0].get('school_id_no', 'N/A')
        result['top_div_outputs'] = int(teacher_summary.iloc[0]['total_outputs'])

    result.setdefault('top_div_teacher', 'N/A')
    result.setdefault('top_div_school', 'N/A')
    result.setdefault('top_div_outputs', 0)

    # Bottleneck will be computed from history separately
    result['bottleneck_milestone'] = 'N/A'
    result['bottleneck_time'] = 0

    return result


def _compute_milestone_durations(history_per_school: dict) -> Tuple[pd.DataFrame, str, float]:
    """Compute average months per milestone from history. Returns (df, bottleneck_milestone, bottleneck_time)."""
    all_durations: Dict[int, list] = {m: [] for m in range(7)}
    for sid, hist in history_per_school.items():
        if 'milestone' not in hist or not hist['milestone']:
            continue
        milestones = hist['milestone']
        for i in range(1, len(milestones)):
            if milestones[i] != milestones[i - 1]:
                start = (milestones.index(milestones[i - 1], 0, i)
                         if milestones[i - 1] in milestones[:i] else i - 1)
                all_durations[milestones[i - 1]].append(i - start)
        last_milestone = milestones[-1]
        start = (milestones.index(last_milestone, 0, len(milestones))
                 if last_milestone in milestones else len(milestones) - 1)
        all_durations[last_milestone].append(len(milestones) - start)

    avg_durations = {m: np.mean(v) if v else np.nan for m, v in all_durations.items()}
    durations_df = pd.DataFrame({
        'Milestone': [f'M{i}' for i in range(7)],
        'Avg Months': [avg_durations.get(i, np.nan) for i in range(7)]
    }).dropna()

    if not durations_df.empty:
        max_row = durations_df.loc[durations_df['Avg Months'].idxmax()]
        return durations_df, str(max_row['Milestone']), float(max_row['Avg Months'])
    return durations_df, "N/A", 0.0


def division_level_analysis(metadata_df: pd.DataFrame, history_per_school: dict,
                             sim_agents: List[SchoolAgent], dark_mode: bool) -> Dict[str, Any]:
    """Render division-level analysis section. Returns key metrics dict."""
    st.markdown("### Division-Level Analysis")

    school_ids_tuple = tuple(a.real_id for a in sim_agents)
    div_metrics = _compute_division_metrics(metadata_df, school_ids_tuple)

    # Merge school names into leaderboard for display
    st.markdown("#### Teacher Productivity Leaderboard (Division-Wide)")
    if 'leaderboard' in div_metrics and not div_metrics['leaderboard'].empty:
        lb = div_metrics['leaderboard']
        display_cols = [c for c in ['teacher_name', 'school_id_no', 'total_outputs',
                                     'teacher_rank', 'educational_attainment', 'years_of_service'] if c in lb.columns]
        st.dataframe(lb[display_cols])
        st.caption("Top 20 teachers across the division by research output count.")
    else:
        st.info("No metadata available for division-wide leaderboard.")

    # Milestone Durations
    st.markdown("#### Milestone Transition Analysis (Average Months per Milestone)")
    durations_df, bottleneck_milestone, bottleneck_time = _compute_milestone_durations(history_per_school)
    if not durations_df.empty:
        template = 'plotly_dark' if dark_mode else 'plotly_white'
        fig_dur = px.bar(
            durations_df, x='Milestone', y='Avg Months',
            title="Average Months Spent per Milestone",
            color='Avg Months', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template, xaxis_title='Milestone', yaxis_title='Average Months')
        st.plotly_chart(fig_dur, use_container_width=True)
        get_figure_download_link(fig_dur, "milestone_durations.html")
        st.caption(f"Schools spend the most time on average in Milestone {bottleneck_milestone} ({bottleneck_time:.1f} months). This indicates a potential bottleneck.")
    else:
        st.info("Not enough transition data to compute milestone durations.")

    div_metrics['bottleneck_milestone'] = bottleneck_milestone
    div_metrics['bottleneck_time'] = bottleneck_time
    return div_metrics


# ============================================================
# SCHOOL COMPARISON DASHBOARD
# ============================================================

def school_comparison_dashboard(survey_df: pd.DataFrame, history_per_school: dict,
                                 school_info: pd.DataFrame, selected_school_ids: List[int],
                                 dark_mode: bool) -> None:
    """Overlay multiple schools' RCSI and milestone progress."""
    st.markdown("### Comparative School Analysis")
    if len(selected_school_ids) < 2:
        st.info("Please select at least two schools to compare.")
        return

    histories = {sid: history_per_school.get(sid) for sid in selected_school_ids
                 if history_per_school.get(sid)}
    if not histories:
        st.info("No simulation history for selected schools. Run the simulation first.")
        return

    fig_comp = make_subplots(rows=2, cols=1, subplot_titles=("RCSI Comparison", "Milestone Comparison"))
    for sid, hist in histories.items():
        match = school_info[school_info['school_id_no'] == sid]
        school_name = match['school_name'].values[0] if not match.empty else f"School {sid}"
        fig_comp.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'],
                                       mode='lines', name=f"{school_name} RCSI"), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'],
                                       mode='lines', name=f"{school_name} Milestone"), row=2, col=1)
    template = 'plotly_dark' if dark_mode else 'plotly_white'
    fig_comp.update_layout(height=600, template=template)
    fig_comp.update_xaxes(title_text="Month", row=1, col=1)
    fig_comp.update_yaxes(title_text="RCSI", row=1, col=1)
    fig_comp.update_xaxes(title_text="Month", row=2, col=1)
    fig_comp.update_yaxes(title_text="Milestone", row=2, col=1)
    st.plotly_chart(fig_comp, use_container_width=True)
    get_figure_download_link(fig_comp, "comparison.html")
    st.caption("Overlay of RCSI and Milestone progress for selected schools.")


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="CDO Division Research Culture Sustainability Framework", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #0D2B5E;'>CDO Division Research Culture Sustainability Framework</h1>",
    unsafe_allow_html=True
)

# --- Session state initialization ---
for key, default in [('max_schools', 200), ('num_schools', 0), ('total_teachers', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"<h2 style='color: {USTP_DARK_BLUE};'>Controls</h2>", unsafe_allow_html=True)
    dark_mode = st.checkbox("Dark Mode", value=False)
    apply_theme(dark_mode)

    st.metric(label="Total Schools Loaded", value=st.session_state.num_schools)
    st.metric(label="Total Teachers Recorded", value=st.session_state.total_teachers)
    if st.session_state.get('total_months', 0) > 0:
        st.metric(label="Simulation Month", value=st.session_state.total_months)

    st.markdown("---")
    st.markdown("#### Baseline Analysis")
    baseline_btn = st.button("Analyze Baseline", use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Policy Levers</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        u_train = st.slider("Training freq.", 0.0, 1.0, 0.5, 0.05,
                            help="Frequency of research training workshops per quarter")
        u_mentor = st.slider("Mentorship ratio", 0.0, 1.0, 0.5, 0.05,
                             help="Ratio of experienced-to-novice researcher pairings")
        u_budget = st.slider("Support budget", 0.0, 1.0, 0.5, 0.05,
                             help="Proportion of budget allocated to research support activities")
    with col2:
        u_lead = st.slider("Leadership commit.", 0.0, 1.0, 0.5, 0.05,
                           help="Degree of school leadership commitment to research culture")
        u_collab = st.slider("Collaboration freq.", 0.0, 1.0, 0.5, 0.05,
                             help="Frequency of inter-school research collaboration events")
    levers = {
        'u_train': u_train, 'u_mentor': u_mentor, 'u_budget': u_budget,
        'u_lead': u_lead, 'u_collab': u_collab
    }

    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Simulation Parameters</h3>", unsafe_allow_html=True)
    duration = st.selectbox("Run duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=9)
    random_events = st.checkbox("Enable random events", value=False,
                                help="Simulate unpredictable events (funding changes, leadership changes, etc.)")
    use_survey = st.checkbox("Override with survey data", value=True,
                             help="Replace simulated values with actual survey data when available")

    st.markdown("---")
    st.markdown("#### Simulation Actions")
    col_buttons = st.columns(3)
    with col_buttons[0]:
        run_btn = st.button("Run", use_container_width=True)
    with col_buttons[1]:
        step_btn = st.button("Step (1 month)", use_container_width=True)
    with col_buttons[2]:
        reset_btn = st.button("Reset", use_container_width=True)
    st.caption("**Run:** Full forecast for selected duration (resets history). "
                "**Step:** Advance one month without resetting.")

    st.markdown("---")
    st.markdown("#### Export Data")
    export_btn = st.button("Export results (CSV)", use_container_width=True)

# --- File Upload ---
with st.expander("Step 1: Upload your CSV files", expanded=True):
    st.markdown("""
    **Instructions:**
    - Upload your **Quarterly Survey** CSV (must contain columns: `month, school_id_no, R, A, C, S, I, P, M`).
    - Upload your **Research Metadata** CSV (must contain columns: `upload_date, teacher_name, school_id_no`).
    - You can download templates below.
    """)
    col1, col2 = st.columns(2)
    with col1:
        survey_file = st.file_uploader("Upload quarterly survey (CSV)", type=["csv"], key="survey")
    with col2:
        metadata_file = st.file_uploader("Upload research metadata (CSV)", type=["csv"], key="metadata")
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
        st.download_button("Survey Template", survey_template,
                           "quarterly_survey_template.csv", "text/csv")
    with c2:
        st.download_button("Metadata Template", metadata_template,
                           "research_metadata_template.csv", "text/csv")

# ============================================================
# MAIN AREA
# ============================================================
if survey_file is not None and metadata_file is not None:
    survey_df_raw = pd.read_csv(survey_file)
    metadata_df_raw = pd.read_csv(metadata_file)
    survey_df, school_info, survey_error = process_survey(survey_df_raw)
    metadata_df, meta_error = process_metadata(metadata_df_raw)

    if survey_error:
        st.error(f"Survey error: {survey_error}")
    elif meta_error:
        st.error(f"Metadata error: {meta_error}")
    else:
        actual_count = len(school_info)
        total_teachers = metadata_df['teacher_name'].nunique()

        # Update session state metrics (consolidated to a single rerun)
        state_changed = False
        if st.session_state.num_schools != actual_count:
            st.session_state.num_schools = actual_count
            state_changed = True
        if st.session_state.total_teachers != total_teachers:
            st.session_state.total_teachers = total_teachers
            state_changed = True
        if state_changed:
            st.rerun()

        st.success(f"Loaded {actual_count} schools and {total_teachers} teachers. Data is valid!")

        # Build school selector mapping (avoids fragile string parsing)
        school_ids = school_info['school_id_no'].tolist()
        id_to_label = {}
        for sid in school_ids:
            name = school_info[school_info['school_id_no'] == sid]['school_name'].values[0]
            id_to_label[sid] = f"ID {sid}: {name}"
        label_to_id = {v: k for k, v in id_to_label.items()}

        selected_school_label = st.selectbox("Select school", list(id_to_label.values()), index=0)
        selected_school_id = label_to_id[selected_school_label]
        selected_school_name = id_to_label[selected_school_id].split(": ", 1)[1]

        # --- Baseline from Uploaded Data ---
        st.markdown("<h2 style='text-align: center;'>Baseline from Uploaded Data</h2>",
                     unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Research Outputs (Recent)")
        df_show = metadata_df[metadata_df['school_id_no'] == selected_school_id].sort_values(
            'upload_date', ascending=False
        )
        if not df_show.empty:
            st.dataframe(df_show[['teacher_name', 'year_undertaken', 'title', 'theme',
                                   'status', 'utilized_by_school']].head(10))
        else:
            st.info("No research outputs for this school.")

        # Radar chart
        latest = get_latest_survey(survey_df, selected_school_id)
        if latest is not None:
            col_left, col_right = st.columns([1, 5])
            with col_left:
                st.markdown("**Legend:**")
                legend_items = "\n".join(
                    f"- **{v} ({MILESTONE_SHORT[i]})** -> {MILESTONE_NAMES[i]}"
                    for i, v in enumerate(VARIABLES)
                )
                st.markdown(f'<div style="font-size: 12px;">{legend_items}</div>',
                            unsafe_allow_html=True)
            with col_right:
                survey_tuple = tuple(latest[v] for v in VARIABLES)
                radar_fig = build_radar_chart(survey_tuple, selected_school_name, dark_mode)
                st.plotly_chart(radar_fig, use_container_width=True)
                get_figure_download_link(radar_fig, "radar_chart.html", "Download Radar Chart")
                st.caption("The distance from the centre (0) to each milestone point represents the "
                           "strength of that milestone. The golden arrow indicates the clockwise "
                           "milestone cycle direction.")
        else:
            st.info("No survey data for current quarter.")

        # Research Outputs Dashboard (data cached, rendering theme-aware)
        with st.expander("Research Outputs Dashboard (for selected school)"):
            metrics = _compute_research_metrics(metadata_df, selected_school_id)
            figs = render_research_dashboard(metrics, selected_school_name, dark_mode)
            if figs:
                for name, fig in figs.items():
                    st.plotly_chart(fig, use_container_width=True)
                    get_figure_download_link(fig, f"{name}.html", f"Download {name}")
                st.metric("School-level Research Utilisation Rate",
                          f"{metrics.get('util_rate', 0):.1f}% -> "
                          f"{metrics.get('util_level', 'N/A')} level",
                          help=metrics.get('util_desc', ''))
                if metrics.get('top_teacher') != "N/A":
                    st.caption(f"Top teacher: {metrics['top_teacher']}")
            else:
                st.info("No data available for this school.")

        # Baseline Synopsis
        if baseline_btn:
            latest_row = get_latest_survey(survey_df, selected_school_id)
            if latest_row is not None:
                st.session_state.baseline_synopsis = generate_baseline_synopsis(
                    latest_row, selected_school_name, metadata_df
                )
                st.session_state.baseline_survey_row = latest_row.to_dict()
                # Compute baseline std devs for significance testing (BUG FIX: was missing)
                school_survey = survey_df[survey_df['school_id_no'] == selected_school_id]
                st.session_state.baseline_std_devs = {
                    v: (school_survey[v].std() if len(school_survey) > 1 else 0.1)
                    for v in VARIABLES
                }
            else:
                st.warning("No survey data available to generate baseline.")

        if 'baseline_synopsis' in st.session_state:
            bs = st.session_state.baseline_synopsis
            st.markdown("### Baseline Synopsis")
            bg_color = '#2E2E2E' if dark_mode else '#E3F2FD'
            text_col = DARK_TEXT if dark_mode else 'inherit'
            st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 5px solid {USTP_GOLD};
                        padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
            <b>School: {selected_school_name}</b><br>
            <b>Baseline RCSI:</b> {bs['baseline_rcsi']:.3f}<br>
            <b>Strengths (>=0.6):</b> {', '.join(bs['strengths']) if bs['strengths'] else 'None'}<br>
            <b>Critical Gaps (<=0.3):</b> {', '.join(bs['gaps']) if bs['gaps'] else 'None'}<br>
            <b>Moderate (0.3-0.6):</b> {', '.join(bs['moderate']) if bs['moderate'] else 'None'}<br>
            <b>Actionable Recommendations:</b><br>
            {'<br>'.join(bs['recommendations'])}
            </div>
            """, unsafe_allow_html=True)

        # Baseline Heatmap
        baseline_heatmap(survey_df, metadata_df, dark_mode)

        # --- Initialize simulation (single source of truth) ---
        if 'sim' not in st.session_state:
            st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events)
            st.session_state.current_month = 0
            st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)

        # --- Run / Step / Reset (DRY: no duplication) ---
        if run_btn:
            st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events)
            st.session_state.current_month = 0
            st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)

            progress = st.progress(0, text="Running simulation...")
            for m in range(1, duration + 1):
                target_month = m
                if use_survey:
                    apply_survey_override(st.session_state.sim.agents, survey_df, target_month)
                st.session_state.sim.step(levers, target_month)
                st.session_state.current_month = target_month
                st.session_state.total_months = target_month
                record_history(st.session_state.history, st.session_state.sim.agents, target_month)
                progress.progress(m / duration, text=f"Month {m}/{duration}")
            progress.empty()
            st.rerun()

        if step_btn:
            target_month = st.session_state.current_month + 1
            if use_survey:
                apply_survey_override(st.session_state.sim.agents, survey_df, target_month)
            st.session_state.sim.step(levers, target_month)
            st.session_state.current_month = target_month
            st.session_state.total_months = target_month
            record_history(st.session_state.history, st.session_state.sim.agents, target_month)
            st.rerun()

        if reset_btn:
            st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events)
            st.session_state.current_month = 0
            st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)
            st.rerun()

        # --- Display Simulation Results ---
        if st.session_state.total_months > 0:
            st.markdown("<h2 style='text-align: center;'>Simulated Data</h2>",
                         unsafe_allow_html=True)
            st.markdown("---")
            hist = st.session_state.history.get(selected_school_id)
            agent = next(
                (a for a in st.session_state.sim.agents if a.real_id == selected_school_id),
                None
            )

            if hist and agent:
                # Main 4-panel chart
                fig1 = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Variable Evolution", "Milestone Progress",
                                    "Research Culture Sustainability Index (RCSI)",
                                    "Improvement per Completed Cycle")
                )
                for i, var in enumerate(VARIABLES):
                    fig1.add_trace(go.Scatter(
                        x=hist['month'], y=hist[var], mode='lines', name=var,
                        line=dict(color=VAR_COLORS[i])
                    ), row=1, col=1)
                fig1.add_trace(go.Scatter(
                    x=hist['month'], y=hist['milestone'], mode='lines', name='Milestone',
                    line=dict(color=DEPED_RED, width=3)
                ), row=1, col=2)
                fig1.add_trace(go.Scatter(
                    x=hist['month'], y=hist['running_outcome'], mode='lines', name='RCSI',
                    line=dict(color=USTP_GOLD, width=3)
                ), row=2, col=1)
                if agent.cycle_improvements:
                    fig1.add_trace(go.Bar(
                        x=[c.cycle_number for c in agent.cycle_improvements],
                        y=[c.total_improvement for c in agent.cycle_improvements],
                        name='RCSI per cycle', marker_color=USTP_DARK_BLUE
                    ), row=2, col=2)
                else:
                    fig1.add_annotation(
                        text="No cycles completed yet",
                        xref="x2 domain", yref="y2 domain",
                        x=0.5, y=0.5, showarrow=False, row=2, col=2
                    )

                text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
                template = 'plotly_dark' if dark_mode else 'plotly_white'
                fig1.update_layout(height=800, showlegend=True,
                                    font=dict(color=text_color), template=template)
                fig1.update_xaxes(title_text="Month", row=1, col=1)
                fig1.update_yaxes(title_text="Value (0-1)", row=1, col=1)
                fig1.update_xaxes(title_text="Month", row=1, col=2)
                fig1.update_yaxes(title_text="Milestone", row=1, col=2)
                fig1.update_xaxes(title_text="Month", row=2, col=1)
                fig1.update_yaxes(title_text="RCSI", row=2, col=1)
                fig1.update_xaxes(title_text="Cycle Number", row=2, col=2)
                fig1.update_yaxes(title_text="RCSI", row=2, col=2)
                st.plotly_chart(fig1, use_container_width=True)
                get_figure_download_link(fig1, "simulation_overview.html",
                                         "Download Simulation Charts")

                with st.expander("Cycle vs Research Outputs"):
                    cycle_research_correlation(agent, metadata_df, selected_school_id, dark_mode)

                with st.expander("Division-Level Analysis"):
                    div_metrics = division_level_analysis(
                        metadata_df, st.session_state.history,
                        st.session_state.sim.agents, dark_mode
                    )

                with st.expander("Comparative School Analysis"):
                    selected_comparison = st.multiselect(
                        "Select schools to compare (choose at least two)",
                        options=school_ids, default=[],
                        format_func=lambda x: id_to_label.get(x, f"ID {x}")
                    )
                    school_comparison_dashboard(
                        survey_df, st.session_state.history, school_info,
                        selected_comparison, dark_mode
                    )

                # RCSI Interpretation Table
                st.markdown("### Research Culture Sustainability Index (RCSI) Interpretation Table")
                st.markdown("""
                | RCSI Range | Level | Description |
                |------------|-------|-------------|
                | 0.0 - 0.2 | Very Low | Little to no accumulated research culture strength. |
                | 0.2 - 0.4 | Low | Minimal ecosystem vitality; research culture still weak. |
                | 0.4 - 0.6 | Moderate | Noticeable strength; research culture developing. |
                | 0.6 - 0.8 | High | Strong ecosystem; research culture becoming sustainable. |
                | 0.8 - 1.0 | Very High | Excellent vitality; research culture fully embedded. |
                """)

                # --- Simulation Synopsis (selected school) ---
                rcsi_val = agent.running_total_outcome
                rcsi_level = classify_rcsi(rcsi_val)
                milestone_name = MILESTONE_NAMES.get(
                    agent.current_milestone, f"Milestone {agent.current_milestone}"
                )

                if agent.cycle_count >= 2:
                    cycle_text = (f"has completed {agent.cycle_count} full cycles, "
                                  f"indicating a self-sustaining research culture.")
                elif agent.cycle_count == 1:
                    cycle_text = "has completed one full cycle, demonstrating initial sustainability."
                else:
                    cycle_text = "has not yet completed any full cycle."

                if agent.current_milestone == 0:
                    milestone_progress = "is at the very beginning of the journey."
                elif agent.current_milestone <= 2:
                    milestone_progress = ("has moved beyond initial readiness but remains "
                                          "in early capacity-building phases.")
                elif agent.current_milestone <= 4:
                    milestone_progress = ("has established structured support and is embedding "
                                          "research into institutional practice.")
                else:
                    milestone_progress = ("is realising tangible impact and is approaching or "
                                          "has achieved cyclical sustainability.")

                key_R = hist['R'][-1] if hist['R'] else 0
                key_M = hist['M'][-1] if hist['M'] else 0

                output_trend_text = ""
                tl = metrics.get('output_timeline')
                if tl is not None and len(tl) >= 2:
                    if tl.iloc[-1]['count'] > tl.iloc[-2]['count']:
                        output_trend_text = "Research output is increasing over time."
                    elif tl.iloc[-1]['count'] < tl.iloc[-2]['count']:
                        output_trend_text = "Research output is declining over time."
                    else:
                        output_trend_text = "Research output has remained stable."
                    avg_output = tl['count'].mean()
                    output_trend_text += f" On average, the school produces {avg_output:.1f} outputs per quarter."

                theme_util_text = ""
                tu = metrics.get('theme_util_df')
                if tu is not None and not tu.empty:
                    max_util = tu.loc[tu['Utilisation Rate'].idxmax()]
                    min_util = tu.loc[tu['Utilisation Rate'].idxmin()]
                    theme_util_text = (f"The most utilised theme is '{max_util['Theme']}' "
                                       f"({max_util['Utilisation Rate']:.0%}), while "
                                       f"'{min_util['Theme']}' has the lowest adoption "
                                       f"({min_util['Utilisation Rate']:.0%}).")

                top_teacher_text = (
                    f"The school's top researcher is {metrics['top_teacher']}."
                    if metrics.get('top_teacher') != "N/A" else ""
                )

                bg_color = '#2E2E2E' if dark_mode else '#E3F2FD'
                text_col = DARK_TEXT if dark_mode else 'inherit'
                coherent_text = f"""
                After {st.session_state.total_months} months, {selected_school_name}
                (ID {selected_school_id}) has reached {milestone_name} and {cycle_text}
                The school's Research Culture Sustainability Index (RCSI) is
                <b>{rcsi_val:.3f}</b>, which falls into the <b>{rcsi_level}</b> level.
                Key indicators: Readiness (R) = {key_R:.2f}, Impact (M) = {key_M:.2f},
                and current Milestone = {agent.current_milestone}.
                This combination suggests that {milestone_progress}
                The RCSI level <b>{rcsi_level.lower()}</b> reinforces this assessment.
                {output_trend_text}
                {theme_util_text}
                {top_teacher_text}
                Overall, the school is on a path toward research culture sustainability,
                but further policy support may be needed.
                """
                st.markdown(f"""
                <div style="background-color: {bg_color}; border-left: 5px solid {USTP_GOLD};
                            padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                <b>School {selected_school_id} ({selected_school_name}) - Simulation Synopsis</b><br>
                {coherent_text}
                </div>
                """, unsafe_allow_html=True)

                # Baseline vs Simulation Comparison
                if ('baseline_synopsis' in st.session_state and
                        'baseline_survey_row' in st.session_state):
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
                            status = ("Improving" if diff > 0.01
                                      else ("Regressing" if diff < -0.01 else "Stable"))
                            std_dev = baseline_std_devs.get(var, 0.1)
                            if abs(diff) >= 0.10:
                                significance = "Both statistically and practically significant"
                            elif abs(diff) >= 0.5 * std_dev:
                                significance = ("Statistically significant, but limited "
                                               "practical impact")
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

                # --- Division-Level Synopsis ---
                total_schools = len(st.session_state.sim.agents)
                early_stage_count = sum(
                    1 for a in st.session_state.sim.agents if a.current_milestone <= 2
                )
                advanced_stage_count = sum(
                    1 for a in st.session_state.sim.agents if a.current_milestone >= 4
                )
                # BUG FIX: transitional_percent was undefined in original
                transitional_count = total_schools - early_stage_count - advanced_stage_count
                early_percent = ((early_stage_count / total_schools) * 100
                                 if total_schools > 0 else 0)
                advanced_percent = ((advanced_stage_count / total_schools) * 100
                                    if total_schools > 0 else 0)
                transitional_percent = ((transitional_count / total_schools) * 100
                                        if total_schools > 0 else 0)

                early_text = (f"{early_percent:.1f}% of schools"
                              if early_percent > 0 else "No schools")
                advanced_text = (f"{advanced_percent:.1f}% of schools"
                                 if advanced_percent > 0 else "No schools")

                if early_percent == 100:
                    sustainability_text = ("All schools are in early milestones; "
                                           "foundational capacity-building is the priority.")
                elif early_percent >= 75:
                    sustainability_text = (f"The vast majority ({early_percent:.1f}%) are in "
                                           f"early milestones; urgent interventions needed.")
                elif early_percent >= 50:
                    sustainability_text = (f"More than half ({early_percent:.1f}%) are in early "
                                           f"milestones; targeted policy support may accelerate progress.")
                elif early_percent > 0:
                    sustainability_text = (f"{early_percent:.1f}% remain in early milestones; "
                                           f"continued efforts are required.")
                else:
                    sustainability_text = ("No schools are in early milestones; the division "
                                           "exhibits a strong, advanced research culture.")

                total_outcome = sum(a.running_total_outcome for a in st.session_state.sim.agents)
                avg_rcsi = total_outcome / total_schools if total_schools > 0 else 0
                level_avg = classify_rcsi(avg_rcsi)
                total_cycles = sum(a.cycle_count for a in st.session_state.sim.agents)
                avg_milestone = np.mean([a.current_milestone for a in st.session_state.sim.agents])
                avg_milestone_interp = interpret_avg_milestone(avg_milestone)

                school_ids_in_sim = [a.real_id for a in st.session_state.sim.agents]
                div_metadata = metadata_df[metadata_df['school_id_no'].isin(school_ids_in_sim)]
                total_utilised = (div_metadata['utilized_by_school'].sum()
                                  if 'utilized_by_school' in div_metadata.columns else 0)
                total_research_outputs = len(div_metadata)
                div_util_rate = ((total_utilised / total_research_outputs * 100)
                                 if total_research_outputs > 0 else 0)

                div_insights = div_metrics if 'div_metrics' in dir() else {}
                top_div_teacher = div_insights.get('top_div_teacher', 'N/A')
                top_div_school = div_insights.get('top_div_school', 'N/A')
                top_div_outputs = div_insights.get('top_div_outputs', 0)
                bottleneck_milestone = div_insights.get('bottleneck_milestone', 'N/A')
                bottleneck_time = div_insights.get('bottleneck_time', 0)

                # Division output trend
                output_trend_div = ""
                if not metadata_df.empty and 'upload_date' in metadata_df.columns:
                    div_timeline = metadata_df.groupby(
                        metadata_df['upload_date'].dt.to_period('Q')
                    ).size()
                    if len(div_timeline) >= 2:
                        if div_timeline.iloc[-1] > div_timeline.iloc[-2]:
                            output_trend_div = "The division's research output is increasing over time."
                        elif div_timeline.iloc[-1] < div_timeline.iloc[-2]:
                            output_trend_div = "The division's research output is declining over time."
                        else:
                            output_trend_div = "The division's research output has remained stable."
                        avg_div_output = div_timeline.mean()
                        output_trend_div += (f" On average, the division produces "
                                             f"{avg_div_output:.1f} outputs per quarter.")
                    else:
                        output_trend_div = "Division output trend data is limited."

                full_bottleneck = MILESTONE_NAMES.get(
                    int(bottleneck_milestone.replace('M', ''))
                    if isinstance(bottleneck_milestone, str) and bottleneck_milestone.startswith('M')
                    else 0,
                    bottleneck_milestone
                )
                bottleneck_insight = (
                    f"Schools spend the most time on average in {full_bottleneck} "
                    f"({bottleneck_time:.1f} months). This is the critical bottleneck."
                    if bottleneck_milestone != "N/A" else ""
                )
                top_teacher_insight = (
                    f"The division's top researcher is {top_div_teacher} from "
                    f"{top_div_school} with {top_div_outputs} outputs."
                    if top_div_teacher != "N/A" else ""
                )

                bg_color_div = '#2E2E2E' if dark_mode else '#E8F5E9'
                st.markdown(f"""
                <div style="background-color: {bg_color_div}; border-left: 5px solid {USTP_GOLD};
                            padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                <b>Division-Level Sustainability Synopsis (all {total_schools} schools)</b><br>
                - Average milestone = {avg_milestone:.1f} - {avg_milestone_interp}<br>
                - Total completed cycles = {total_cycles}<br>
                - Average RCSI = <b>{avg_rcsi:.3f}</b> - <b>{level_avg}</b> level.<br>
                - Average research utilisation rate = <b>{div_util_rate:.1f}%</b>.<br>
                - Stage distribution: {early_text} are in early stages (M<=2),
                  {transitional_percent:.1f}% transitional (M3),
                  and {advanced_text} are advanced (M>=4).<br>
                <i>Division-wide sustainability assessment:</i> {sustainability_text}<br><br>
                <b>Productivity:</b> {output_trend_div}<br>
                <b>Bottleneck:</b> {bottleneck_insight}<br>
                <b>Top Division Researcher:</b> {top_teacher_insight}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Graph Interpretations"):
                    st.markdown("""
                    - **Variable Evolution:** Shows how R, A, C, S, I, P, M change over time.
                      Higher values (closer to 1) mean stronger readiness, awareness, capacity, etc.
                    - **Milestone Progress:** The school moves through milestones 0-6. Reaching
                      milestone 6 and cycling back indicates a full sustainable cycle.
                    - **Research Culture Sustainability Index (RCSI):** Cumulative strength of
                      the research ecosystem, derived from Impact Realization (M) and
                      Collaboration (P).
                    - **Improvement per Completed Cycle:** Each bar shows the RCSI contributed by
                      one cycle. Higher bars in later cycles indicate increasing effectiveness.
                    - **Radar Chart:** Current snapshot of the seven milestone-linked variables.
                    - **Research Outputs Dashboard:** Tracks themes, publication status,
                      utilisation, teacher productivity, experience vs output, timeline,
                      top teachers, and breakdown by rank and attainment.
                    - **Division-Level Analysis:** Milestone transition bottlenecks and
                      teacher leaderboard.
                    - **Comparative Analysis:** Overlay multiple schools' RCSI and milestone
                      progress.
                    - **Cycle vs Research Outputs:** Shows how research output accumulation
                      relates to cycle progression.
                    """)

            # Export button
            if export_btn:
                all_data = []
                for agent in st.session_state.sim.agents:
                    h = st.session_state.history[agent.real_id]
                    for t in range(len(h['month'])):
                        row = {
                            'school_id': agent.real_id, 'month': h['month'][t],
                            'milestone': h['milestone'][t],
                            'running_outcome': h['running_outcome'][t]
                        }
                        for var in VARIABLES:
                            row[var] = h[var][t]
                        all_data.append(row)
                df_hist = pd.DataFrame(all_data)

                cycle_records = []
                for agent in st.session_state.sim.agents:
                    for rec in agent.cycle_improvements:
                        cycle_records.append({
                            'school_id': agent.real_id,
                            'cycle_number': rec.cycle_number,
                            'total_improvement': rec.total_improvement,
                            'completion_month': rec.completion_month
                        })
                df_cycles = pd.DataFrame(cycle_records)

                # BUG FIX: proper keyword args (missing closing paren in original)
                st.download_button(
                    "Download simulation history",
                    data=df_hist.to_csv(index=False).encode('utf-8'),
                    file_name="simulation_history.csv",
                    mime="text/csv"
                )
                st.download_button(
                    "Download cycle improvements",
                    data=df_cycles.to_csv(index=False).encode('utf-8'),
                    file_name="cycle_improvements.csv",
                    mime="text/csv"
                )
else:
    st.info("Please upload quarterly survey and research metadata CSV files to begin.")
# ============================================================
# Phase 2 Enhanced Digital Twin - CDO Research Culture Framework
# Includes ALL Phase 1 refactored code + Phase 2 extensions
# Calibration | Clustering | Sensitivity | Monte Carlo | Causal
# ============================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import math
import base64


# --- PHASE 2 additional imports ---
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
# ============================================================
# CONSTANTS
# ============================================================

# --- Colour Palette ---
USTP_DARK_BLUE = "#0D2B5E"
USTP_GOLD = "#F5A623"
DEPED_RED = "#D32F2F"
DEPED_MAROON = "#8B0000"
LIGHT_BG = "#F8F9FA"
DARK_BG = "#1E1E1E"
DARK_TEXT = "#FFFFFF"
LIGHT_TEXT = "#000000"

# --- Simulation Model Constants ---
BASE_YEAR = 2026
RANDOM_EVENT_PROB = 0.00417          # ~1 event per 240 months
LOSS_CHAMPION_PENALTY = 0.10
FUNDING_BOOST = 0.15
LEADERSHIP_PENALTY = 0.20
CYCLE_R_BONUS = 0.10
CYCLE_M_DECAY_FACTOR = 0.50
CYCLE_M_MIN_AFTER_DECAY = 0.20
CYCLE_BONUS_BASE = 0.03
CYCLE_BONUS_M_SCALE = 0.03
MONTHLY_OUTCOME_BASE = 0.001
MILESTONE_MIN_MONTHS = 6
VALUE_FLOOR = 0.1
VALUE_CEIL = 1.0

# --- Variable names ---
VARIABLES = ['R', 'A', 'C', 'S', 'I', 'P', 'M']

# --- Milestone definitions ---
MILESTONE_NAMES = {
    0: "Milestone 0 (Readiness and Relevance)",
    1: "Milestone 1 (Awareness to Action)",
    2: "Milestone 2 (Capacity Spark)",
    3: "Milestone 3 (Structured Support)",
    4: "Milestone 4 (Institutional Anchoring)",
    5: "Milestone 5 (Community of Practice)",
    6: "Milestone 6 (Impact Realization)",
}

MILESTONE_SHORT = {k: f"M{k}" for k in range(7)}

VAR_FULL_NAMES = {
    'R': 'Readiness (R)',
    'A': 'Awareness (A)',
    'C': 'Capacity (C)',
    'S': 'Structured Support (S)',
    'I': 'Institutional Anchoring (I)',
    'P': 'Community of Practice (P)',
    'M': 'Impact Realization (M)',
}

# --- Milestone transition thresholds ---
MILESTONE_THRESHOLDS = {
    0: ('A', 0.8, 1),
    1: ('C', 0.7, 2),
    2: ('S', 0.7, 3),
    3: ('I', 0.8, 4),
    4: ('P', 0.8, 5),
    5: ('M', 0.7, 6),
}

# --- RCSI Classification ---
RCSI_LEVELS = [
    (0.0, 0.2, "Very Low"),
    (0.2, 0.4, "Low"),
    (0.4, 0.6, "Moderate"),
    (0.6, 0.8, "High"),
    (0.8, 1.0, "Very High"),
]

# --- Data processing ---
REQUIRED_SURVEY_COLS = ['month', 'school_id_no'] + VARIABLES
OPTIONAL_SURVEY_COLS = ['school_name']

REQUIRED_META_COLS = ['upload_date', 'teacher_name', 'school_id_no']
OPTIONAL_META_COLS = {
    'document_type': 'abstract',
    'title': '',
    'theme': 'Uncategorized',
    'status': 'unpublished',
    'publication_link': '',
    'utilized_by_school': False,
    'utilization_date': '',
    'year_undertaken': 2025,
    'years_of_service': None,
    'teacher_rank': None,
    'educational_attainment': None,
}

# --- Plot colours for the 7 variables ---
VAR_COLORS = ['#1E88E5', USTP_GOLD, '#8E44AD', '#2ECC71', '#E67E22', DEPED_RED, '#1ABC9C']


# --- PHASE 2: Calibration & Clustering Constants ---
DEFAULT_COEFFICIENTS: Dict[str, float] = {
    'R_M': 0.02, 'A_R': 0.04, 'A_train': 0.02, 'A_M': 0.01,
    'C_train': 0.03, 'C_mentor': 0.02, 'S_budget': 0.04, 'S_mentor': 0.02,
    'I_lead': 0.03, 'I_S': 0.02, 'P_collab': 0.04, 'P_I': 0.02,
    'M_C': 0.02, 'M_P': 0.02, 'const_R': -0.01, 'const_A': -0.005,
    'const_C': -0.01, 'const_S': -0.01, 'const_I': -0.005,
    'const_P': -0.01, 'const_M': -0.005,
}

CLUSTER_MULTIPLIERS: Dict[int, float] = {0: 1.0, 1: 1.2, 2: 0.8}
CLUSTER_MAX_CLUSTERS = 3
MC_DEFAULT_RUNS = 30
MC_INIT_NOISE_STD = 0.02
MC_COEFF_NOISE_STD = 0.05
SENSITIVITY_DELTA = 0.10
CALIBRATION_MIN_SCHOOLS = 2

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def classify_rcsi(value: float) -> str:
    """Classify an RCSI value into a level string."""
    for low, high, lev in RCSI_LEVELS:
        if low <= value < high:
            return lev
    return "Very High"


def interpret_avg_milestone(avg_milestone: float) -> str:
    """Return a human-readable interpretation of the average milestone."""
    milestone_desc = [
        (0.5, "between Milestone 0 (Readiness and Relevance) and Milestone 1 (Awareness to Action), approaching M1"),
        (1.5, "between Milestone 1 (Awareness to Action) and Milestone 2 (Capacity Spark)"),
        (2.5, "between Milestone 2 (Capacity Spark) and Milestone 3 (Structured Support)"),
        (3.5, "between Milestone 3 (Structured Support) and Milestone 4 (Institutional Anchoring)"),
        (4.5, "between Milestone 4 (Institutional Anchoring) and Milestone 5 (Community of Practice)"),
        (5.5, "between Milestone 5 (Community of Practice) and Milestone 6 (Impact Realization)"),
        (float('inf'), "at or beyond Milestone 6 (Impact Realization)"),
    ]
    for threshold, desc in milestone_desc:
        if avg_milestone < threshold:
            return f"{avg_milestone:.1f} -> {desc}"
    return f"{avg_milestone:.1f} -> at or beyond Milestone 6 (Impact Realization)"


def interpret_utilisation_rate(rate: float) -> Tuple[str, str]:
    """Return (level, description) for a utilisation rate percentage."""
    if rate < 20:
        return "Very Low", "Research is rarely adopted into practice; significant gap between production and use."
    elif rate < 40:
        return "Low", "Limited adoption; most research outputs are not utilised."
    elif rate < 60:
        return "Moderate", "Roughly half of research outputs are adopted; room for improvement."
    elif rate < 80:
        return "High", "Strong translation of research into practice; research is valued."
    else:
        return "Very High", "Excellent utilisation; research is consistently applied to improve practice."


def month_str_to_num(month_str: Any) -> int:
    """Convert a 'YYYY-MM' string to a month number relative to BASE_YEAR."""
    try:
        parts = str(month_str).strip().split('-')
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            return (year - BASE_YEAR) * 12 + month
    except (ValueError, TypeError, AttributeError):
        pass
    return 0


def date_to_month_num(d: Any) -> int:
    """Convert a pandas Timestamp / datetime to month number relative to BASE_YEAR."""
    try:
        return (d.year - BASE_YEAR) * 12 + d.month
    except (ValueError, TypeError, AttributeError):
        return 0


# ============================================================
# THEME
# ============================================================

def apply_theme(dark_mode: bool) -> None:
    """Apply light or dark mode CSS to the Streamlit app."""
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


# ============================================================
# CHART DOWNLOAD HELPER
# ============================================================

def get_figure_download_link(fig, filename: str = "chart.html",
                             link_text: str = "Download chart (interactive HTML)") -> None:
    """Generate a download link for an interactive Plotly figure as a self-contained HTML file."""
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    b64 = base64.b64encode(html_str.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CycleRecord:
    cycle_number: int
    total_improvement: float
    completion_month: int


class SchoolAgent:
    """Represents a single school in the simulation with 7 milestone-linked variables."""

    def __init__(self, unique_id: int,
                 initial_R: float = 0.3, initial_A: float = 0.2,
                 initial_C: float = 0.2, initial_S: float = 0.1,
                 initial_I: float = 0.1, initial_P: float = 0.1,
                 initial_M: float = 0.0,
                 random_events_enabled: bool = False):
        self.id = unique_id
        self.real_id: int = unique_id
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
        self.random_events_enabled = random_events_enabled
        self.model_time = 0
        self._rng = np.random.RandomState()

    def apply_random_event(self) -> None:
        if not self.random_events_enabled:
            return
        if self._rng.rand() < RANDOM_EVENT_PROB:
            event_type = self._rng.choice(["loss_champion", "funding", "leadership_change"])
            if event_type == "loss_champion":
                for var in VARIABLES:
                    setattr(self, var, max(VALUE_FLOOR, getattr(self, var) - LOSS_CHAMPION_PENALTY))
            elif event_type == "funding":
                self.S = min(VALUE_CEIL, self.S + FUNDING_BOOST)
            elif event_type == "leadership_change":
                self.I = max(VALUE_FLOOR, self.I - LEADERSHIP_PENALTY)

    def _update_milestone(self) -> None:
        self.months_in_milestone += 1
        next_milestone = self.current_milestone

        if self.current_milestone in MILESTONE_THRESHOLDS:
            var_name, threshold, target = MILESTONE_THRESHOLDS[self.current_milestone]
            if getattr(self, var_name) >= threshold:
                next_milestone = target
        elif self.current_milestone == 6:
            if self.M >= 0.9 and self.R >= 0.8:
                self._complete_cycle()
                next_milestone = 0

        if next_milestone != self.current_milestone and self.months_in_milestone >= MILESTONE_MIN_MONTHS:
            self.current_milestone = next_milestone
            self.months_in_milestone = 0

    def _complete_cycle(self) -> None:
        old_M = self.M
        self.R = min(VALUE_CEIL, self.R + CYCLE_R_BONUS)
        self.M = max(CYCLE_M_MIN_AFTER_DECAY, self.M * CYCLE_M_DECAY_FACTOR)
        bonus = CYCLE_BONUS_BASE + CYCLE_BONUS_M_SCALE * old_M
        self.running_total_outcome += bonus
        self.current_cycle_accumulator += bonus
        self.cycle_count += 1
        self.cycle_improvements.append(
            CycleRecord(
                cycle_number=self.cycle_count,
                total_improvement=self.current_cycle_accumulator,
                completion_month=self.model_time,
            )
        )
        self.current_cycle_accumulator = 0.0


# ============================================================
# SIMULATION ENGINE
# ============================================================

class Simulation:
    """Manages a population of SchoolAgents and runs the monthly step."""

    def __init__(self, num_schools: int = 1, random_events: bool = False):
        self.agents: List[SchoolAgent] = [
            SchoolAgent(i, random_events_enabled=random_events) for i in range(num_schools)
        ]

    def step(self, levers: Dict[str, float], month: int) -> None:
        """Advance all agents by one month (vectorized math + per-agent bookkeeping)."""
        u_train = levers['u_train']
        u_mentor = levers['u_mentor']
        u_budget = levers['u_budget']
        u_lead = levers['u_lead']
        u_collab = levers['u_collab']

        R = np.array([a.R for a in self.agents])
        A = np.array([a.A for a in self.agents])
        C = np.array([a.C for a in self.agents])
        S = np.array([a.S for a in self.agents])
        I = np.array([a.I for a in self.agents])
        P = np.array([a.P for a in self.agents])
        M = np.array([a.M for a in self.agents])

        u_lead_eff = np.minimum(1.0, u_lead + 0.05 * M)

        R_new = np.clip(R + 0.02 * M - 0.01 * (1 - u_lead_eff), VALUE_FLOOR, VALUE_CEIL)
        A_new = np.clip(A + 0.04 * R + 0.02 * u_train + 0.01 * M - 0.005, VALUE_FLOOR, VALUE_CEIL)
        C_new = np.clip(C + 0.03 * u_train + 0.02 * u_mentor - 0.01, VALUE_FLOOR, VALUE_CEIL)
        S_new = np.clip(S + 0.04 * u_budget + 0.02 * u_mentor - 0.01 * (1 - u_lead_eff), VALUE_FLOOR, VALUE_CEIL)
        I_new = np.clip(I + 0.03 * u_lead_eff + 0.02 * S - 0.005, VALUE_FLOOR, VALUE_CEIL)
        P_new = np.clip(P + 0.04 * u_collab + 0.02 * I - 0.01, VALUE_FLOOR, VALUE_CEIL)
        M_new = np.clip(M + 0.02 * C + 0.02 * P - 0.005, VALUE_FLOOR, VALUE_CEIL)

        for i, agent in enumerate(self.agents):
            agent.R, agent.A, agent.C = float(R_new[i]), float(A_new[i]), float(C_new[i])
            agent.S, agent.I, agent.P = float(S_new[i]), float(I_new[i]), float(P_new[i])
            agent.M = float(M_new[i])

            monthly_gain = MONTHLY_OUTCOME_BASE * agent.M * (1 + agent.P)
            agent.running_total_outcome += monthly_gain
            agent.current_cycle_accumulator += monthly_gain
            agent.model_time = month
            agent._update_milestone()
            agent.apply_random_event()

    def get_agent(self, idx: int = 0) -> SchoolAgent:
        return self.agents[idx]

# ============================================================
# PHASE 2: CALIBRATABLE AGENT (extends Phase 1 SchoolAgent)
# ============================================================

class CalibratableAgent(SchoolAgent):
    """SchoolAgent that uses per-agent calibrated coefficients instead of hardcoded values."""

    def __init__(self, unique_id: int,
                 initial_R: float = 0.3, initial_A: float = 0.2,
                 initial_C: float = 0.2, initial_S: float = 0.1,
                 initial_I: float = 0.1, initial_P: float = 0.1,
                 initial_M: float = 0.0,
                 coeff_dict: Optional[Dict[str, float]] = None,
                 random_events_enabled: bool = False):
        super().__init__(
            unique_id,
            initial_R=initial_R, initial_A=initial_A,
            initial_C=initial_C, initial_S=initial_S,
            initial_I=initial_I, initial_P=initial_P,
            initial_M=initial_M,
            random_events_enabled=random_events_enabled,
        )
        self.coeff: Dict[str, float] = (
            coeff_dict if coeff_dict is not None else DEFAULT_COEFFICIENTS.copy()
        )

    def step_calibratable(self, levers: Dict[str, float]) -> None:
        """Advance this agent one month using its own coefficient dict."""
        c = self.coeff
        u_train = levers['u_train']
        u_mentor = levers['u_mentor']
        u_budget = levers['u_budget']
        u_lead = levers['u_lead']
        u_collab = levers['u_collab']

        u_lead_eff = min(1.0, u_lead + 0.05 * self.M)

        R_new = self.R + c['R_M'] * self.M + c['const_R'] * (1 - u_lead_eff)
        A_new = self.A + c['A_R'] * self.R + c['A_train'] * u_train + c['A_M'] * self.M + c['const_A']
        C_new = self.C + c['C_train'] * u_train + c['C_mentor'] * u_mentor + c['const_C']
        S_new = self.S + c['S_budget'] * u_budget + c['S_mentor'] * u_mentor + c['const_S'] * (1 - u_lead_eff)
        I_new = self.I + c['I_lead'] * u_lead_eff + c['I_S'] * self.S + c['const_I']
        P_new = self.P + c['P_collab'] * u_collab + c['P_I'] * self.I + c['const_P']
        M_new = self.M + c['M_C'] * self.C + c['M_P'] * self.P + c['const_M']

        self.R = max(VALUE_FLOOR, min(VALUE_CEIL, R_new))
        self.A = max(VALUE_FLOOR, min(VALUE_CEIL, A_new))
        self.C = max(VALUE_FLOOR, min(VALUE_CEIL, C_new))
        self.S = max(VALUE_FLOOR, min(VALUE_CEIL, S_new))
        self.I = max(VALUE_FLOOR, min(VALUE_CEIL, I_new))
        self.P = max(VALUE_FLOOR, min(VALUE_CEIL, P_new))
        self.M = max(VALUE_FLOOR, min(VALUE_CEIL, M_new))

        monthly_gain = MONTHLY_OUTCOME_BASE * self.M * (1 + self.P)
        self.running_total_outcome += monthly_gain
        self.current_cycle_accumulator += monthly_gain
        self._update_milestone()
        self.apply_random_event()


# ============================================================
# PHASE 2: CALIBRATABLE SIMULATION (extends Phase 1 Simulation)
# ============================================================

class CalibratableSimulation(Simulation):
    """Simulation that accepts per-agent parameters and uses per-agent coefficients."""

    def __init__(self, agent_params: Optional[list] = None,
                 random_events: bool = False,
                 school_ids: Optional[List[int]] = None):
        self.agents: List[CalibratableAgent] = []
        if agent_params:
            for i, params in enumerate(agent_params):
                init_R, init_A, init_C, init_S, init_I, init_P, init_M, coeff = params
                agent = CalibratableAgent(
                    i,
                    initial_R=init_R, initial_A=init_A,
                    initial_C=init_C, initial_S=initial_S,
                    initial_I=init_I, initial_P=init_P,
                    initial_M=init_M,
                    coeff_dict=coeff,
                    random_events_enabled=random_events,
                )
                if school_ids and i < len(school_ids):
                    agent.real_id = school_ids[i]
                self.agents.append(agent)
        else:
            n = len(school_ids) if school_ids else 1
            for i in range(n):
                agent = CalibratableAgent(i, random_events_enabled=random_events)
                if school_ids and i < len(school_ids):
                    agent.real_id = school_ids[i]
                self.agents.append(agent)

    def step(self, levers: Dict[str, float], month: int) -> None:
        """Advance all agents by one month using per-agent coefficients (vectorized)."""
        n = len(self.agents)
        if n == 0:
            return

        u_train = levers['u_train']
        u_mentor = levers['u_mentor']
        u_budget = levers['u_budget']
        u_lead = levers['u_lead']
        u_collab = levers['u_collab']

        c_R_M      = np.array([a.coeff['R_M']      for a in self.agents])
        c_A_R      = np.array([a.coeff['A_R']      for a in self.agents])
        c_A_train  = np.array([a.coeff['A_train']  for a in self.agents])
        c_A_M      = np.array([a.coeff['A_M']      for a in self.agents])
        c_C_train  = np.array([a.coeff['C_train']  for a in self.agents])
        c_C_mentor = np.array([a.coeff['C_mentor'] for a in self.agents])
        c_S_budget = np.array([a.coeff['S_budget'] for a in self.agents])
        c_S_mentor = np.array([a.coeff['S_mentor'] for a in self.agents])
        c_I_lead   = np.array([a.coeff['I_lead']   for a in self.agents])
        c_I_S      = np.array([a.coeff['I_S']      for a in self.agents])
        c_P_collab = np.array([a.coeff['P_collab'] for a in self.agents])
        c_P_I      = np.array([a.coeff['P_I']      for a in self.agents])
        c_M_C      = np.array([a.coeff['M_C']      for a in self.agents])
        c_M_P      = np.array([a.coeff['M_P']      for a in self.agents])
        c_const_R  = np.array([a.coeff['const_R']  for a in self.agents])
        c_const_A  = np.array([a.coeff['const_A']  for a in self.agents])
        c_const_C  = np.array([a.coeff['const_C']  for a in self.agents])
        c_const_S  = np.array([a.coeff['const_S']  for a in self.agents])
        c_const_I  = np.array([a.coeff['const_I']  for a in self.agents])
        c_const_P  = np.array([a.coeff['const_P']  for a in self.agents])
        c_const_M  = np.array([a.coeff['const_M']  for a in self.agents])

        R = np.array([a.R for a in self.agents])
        A = np.array([a.A for a in self.agents])
        C = np.array([a.C for a in self.agents])
        S = np.array([a.S for a in self.agents])
        I = np.array([a.I for a in self.agents])
        P = np.array([a.P for a in self.agents])
        M = np.array([a.M for a in self.agents])

        u_lead_eff = np.minimum(1.0, u_lead + 0.05 * M)

        R_new = np.clip(R + c_R_M * M + c_const_R * (1 - u_lead_eff), VALUE_FLOOR, VALUE_CEIL)
        A_new = np.clip(A + c_A_R * R + c_A_train * u_train + c_A_M * M + c_const_A, VALUE_FLOOR, VALUE_CEIL)
        C_new = np.clip(C + c_C_train * u_train + c_C_mentor * u_mentor + c_const_C, VALUE_FLOOR, VALUE_CEIL)
        S_new = np.clip(S + c_S_budget * u_budget + c_S_mentor * u_mentor + c_const_S * (1 - u_lead_eff), VALUE_FLOOR, VALUE_CEIL)
        I_new = np.clip(I + c_I_lead * u_lead_eff + c_I_S * S + c_const_I, VALUE_FLOOR, VALUE_CEIL)
        P_new = np.clip(P + c_P_collab * u_collab + c_P_I * I + c_const_P, VALUE_FLOOR, VALUE_CEIL)
        M_new = np.clip(M + c_M_C * C + c_M_P * P + c_const_M, VALUE_FLOOR, VALUE_CEIL)

        for i, agent in enumerate(self.agents):
            agent.R, agent.A, agent.C = float(R_new[i]), float(A_new[i]), float(C_new[i])
            agent.S, agent.I, agent.P = float(S_new[i]), float(I_new[i]), float(P_new[i])
            agent.M = float(M_new[i])

            monthly_gain = MONTHLY_OUTCOME_BASE * agent.M * (1 + agent.P)
            agent.running_total_outcome += monthly_gain
            agent.current_cycle_accumulator += monthly_gain
            agent.model_time = month
            agent._update_milestone()
            agent.apply_random_event()

    def get_agent(self, idx: int = 0) -> 'CalibratableAgent':
        return self.agents[idx]


# ============================================================
# PHASE 2: SIMULATION HELPERS (extends Phase 1 helpers)
# ============================================================

def init_calibratable_simulation(
    school_ids: List[int],
    survey_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    calibrated_coeff: Optional[Dict[str, float]],
    random_events: bool,
) -> CalibratableSimulation:
    """Create a CalibratableSimulation with per-school initial values and calibrated coefficients."""
    agent_params = get_agent_params(school_ids, survey_df, metadata_df, calibrated_coeff)
    sim = CalibratableSimulation(
        agent_params=agent_params, random_events=random_events, school_ids=school_ids,
    )
    seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
    return sim


def get_agent_params(
    school_ids: List[int],
    survey_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    calibrated_coeff: Optional[Dict[str, float]],
) -> list:
    """Build agent_params list: [(init_R, ..., init_M, coeff_dict), ...] with cluster-based multipliers."""
    cluster_map, multipliers = cluster_schools(metadata_df, school_ids)
    base_coeff = calibrated_coeff if calibrated_coeff else DEFAULT_COEFFICIENTS.copy()

    params = []
    for sid in school_ids:
        latest = get_latest_survey(survey_df, sid)
        if latest is not None:
            init_vals = (latest['R'], latest['A'], latest['C'],
                         latest['S'], latest['I'], latest['P'], latest['M'])
        else:
            init_vals = (0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0)

        mult = multipliers.get(cluster_map.get(sid, 0), 1.0)
        coeff = {k: v * mult for k, v in base_coeff.items()}
        params.append((*init_vals, coeff))
    return params


# ============================================================
# PHASE 2: CALIBRATION (sklearn-based coefficient estimation)
# ============================================================

def calibrate_coefficients(survey_df: pd.DataFrame) -> Tuple[Optional[Dict[str, float]], str]:
    """Estimate model coefficients from historical survey data using OLS regression.

    Returns (calibrated_coefficients_or_None, status_message).
    """
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn not installed. Using default coefficients."
    if survey_df is None or survey_df.empty:
        return None, "No survey data for calibration. Using defaults."
    if len(survey_df['school_id_no'].unique()) < CALIBRATION_MIN_SCHOOLS:
        return None, (f"Need at least {CALIBRATION_MIN_SCHOOLS} schools for calibration. "
                       f"Found {len(survey_df['school_id_no'].unique())}. Using defaults.")

    u_train = u_mentor = u_budget = u_lead = u_collab = 0.5

    datasets: Dict[str, Tuple[list, list]] = {var: ([], []) for var in VARIABLES}

    for sid in survey_df['school_id_no'].unique():
        sdf = survey_df[survey_df['school_id_no'] == sid].sort_values('month_num')
        if len(sdf) < 2:
            continue
        for i in range(len(sdf) - 1):
            curr, nxt = sdf.iloc[i], sdf.iloc[i + 1]
            u_lead_eff = min(1.0, u_lead + 0.05 * curr['M'])

            datasets['R'][0].append([curr['M']])
            datasets['R'][1].append(nxt['R'] - curr['R'])
            datasets['A'][0].append([curr['R'], u_train, curr['M']])
            datasets['A'][1].append(nxt['A'] - curr['A'])
            datasets['C'][0].append([u_train, u_mentor])
            datasets['C'][1].append(nxt['C'] - curr['C'])
            datasets['S'][0].append([u_budget, u_mentor])
            datasets['S'][1].append(nxt['S'] - curr['S'])
            datasets['I'][0].append([u_lead_eff, curr['S']])
            datasets['I'][1].append(nxt['I'] - curr['I'])
            datasets['P'][0].append([u_collab, curr['I']])
            datasets['P'][1].append(nxt['P'] - curr['P'])
            datasets['M'][0].append([curr['C'], curr['P']])
            datasets['M'][1].append(nxt['M'] - curr['M'])

    regression_specs = [
        ('R', ['R_M'],                    'const_R'),
        ('A', ['A_R', 'A_train', 'A_M'], 'const_A'),
        ('C', ['C_train', 'C_mentor'],    'const_C'),
        ('S', ['S_budget', 'S_mentor'],   'const_S'),
        ('I', ['I_lead', 'I_S'],          'const_I'),
        ('P', ['P_collab', 'P_I'],        'const_P'),
        ('M', ['M_C', 'M_P'],             'const_M'),
    ]

    coeff: Dict[str, float] = {}
    try:
        for var, coeff_names, const_name in regression_specs:
            X_data, y_data = datasets[var]
            if len(X_data) < 5:
                continue
            model = LinearRegression().fit(X_data, y_data)
            for name, val in zip(coeff_names, model.coef_):
                coeff[name] = float(val)
            coeff[const_name] = float(model.intercept_)

        for k, v in DEFAULT_COEFFICIENTS.items():
            if k not in coeff:
                coeff[k] = v

        return coeff, "Calibration successful. Coefficients estimated from survey data."
    except (ValueError, TypeError) as e:
        return None, f"Calibration failed: {e}. Using default coefficients."


# ============================================================
# PHASE 2: CLUSTERING (heterogeneous school profiles)
# ============================================================

@st.cache_data(show_spinner=False)
def cluster_schools(
    metadata_df: Optional[pd.DataFrame],
    school_ids: List[int],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Cluster schools by research profile features using KMeans.

    Returns (school_id -> cluster_label, cluster_label -> multiplier).
    """
    if not SKLEARN_AVAILABLE or metadata_df is None or metadata_df.empty:
        return {sid: 0 for sid in school_ids}, {0: 1.0}

    features, valid_ids = [], []
    for sid in school_ids:
        sm = metadata_df[metadata_df['school_id_no'] == sid]
        if sm.empty:
            continue
        n_teachers = sm['teacher_name'].nunique()
        n_themes = sm['theme'].nunique()
        avg_util = float(sm['utilized_by_school'].mean()) if 'utilized_by_school' in sm.columns else 0.0
        pub_count = len(sm[sm['status'] == 'published'])
        pub_rate = pub_count / len(sm) if len(sm) > 0 else 0.0
        features.append([n_teachers, n_themes, avg_util, pub_rate])
        valid_ids.append(sid)

    if len(valid_ids) < 2:
        return {sid: 0 for sid in school_ids}, {0: 1.0}

    X = np.array(features)
    n_clusters = min(CLUSTER_MAX_CLUSTERS, len(X))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    cluster_map: Dict[int, int] = {sid: int(cl) for sid, cl in zip(valid_ids, clusters)}
    for sid in school_ids:
        if sid not in cluster_map:
            cluster_map[sid] = 0

    return cluster_map, CLUSTER_MULTIPLIERS.copy()


# ============================================================
# PHASE 2: SENSITIVITY ANALYSIS (tornado chart)
# ============================================================

def run_sensitivity(
    agent_params: list,
    levers: Dict[str, float],
    duration: int,
    use_survey: bool,
    survey_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    selected_school_id: int,
    school_ids: List[int],
) -> go.Figure:
    """Compute tornado chart: sensitivity of final RCSI to each policy lever (+-10%).

    Uses CalibratableSimulation with school_ids so agent.real_id is set
    at construction time (no post-hoc assignment needed).
    """
    baseline = levers.copy()
    lever_names = ['u_train', 'u_mentor', 'u_budget', 'u_lead', 'u_collab']

    sim_base = CalibratableSimulation(
        agent_params=agent_params, random_events=False, school_ids=school_ids,
    )
    seed_agents_from_metadata(sim_base.agents, school_ids, metadata_df)

    for m in range(1, duration + 1):
        if use_survey:
            apply_survey_override(sim_base.agents, survey_df, m)
        sim_base.step(baseline, m)

    agent_base = next(
        (a for a in sim_base.agents if a.real_id == selected_school_id),
        sim_base.agents[0] if sim_base.agents else None,
    )
    if agent_base is None:
        return go.Figure()
    base_rcsi = agent_base.running_total_outcome

    results: Dict[Tuple[str, float], float] = {}
    for lever in lever_names:
        for delta in [-SENSITIVITY_DELTA, SENSITIVITY_DELTA]:
            test_levers = baseline.copy()
            test_levers[lever] = max(0.0, min(1.0, baseline[lever] + delta))

            sim = CalibratableSimulation(
                agent_params=agent_params, random_events=False, school_ids=school_ids,
            )
            seed_agents_from_metadata(sim.agents, school_ids, metadata_df)

            for m in range(1, duration + 1):
                if use_survey:
                    apply_survey_override(sim.agents, survey_df, m)
                sim.step(test_levers, m)

            agent = next(
                (a for a in sim.agents if a.real_id == selected_school_id),
                sim.agents[0] if sim.agents else None,
            )
            if agent is None:
                continue
            results[(lever, delta)] = agent.running_total_outcome

    tornado_data = []
    for lever in lever_names:
        low = results.get((lever, -SENSITIVITY_DELTA), base_rcsi) - base_rcsi
        high = results.get((lever, SENSITIVITY_DELTA), base_rcsi) - base_rcsi
        tornado_data.append({
            'Lever': lever.replace('u_', '').replace('_', ' ').title(),
            'Low Change': low,
            'High Change': high,
        })

    df = pd.DataFrame(tornado_data)
    df_melt = df.melt(id_vars='Lever', var_name='Direction', value_name='Change')
    fig = px.bar(
        df_melt, x='Change', y='Lever', color='Direction', orientation='h',
        title='Sensitivity of Final RCSI to Policy Levers (+-10%)',
        color_discrete_map={'Low Change': DEPED_RED, 'High Change': USTP_GOLD},
    )
    fig.update_layout(template='plotly_white')
    return fig


# ============================================================
# PHASE 2: MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_sim(
    num_runs: int,
    agent_params: list,
    levers: Dict[str, float],
    duration: int,
    use_survey: bool,
    survey_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    selected_school_id: int,
    school_ids: List[int],
) -> Dict[str, Any]:
    """Run Monte Carlo simulation with noisy initial conditions and coefficients.

    Uses CalibratableSimulation with school_ids so real_id is set at
    construction time. Returns initial conditions for proper causal analysis.
    """
    all_rcsi: List[List[float]] = []
    all_milestone: List[List[int]] = []
    all_initial_conditions: List[Dict[str, float]] = []

    for _run in range(num_runs):
        noisy_params = []
        for params in agent_params:
            *init_vals, coeff = params
            new_init = [
                max(VALUE_FLOOR, min(VALUE_CEIL, v + np.random.normal(0, MC_INIT_NOISE_STD)))
                for v in init_vals
            ]
            noisy_coeff = {k: v * np.random.normal(1, MC_COEFF_NOISE_STD)
                           for k, v in coeff.items()}
            noisy_params.append((*new_init, noisy_coeff))

        sim = CalibratableSimulation(
            agent_params=noisy_params, random_events=True, school_ids=school_ids,
        )
        seed_agents_from_metadata(sim.agents, school_ids, metadata_df)

        target_agent = next(
            (a for a in sim.agents if a.real_id == selected_school_id),
            sim.agents[0] if sim.agents else None,
        )
        if target_agent is None:
            continue

        all_initial_conditions.append({v: getattr(target_agent, v) for v in VARIABLES})

        rcsi_history: List[float] = []
        mil_history: List[int] = []
        for m in range(1, duration + 1):
            if use_survey:
                apply_survey_override(sim.agents, survey_df, m)
            sim.step(levers, m)
            rcsi_history.append(target_agent.running_total_outcome)
            mil_history.append(target_agent.current_milestone)

        all_rcsi.append(rcsi_history)
        all_milestone.append(mil_history)

    if not all_rcsi:
        return {
            'months': np.arange(1, duration + 1),
            'rcsi': {'p10': np.zeros(duration), 'p50': np.zeros(duration), 'p90': np.zeros(duration)},
            'milestone': {'p10': np.zeros(duration), 'p50': np.zeros(duration), 'p90': np.zeros(duration)},
            'final_rcsi': np.array([]),
            'initial_conditions': [],
        }

    all_rcsi_arr = np.array(all_rcsi)
    all_milestone_arr = np.array(all_milestone)
    months = np.arange(1, duration + 1)

    return {
        'months': months,
        'rcsi': {
            'p10': np.percentile(all_rcsi_arr, 10, axis=0),
            'p50': np.percentile(all_rcsi_arr, 50, axis=0),
            'p90': np.percentile(all_rcsi_arr, 90, axis=0),
        },
        'milestone': {
            'p10': np.percentile(all_milestone_arr, 10, axis=0),
            'p50': np.percentile(all_milestone_arr, 50, axis=0),
            'p90': np.percentile(all_milestone_arr, 90, axis=0),
        },
        'final_rcsi': all_rcsi_arr[:, -1],
        'initial_conditions': all_initial_conditions,
    }


def plot_monte_carlo_bands(mc_data: Dict[str, Any], dark_mode: bool) -> go.Figure:
    """Plot Monte Carlo uncertainty bands for RCSI and Milestone."""
    months = mc_data['months']
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("RCSI with Uncertainty Bands", "Milestone with Uncertainty Bands"),
    )
    # RCSI
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p10'], mode='lines',
                             name='P10 RCSI', line=dict(color=USTP_GOLD, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p50'], mode='lines',
                             name='Median RCSI', line=dict(color=USTP_GOLD)), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p90'], mode='lines',
                             name='P90 RCSI', line=dict(color=USTP_GOLD, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p10'], showlegend=False,
                             line=dict(color='rgba(0,0,0,0)'), hoverinfo='none'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p90'], fill='tonexty',
                             fillcolor='rgba(245, 166, 35, 0.2)',
                             line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='none'), row=1, col=1)
    # Milestone
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p10'], mode='lines',
                             name='P10 Milestone', line=dict(color=DEPED_RED, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p50'], mode='lines',
                             name='Median Milestone', line=dict(color=DEPED_RED)), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p90'], mode='lines',
                             name='P90 Milestone', line=dict(color=DEPED_RED, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p10'], showlegend=False,
                             line=dict(color='rgba(0,0,0,0)'), hoverinfo='none'), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p90'], fill='tonexty',
                             fillcolor='rgba(211, 47, 47, 0.2)',
                             line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='none'), row=2, col=1)

    fig.update_layout(height=700, template='plotly_dark' if dark_mode else 'plotly_white')
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="RCSI", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Milestone", row=2, col=1)
    return fig


# ============================================================
# PHASE 2: CAUSAL ANALYSIS
# ============================================================

def causal_analysis(mc_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Estimate the causal impact of each baseline variable on final RCSI.

    Uses the noisy initial conditions from Monte Carlo runs as the predictor
    matrix (which has genuine variance), and final RCSI as the response.

    The original implementation created X from N identical baseline copies
    (zero variance), making regression meaningless. This version uses the
    actual per-run initial conditions from Monte Carlo.
    """
    if not SKLEARN_AVAILABLE:
        return None

    final_rcsi = mc_data.get('final_rcsi')
    initial_conditions = mc_data.get('initial_conditions')

    if final_rcsi is None or initial_conditions is None:
        return None
    if len(final_rcsi) < 10:
        return None

    X = np.array([list(ic.values()) for ic in initial_conditions])
    y = np.array(final_rcsi)

    # Check for sufficient variance in predictors
    if np.any(X.std(axis=0) < 1e-6):
        return None

    try:
        model = LinearRegression().fit(X, y)
        coef_dict = {name: float(coef) for name, coef in zip(VARIABLES, model.coef_)}
        coef_dict['_r_squared'] = float(model.score(X, y))
        return coef_dict
    except (ValueError, TypeError):
        return None

# ============================================================
# SIMULATION HELPERS (eliminates duplicated init/record logic)
# ============================================================

def create_empty_history(school_ids: List[int]) -> Dict[int, Dict[str, list]]:
    """Create an empty history dictionary for all schools."""
    return {
        sid: {var: [] for var in VARIABLES + ['month', 'milestone', 'running_outcome']}
        for sid in school_ids
    }


def init_simulation_with_data(school_ids: List[int], metadata_df: pd.DataFrame,
                               random_events: bool) -> Simulation:
    """Create and seed a Simulation from school IDs and metadata."""
    sim = Simulation(num_schools=len(school_ids), random_events=random_events)
    seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
    return sim


def seed_agents_from_metadata(agents: List[SchoolAgent], school_ids: List[int],
                                metadata_df: pd.DataFrame) -> None:
    """Apply metadata-based boosts to newly created agents."""
    for idx, agent in enumerate(agents):
        agent.real_id = school_ids[idx]
        sm = metadata_df[metadata_df['school_id_no'] == agent.real_id]
        if sm.empty:
            continue
        agent.A = min(VALUE_CEIL, agent.A + len(sm[sm['document_type'] == 'abstract']) * 0.01)
        agent.M = min(VALUE_CEIL, agent.M + len(sm[sm['status'] == 'published']) * 0.02)
        agent.C = min(VALUE_CEIL, agent.C + len(sm[sm['document_type'] == 'full_paper']) * 0.005)
        agent.P = min(VALUE_CEIL, agent.P + sm['theme'].nunique() * 0.01)


def record_history(history: Dict[int, Dict[str, list]], agents: List[SchoolAgent],
                   total_months: int) -> None:
    """Snapshot all agent states into the history dict."""
    for agent in agents:
        h = history[agent.real_id]
        h['month'].append(total_months)
        for var in VARIABLES:
            h[var].append(getattr(agent, var))
        h['milestone'].append(agent.current_milestone)
        h['running_outcome'].append(agent.running_total_outcome)


def apply_survey_override(agents: List[SchoolAgent], survey_df: pd.DataFrame,
                           target_month: int) -> None:
    """Override agent variables with real survey data if available for the target month."""
    for agent in agents:
        row = survey_df[
            (survey_df['school_id_no'] == agent.real_id) &
            (survey_df['month_num'] == target_month)
        ]
        if not row.empty:
            r = row.iloc[0]
            for var in VARIABLES:
                setattr(agent, var, r[var])


# ============================================================
# DATA PROCESSING (cached)
# ============================================================

@st.cache_data(show_spinner="Processing survey data...")
def process_survey(_survey_df: pd.DataFrame):
    """Validate and process survey CSV. Returns (valid_df, school_info, error_msg)."""
    if _survey_df is None:
        return None, None, "No survey file uploaded."
    try:
        df = _survey_df.copy()

        missing_req = [col for col in REQUIRED_SURVEY_COLS if col not in df.columns]
        if missing_req:
            return None, None, f"Missing required survey columns: {', '.join(missing_req)}"

        if 'school_id_no' in df.columns:
            df['school_id_no'] = df['school_id_no'].astype(int)
        elif 'school_id' in df.columns:
            df['school_id_no'] = df['school_id'].astype(str).apply(
                lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x)
            )
        else:
            return None, None, "Survey file must contain 'school_id_no' or 'school_id' column."

        if 'school_name' not in df.columns:
            df['school_name'] = df['school_id_no'].apply(lambda x: f"School_{x}")
        else:
            df['school_name'] = df['school_name'].fillna(
                df['school_id_no'].apply(lambda x: f"School_{x}")
            )

        df['month_num'] = df['month'].apply(month_str_to_num)

        for v in VARIABLES:
            if not pd.api.types.is_numeric_dtype(df[v]):
                df[v] = pd.to_numeric(df[v], errors='coerce')
                if df[v].isna().any():
                    return None, None, f"Column {v} must be numeric."
            if df[v].isna().any():
                return None, None, f"Column {v} contains missing values."
            if (df[v] < 0).any() or (df[v] > 1).any():
                return None, None, f"Column {v} values must be between 0 and 1."

        school_info = df[['school_id_no', 'school_name']].drop_duplicates().sort_values('school_id_no')
        return df, school_info, None

    except Exception as e:
        return None, None, f"Error processing survey: {str(e)}"


@st.cache_data(show_spinner="Processing metadata...")
def process_metadata(_metadata_df: pd.DataFrame):
    """Validate and process metadata CSV. Returns (valid_df, error_msg)."""
    if _metadata_df is None:
        return None, "No metadata file uploaded."
    try:
        df = _metadata_df.copy()

        missing_req = [col for col in REQUIRED_META_COLS if col not in df.columns]
        if missing_req:
            return None, f"Missing required metadata columns: {', '.join(missing_req)}"

        if 'school_id_no' not in df.columns:
            if 'school' in df.columns:
                df['school_id_no'] = df['school'].astype(str).apply(
                    lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x)
                )
            else:
                return None, "Metadata must have 'school_id_no' or 'school' column."
        df['school_id_no'] = df['school_id_no'].astype(int)

        for col, default in OPTIONAL_META_COLS.items():
            if col not in df.columns:
                df[col] = default
                st.warning(f"Optional column '{col}' missing in metadata. Using default: {default}")
            else:
                df[col] = df[col].fillna(default)

        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
        if df['upload_date'].isna().any():
            return None, "Invalid dates in 'upload_date' column."

        if df['utilized_by_school'].dtype != bool:
            df['utilized_by_school'] = (
                df['utilized_by_school']
                .astype(str)
                .str.lower()
                .map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False})
                .fillna(False)
            )

        return df, None
    except Exception as e:
        return None, f"Error processing metadata: {str(e)}"


def get_latest_survey(survey_df: pd.DataFrame, school_id: int) -> Optional[pd.Series]:
    """Get the most recent survey row for a given school."""
    school_data = survey_df[survey_df['school_id_no'] == school_id]
    if school_data.empty:
        return None
    return school_data.sort_values('month_num').iloc[-1]


# ============================================================
# CHART BUILDERS
# ============================================================

def add_circular_arrow(fig: go.Figure, cx: float = 0.5, cy: float = 0.5,
                        radius: float = 0.48, start_deg: int = 0, end_deg: int = 350,
                        arrow_length: float = 0.04, arrow_width: float = 0.02) -> None:
    """Add a clockwise circular arrow around the radar using paper coordinates."""
    def deg_to_xy(deg):
        rad = math.radians(deg)
        return cx + radius * math.sin(rad), cy - radius * math.cos(rad)

    x_start, y_start = deg_to_xy(start_deg)
    x_end, y_end = deg_to_xy(end_deg)
    arc_path = (f"M {x_start:.4f},{y_start:.4f} "
                f"A {radius:.4f},{radius:.4f} 0 1 1 {x_end:.4f},{y_end:.4f}")
    fig.add_shape(type="path", path=arc_path, line=dict(color=USTP_GOLD, width=2),
                  xref="paper", yref="paper")

    rx, ry = x_end - cx, y_end - cy
    tx, ty = -ry, rx
    norm = math.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    tip_x, tip_y = x_end, y_end
    bx, by = tip_x - arrow_length * tx, tip_y - arrow_length * ty
    px, py = -ty * arrow_width, tx * arrow_width
    path_head = (f"M {tip_x:.4f},{tip_y:.4f} "
                 f"L {bx + px:.4f},{by + py:.4f} "
                 f"L {bx - px:.4f},{by - py:.4f} Z")
    fig.add_shape(type="path", path=path_head, fillcolor=USTP_GOLD,
                  line=dict(color=USTP_GOLD), xref="paper", yref="paper")


@st.cache_data(show_spinner=False)
def build_radar_chart(survey_values_tuple: tuple, school_name: str, dark_mode: bool) -> go.Figure:
    """Cacheable radar chart builder."""
    labels = [f"{v} ({MILESTONE_SHORT[i]})" for i, v in enumerate(VARIABLES)]
    values = list(survey_values_tuple)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill='toself', name=school_name,
        line_color=USTP_GOLD, fillcolor="rgba(245, 166, 35, 0.3)",
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>'
    ))
    text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
    fig.update_layout(
        template='plotly_dark' if dark_mode else 'plotly_white',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0],
                            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                            color=text_color),
            angularaxis=dict(direction="clockwise", tickfont=dict(size=11, color=text_color))
        ),
        title=f"Current Research Culture Profile (latest quarter)<br>{school_name}",
        showlegend=False, font=dict(color=text_color),
        annotations=[
            dict(text="Milestone cycle direction (clockwise)", xref="paper", yref="paper",
                 x=0.5, y=-0.12, showarrow=False,
                 font=dict(size=13, color=text_color),
                 bgcolor="rgba(255,255,255,0.0)", bordercolor="rgba(0,0,0,0)")
        ],
        height=500, margin=dict(l=60, r=80, t=80, b=100)
    )
    add_circular_arrow(fig)
    return fig


# ============================================================
# RESEARCH OUTPUTS DASHBOARD (data + rendering separated for caching)
# ============================================================

@st.cache_data(show_spinner=False)
def _compute_research_metrics(metadata_df: pd.DataFrame, school_id: int) -> Dict[str, Any]:
    """Pure data computation -- no dark_mode dependency, no Plotly figures."""
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id]
    metrics: Dict[str, Any] = {}

    if school_meta.empty:
        return metrics

    # 1) Theme Distribution
    theme_counts = school_meta['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    metrics['theme_counts'] = theme_counts
    metrics['top_theme'] = theme_counts.iloc[0]['Theme'] if not theme_counts.empty else "N/A"

    # 2) Theme Utilisation Rate
    if 'utilized_by_school' in school_meta.columns:
        theme_util = school_meta.groupby('theme')['utilized_by_school'].mean().reset_index()
        theme_util.columns = ['Theme', 'Utilisation Rate']
        metrics['theme_util_df'] = theme_util
        metrics['top_util_theme'] = (
            theme_util.loc[theme_util['Utilisation Rate'].idxmax(), 'Theme']
            if not theme_util.empty else "N/A"
        )
    else:
        metrics['theme_util_df'] = None
        metrics['top_util_theme'] = "N/A"

    # 3) Publication Status
    status_counts = school_meta['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    metrics['status_counts'] = status_counts
    published = (status_counts[status_counts['Status'] == 'published']['Count'].sum()
                 if not status_counts.empty else 0)
    total = status_counts['Count'].sum() if not status_counts.empty else 0
    metrics['pub_rate'] = (published / total * 100) if total > 0 else 0
    metrics['total_outputs'] = total

    # 4) Research Output Timeline
    if 'upload_date' in school_meta.columns:
        school_meta_copy = school_meta.copy()
        school_meta_copy['quarter'] = school_meta_copy['upload_date'].dt.to_period('Q').astype(str)
        output_timeline = school_meta_copy.groupby('quarter').size().reset_index(name='count')
        metrics['output_timeline'] = output_timeline if not output_timeline.empty else None
    else:
        metrics['output_timeline'] = None

    # 5) Utilisation Over Time
    if 'upload_date' in school_meta.columns and 'utilized_by_school' in school_meta.columns:
        school_meta_copy = school_meta.copy()
        school_meta_copy['quarter'] = school_meta_copy['upload_date'].dt.to_period('Q').astype(str)
        util_timeline = school_meta_copy.groupby('quarter')['utilized_by_school'].mean().reset_index()
        util_timeline.columns = ['quarter', 'utilisation_rate']
        metrics['util_timeline'] = util_timeline if not util_timeline.empty else None
    else:
        metrics['util_timeline'] = None

    # 6) School-level Utilisation Rate
    utilised = school_meta['utilized_by_school'].sum() if 'utilized_by_school' in school_meta.columns else 0
    total = len(school_meta)
    util_rate = (utilised / total * 100) if total > 0 else 0
    level, desc = interpret_utilisation_rate(util_rate)
    metrics['util_rate'] = util_rate
    metrics['util_level'] = level
    metrics['util_desc'] = desc

    # 7) Teacher Productivity (Top 10)
    teacher_counts = school_meta['teacher_name'].value_counts().reset_index().head(10)
    teacher_counts.columns = ['Teacher', 'Number of Outputs']
    metrics['teacher_counts'] = teacher_counts
    metrics['top_teacher'] = teacher_counts.iloc[0]['Teacher'] if not teacher_counts.empty else "N/A"

    # 8) Years of Service vs Output
    if 'years_of_service' in school_meta.columns and not school_meta['years_of_service'].isna().all():
        teacher_summary = school_meta.groupby('teacher_name').agg(
            output_count=('document_type', 'count'),
            years_of_service=('years_of_service', 'first')
        ).reset_index().dropna(subset=['years_of_service'])
        if len(teacher_summary) > 1:
            x = teacher_summary['years_of_service']
            y = teacher_summary['output_count']
            z = np.polyfit(x, y, 1)
            trend_x = np.linspace(x.min(), x.max(), 100)
            metrics['service_data'] = {
                'teacher_summary': teacher_summary,
                'trend_x': trend_x, 'trend_y': np.poly1d(z)(trend_x),
                'slope': z[0],
                'avg_service': teacher_summary['years_of_service'].mean(),
                'avg_output': teacher_summary['output_count'].mean(),
            }
        else:
            metrics['service_data'] = None
    else:
        metrics['service_data'] = None

    # 9) Teacher Rank breakdown
    if 'teacher_rank' in school_meta.columns and not school_meta['teacher_rank'].isna().all():
        rank_group = school_meta.groupby('teacher_rank').size().reset_index(name='total_outputs')
        teacher_rank_counts = school_meta.groupby('teacher_rank')['teacher_name'].nunique().reset_index(name='num_teachers')
        rank_summary = rank_group.merge(teacher_rank_counts, on='teacher_rank')
        rank_summary['avg_outputs'] = rank_summary['total_outputs'] / rank_summary['num_teachers']
        metrics['rank_summary'] = rank_summary
    else:
        metrics['rank_summary'] = None

    # 10) Educational Attainment breakdown
    if 'educational_attainment' in school_meta.columns and not school_meta['educational_attainment'].isna().all():
        edu_group = school_meta.groupby('educational_attainment').size().reset_index(name='total_outputs')
        teacher_edu_counts = school_meta.groupby('educational_attainment')['teacher_name'].nunique().reset_index(name='num_teachers')
        edu_summary = edu_group.merge(teacher_edu_counts, on='educational_attainment')
        edu_summary['avg_outputs'] = edu_summary['total_outputs'] / edu_summary['num_teachers']
        metrics['edu_summary'] = edu_summary
    else:
        metrics['edu_summary'] = None

    return metrics


def render_research_dashboard(metrics: Dict[str, Any], school_name: str,
                               dark_mode: bool) -> Dict[str, go.Figure]:
    """Build Plotly figures from cached metrics. Returns {name: fig} dict."""
    figs = {}
    template = 'plotly_dark' if dark_mode else 'plotly_white'

    if 'theme_counts' in metrics:
        tc = metrics['theme_counts']
        figs['theme_distribution'] = px.bar(
            tc, x='Theme', y='Count', title=f"Theme Distribution - {school_name}",
            color='Theme', color_discrete_sequence=[USTP_GOLD, DEPED_RED, USTP_DARK_BLUE]
        ).update_layout(template=template)

    if metrics.get('theme_util_df') is not None:
        tu = metrics['theme_util_df']
        figs['theme_utilisation'] = px.bar(
            tu, x='Theme', y='Utilisation Rate', title=f"Theme Utilisation Rate - {school_name}",
            color='Utilisation Rate', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    if 'status_counts' in metrics:
        sc = metrics['status_counts']
        figs['publication_status'] = px.bar(
            sc, x='Status', y='Count', title=f"Publication Status - {school_name}",
            color='Status', color_discrete_sequence=[USTP_DARK_BLUE, USTP_GOLD, DEPED_MAROON]
        ).update_layout(template=template)

    if metrics.get('output_timeline') is not None:
        tl = metrics['output_timeline']
        figs['output_timeline'] = px.line(
            tl, x='quarter', y='count', title=f"Research Output Timeline - {school_name}",
            markers=True
        ).update_layout(template=template, xaxis_title='Quarter', yaxis_title='Number of Outputs')

    if metrics.get('util_timeline') is not None:
        ut = metrics['util_timeline']
        figs['util_timeline'] = px.line(
            ut, x='quarter', y='utilisation_rate',
            title=f"Utilisation Rate Over Time - {school_name}", markers=True
        ).update_layout(template=template, xaxis_title='Quarter', yaxis_title='Utilisation Rate')

    if 'teacher_counts' in metrics:
        tp = metrics['teacher_counts']
        figs['teacher_productivity'] = px.bar(
            tp, x='Number of Outputs', y='Teacher', orientation='h',
            title=f"Teacher Productivity (Top 10) - {school_name}",
            color='Number of Outputs', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    sd = metrics.get('service_data')
    if sd is not None:
        ts = sd['teacher_summary']
        text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts['years_of_service'], y=ts['output_count'], mode='markers',
            marker=dict(size=12, color=USTP_GOLD, line=dict(color=USTP_DARK_BLUE, width=1)),
            text=ts['teacher_name'], hoverinfo='text+x+y', name='Teachers'
        ))
        fig.add_trace(go.Scatter(
            x=sd['trend_x'], y=sd['trend_y'], mode='lines',
            line=dict(color=USTP_DARK_BLUE, width=2, dash='dash'), name='Trend'
        ))
        fig.update_layout(
            template=template, title=f"Years of Service vs Research Outputs - {school_name}",
            xaxis_title="Years of Service", yaxis_title="Number of Research Outputs",
            font=dict(color=text_color), showlegend=True, height=400
        )
        figs['service_vs_output'] = fig

    if metrics.get('rank_summary') is not None:
        rs = metrics['rank_summary']
        figs['rank_breakdown'] = px.bar(
            rs, x='teacher_rank', y='total_outputs',
            title=f"Research Outputs by Teacher Rank - {school_name}",
            labels={'total_outputs': 'Total Outputs', 'teacher_rank': 'Teacher Rank'},
            color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    if metrics.get('edu_summary') is not None:
        es = metrics['edu_summary']
        figs['edu_breakdown'] = px.bar(
            es, x='educational_attainment', y='total_outputs',
            title=f"Research Outputs by Educational Attainment - {school_name}",
            labels={'total_outputs': 'Total Outputs', 'educational_attainment': 'Educational Attainment'},
            color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template)

    return figs


# ============================================================
# BASELINE ANALYSIS
# ============================================================

def generate_baseline_synopsis(survey_row: pd.Series, school_name: str,
                                _metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate baseline synopsis from the latest survey row."""
    values = {v: survey_row[v] for v in VARIABLES}
    strengths = [v for v in VARIABLES if values[v] >= 0.6]
    gaps = [v for v in VARIABLES if values[v] <= 0.3]
    moderate = [v for v in VARIABLES if 0.3 < values[v] < 0.6]
    baseline_rcsi = np.mean(list(values.values()))

    gap_actions = {
        'C': "Build Teacher Capacity (C). Conduct training workshops on research methods and data analysis.",
        'S': "Improve Structured Support (S). Allocate budget and time for research activities.",
        'I': "Institutional Anchoring (I). Embed research into school plans and regular meetings.",
        'P': "Strengthen Community of Practice (P). Establish regular research sharing forums and peer mentoring.",
        'M': "Enhance Impact Realization (M). Document and share evidence of research impact.",
    }
    recommendations = [f"**Priority: {gap_actions[var]}**" for var in gaps if var in gap_actions]
    if not recommendations:
        recommendations.append("All variables are at moderate or high levels. Maintain current policies and focus on continuous improvement.")

    return {
        'strengths': strengths, 'gaps': gaps, 'moderate': moderate,
        'baseline_rcsi': baseline_rcsi, 'recommendations': recommendations, 'values': values,
    }


def baseline_heatmap(survey_df: Optional[pd.DataFrame], metadata_df: Optional[pd.DataFrame],
                      dark_mode: bool) -> None:
    """Display historical correlation matrix."""
    st.markdown("### Historical Correlation Matrix (Diagnostic)")
    if survey_df is None or metadata_df is None:
        st.info("Insufficient data to compute correlation (need survey and metadata).")
        return
    survey_agg = survey_df.groupby(['school_id_no', 'month_num'])[VARIABLES].mean().reset_index()
    meta = metadata_df.copy()
    meta['month_num'] = meta['upload_date'].apply(date_to_month_num)
    output_counts = meta.groupby(['school_id_no', 'month_num']).size().reset_index(name='output_count')
    merged = survey_agg.merge(output_counts, on=['school_id_no', 'month_num'], how='inner')
    if merged.empty:
        st.info("Insufficient data to compute correlation.")
        return
    corr = merged[VARIABLES + ['output_count']].corr()
    fig_corr = px.imshow(
        corr, text_auto=True, title="Correlation Matrix",
        color_continuous_scale='Blues', aspect='auto',
        labels=dict(color="Correlation Coefficient (r)")
    )
    fig_corr.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("This heatmap shows the correlation between the seven variables and research output count in the uploaded historical data.")


# ============================================================
# CYCLE vs RESEARCH OUTPUTS
# ============================================================

def cycle_research_correlation(agent: SchoolAgent, metadata_df: pd.DataFrame,
                                school_id: int, dark_mode: bool) -> None:
    """Show how cycle progression relates to cumulative research outputs."""
    if not agent.cycle_improvements:
        st.info("No cycles completed yet for this school.")
        return
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id].copy()
    school_meta['month_num'] = school_meta['upload_date'].apply(date_to_month_num)
    cumulative_outputs = [
        len(school_meta[school_meta['month_num'] <= rec.completion_month])
        for rec in agent.cycle_improvements
    ]
    text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[c.cycle_number for c in agent.cycle_improvements], y=cumulative_outputs,
        mode='markers+lines', marker=dict(size=10, color=USTP_GOLD),
        line=dict(color=USTP_DARK_BLUE), name='Research outputs'
    ))
    fig.update_layout(
        template='plotly_dark' if dark_mode else 'plotly_white',
        title="Cycle vs Cumulative Research Outputs",
        xaxis_title="Cycle Number", yaxis_title="Number of Research Outputs (cumulative)",
        showlegend=False, font=dict(color=text_color)
    )
    st.plotly_chart(fig, use_container_width=True)
    if len(cumulative_outputs) >= 2:
        increase = cumulative_outputs[-1] - cumulative_outputs[-2]
        if increase > 0:
            st.caption(f"Research output accumulation increases with each cycle (+{increase} outputs from previous cycle).")
        else:
            st.caption("Research output growth has plateaued across cycles.")
    elif len(cumulative_outputs) == 1:
        st.caption("First cycle completed. Continued output needed.")


# ============================================================
# DIVISION-LEVEL ANALYSIS (with caching)
# ============================================================

@st.cache_data(show_spinner=False)
def _compute_division_metrics(metadata_df: pd.DataFrame, school_ids_tuple: tuple) -> Dict[str, Any]:
    """Cached pure computation for division-level insights. school_ids_tuple for hashability."""
    school_ids = list(school_ids_tuple)
    result: Dict[str, Any] = {}

    if metadata_df.empty:
        result.update(top_div_teacher='N/A', top_div_school='N/A', top_div_outputs=0,
                      bottleneck_milestone='N/A', bottleneck_time=0)
        return result

    # Teacher Leaderboard
    teacher_summary = metadata_df.groupby(['teacher_name', 'school_id_no']).size().reset_index(name='total_outputs')
    for col in ['teacher_rank', 'educational_attainment', 'years_of_service']:
        if col in metadata_df.columns:
            info = metadata_df.groupby('teacher_name')[col].first().reset_index()
            teacher_summary = teacher_summary.merge(info, on='teacher_name', how='left')
    teacher_summary = teacher_summary.sort_values('total_outputs', ascending=False).head(20)
    result['leaderboard'] = teacher_summary
    if not teacher_summary.empty:
        result['top_div_teacher'] = teacher_summary.iloc[0]['teacher_name']
        result['top_div_school'] = teacher_summary.iloc[0].get('school_id_no', 'N/A')
        result['top_div_outputs'] = int(teacher_summary.iloc[0]['total_outputs'])

    result.setdefault('top_div_teacher', 'N/A')
    result.setdefault('top_div_school', 'N/A')
    result.setdefault('top_div_outputs', 0)

    # Bottleneck will be computed from history separately
    result['bottleneck_milestone'] = 'N/A'
    result['bottleneck_time'] = 0

    return result


def _compute_milestone_durations(history_per_school: dict) -> Tuple[pd.DataFrame, str, float]:
    """Compute average months per milestone from history. Returns (df, bottleneck_milestone, bottleneck_time)."""
    all_durations: Dict[int, list] = {m: [] for m in range(7)}
    for sid, hist in history_per_school.items():
        if 'milestone' not in hist or not hist['milestone']:
            continue
        milestones = hist['milestone']
        for i in range(1, len(milestones)):
            if milestones[i] != milestones[i - 1]:
                start = (milestones.index(milestones[i - 1], 0, i)
                         if milestones[i - 1] in milestones[:i] else i - 1)
                all_durations[milestones[i - 1]].append(i - start)
        last_milestone = milestones[-1]
        start = (milestones.index(last_milestone, 0, len(milestones))
                 if last_milestone in milestones else len(milestones) - 1)
        all_durations[last_milestone].append(len(milestones) - start)

    avg_durations = {m: np.mean(v) if v else np.nan for m, v in all_durations.items()}
    durations_df = pd.DataFrame({
        'Milestone': [f'M{i}' for i in range(7)],
        'Avg Months': [avg_durations.get(i, np.nan) for i in range(7)]
    }).dropna()

    if not durations_df.empty:
        max_row = durations_df.loc[durations_df['Avg Months'].idxmax()]
        return durations_df, str(max_row['Milestone']), float(max_row['Avg Months'])
    return durations_df, "N/A", 0.0


def division_level_analysis(metadata_df: pd.DataFrame, history_per_school: dict,
                             sim_agents: List[SchoolAgent], dark_mode: bool) -> Dict[str, Any]:
    """Render division-level analysis section. Returns key metrics dict."""
    st.markdown("### Division-Level Analysis")

    school_ids_tuple = tuple(a.real_id for a in sim_agents)
    div_metrics = _compute_division_metrics(metadata_df, school_ids_tuple)

    # Merge school names into leaderboard for display
    st.markdown("#### Teacher Productivity Leaderboard (Division-Wide)")
    if 'leaderboard' in div_metrics and not div_metrics['leaderboard'].empty:
        lb = div_metrics['leaderboard']
        display_cols = [c for c in ['teacher_name', 'school_id_no', 'total_outputs',
                                     'teacher_rank', 'educational_attainment', 'years_of_service'] if c in lb.columns]
        st.dataframe(lb[display_cols])
        st.caption("Top 20 teachers across the division by research output count.")
    else:
        st.info("No metadata available for division-wide leaderboard.")

    # Milestone Durations
    st.markdown("#### Milestone Transition Analysis (Average Months per Milestone)")
    durations_df, bottleneck_milestone, bottleneck_time = _compute_milestone_durations(history_per_school)
    if not durations_df.empty:
        template = 'plotly_dark' if dark_mode else 'plotly_white'
        fig_dur = px.bar(
            durations_df, x='Milestone', y='Avg Months',
            title="Average Months Spent per Milestone",
            color='Avg Months', color_continuous_scale=['#F5A623', '#0D2B5E']
        ).update_layout(template=template, xaxis_title='Milestone', yaxis_title='Average Months')
        st.plotly_chart(fig_dur, use_container_width=True)
        get_figure_download_link(fig_dur, "milestone_durations.html")
        st.caption(f"Schools spend the most time on average in Milestone {bottleneck_milestone} ({bottleneck_time:.1f} months). This indicates a potential bottleneck.")
    else:
        st.info("Not enough transition data to compute milestone durations.")

    div_metrics['bottleneck_milestone'] = bottleneck_milestone
    div_metrics['bottleneck_time'] = bottleneck_time
    return div_metrics


# ============================================================
# SCHOOL COMPARISON DASHBOARD
# ============================================================

def school_comparison_dashboard(survey_df: pd.DataFrame, history_per_school: dict,
                                 school_info: pd.DataFrame, selected_school_ids: List[int],
                                 dark_mode: bool) -> None:
    """Overlay multiple schools' RCSI and milestone progress."""
    st.markdown("### Comparative School Analysis")
    if len(selected_school_ids) < 2:
        st.info("Please select at least two schools to compare.")
        return

    histories = {sid: history_per_school.get(sid) for sid in selected_school_ids
                 if history_per_school.get(sid)}
    if not histories:
        st.info("No simulation history for selected schools. Run the simulation first.")
        return

    fig_comp = make_subplots(rows=2, cols=1, subplot_titles=("RCSI Comparison", "Milestone Comparison"))
    for sid, hist in histories.items():
        match = school_info[school_info['school_id_no'] == sid]
        school_name = match['school_name'].values[0] if not match.empty else f"School {sid}"
        fig_comp.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'],
                                       mode='lines', name=f"{school_name} RCSI"), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'],
                                       mode='lines', name=f"{school_name} Milestone"), row=2, col=1)
    template = 'plotly_dark' if dark_mode else 'plotly_white'
    fig_comp.update_layout(height=600, template=template)
    fig_comp.update_xaxes(title_text="Month", row=1, col=1)
    fig_comp.update_yaxes(title_text="RCSI", row=1, col=1)
    fig_comp.update_xaxes(title_text="Month", row=2, col=1)
    fig_comp.update_yaxes(title_text="Milestone", row=2, col=1)
    st.plotly_chart(fig_comp, use_container_width=True)
    get_figure_download_link(fig_comp, "comparison.html")
    st.caption("Overlay of RCSI and Milestone progress for selected schools.")


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="CDO Division Research Culture Sustainability Framework", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #0D2B5E;'>CDO Division Research Culture Sustainability Framework</h1>",
    unsafe_allow_html=True
)

# --- Session state initialization ---
for key, default in [('max_schools', 200), ('num_schools', 0), ('total_teachers', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"<h2 style='color: {USTP_DARK_BLUE};'>Controls</h2>", unsafe_allow_html=True)
    dark_mode = st.checkbox("Dark Mode", value=False)
    apply_theme(dark_mode)

    st.metric(label="Total Schools Loaded", value=st.session_state.num_schools)
    st.metric(label="Total Teachers Recorded", value=st.session_state.total_teachers)
    if st.session_state.get('total_months', 0) > 0:
        st.metric(label="Simulation Month", value=st.session_state.total_months)

    st.markdown("---")
    st.markdown("#### Baseline Analysis")
    baseline_btn = st.button("Analyze Baseline", use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Policy Levers</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        u_train = st.slider("Training freq.", 0.0, 1.0, 0.5, 0.05,
                            help="Frequency of research training workshops per quarter")
        u_mentor = st.slider("Mentorship ratio", 0.0, 1.0, 0.5, 0.05,
                             help="Ratio of experienced-to-novice researcher pairings")
        u_budget = st.slider("Support budget", 0.0, 1.0, 0.5, 0.05,
                             help="Proportion of budget allocated to research support activities")
    with col2:
        u_lead = st.slider("Leadership commit.", 0.0, 1.0, 0.5, 0.05,
                           help="Degree of school leadership commitment to research culture")
        u_collab = st.slider("Collaboration freq.", 0.0, 1.0, 0.5, 0.05,
                             help="Frequency of inter-school research collaboration events")
    levers = {
        'u_train': u_train, 'u_mentor': u_mentor, 'u_budget': u_budget,
        'u_lead': u_lead, 'u_collab': u_collab
    }

    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Simulation Parameters</h3>", unsafe_allow_html=True)
    duration = st.selectbox("Run duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=9)
    random_events = st.checkbox("Enable random events", value=False,
                                help="Simulate unpredictable events (funding changes, leadership changes, etc.)")
    use_survey = st.checkbox("Override with survey data", value=True,
                             help="Replace simulated values with actual survey data when available")

    st.markdown("---")
    st.markdown("#### Simulation Actions")
    col_buttons = st.columns(3)
    with col_buttons[0]:
        run_btn = st.button("Run", use_container_width=True)
    with col_buttons[1]:
        step_btn = st.button("Step (1 month)", use_container_width=True)
    with col_buttons[2]:
        reset_btn = st.button("Reset", use_container_width=True)
    st.caption("**Run:** Full forecast for selected duration (resets history). "
                "**Step:** Advance one month without resetting.")

    st.markdown("---")
    st.markdown("#### Export Data")
    export_btn = st.button("Export results (CSV)", use_container_width=True)


    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Monte Carlo (Phase 2)</h3>",
                unsafe_allow_html=True)
    mc_enabled = st.checkbox("Enable Monte Carlo analysis", value=False,
                             help="Run multiple simulations with randomised parameters to estimate uncertainty")
    mc_runs = st.number_input("Number of Monte Carlo runs", min_value=10, max_value=100,
                              value=MC_DEFAULT_RUNS, step=10,
                              help="More runs give tighter uncertainty bands but take longer")
# --- File Upload ---
with st.expander("Step 1: Upload your CSV files", expanded=True):
    st.markdown("""
    **Instructions:**
    - Upload your **Quarterly Survey** CSV (must contain columns: `month, school_id_no, R, A, C, S, I, P, M`).
    - Upload your **Research Metadata** CSV (must contain columns: `upload_date, teacher_name, school_id_no`).
    - You can download templates below.
    """)
    col1, col2 = st.columns(2)
    with col1:
        survey_file = st.file_uploader("Upload quarterly survey (CSV)", type=["csv"], key="survey")
    with col2:
        metadata_file = st.file_uploader("Upload research metadata (CSV)", type=["csv"], key="metadata")
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
        st.download_button("Survey Template", survey_template,
                           "quarterly_survey_template.csv", "text/csv")
    with c2:
        st.download_button("Metadata Template", metadata_template,
                           "research_metadata_template.csv", "text/csv")

# ============================================================
# MAIN AREA
# ============================================================
if survey_file is not None and metadata_file is not None:
    survey_df_raw = pd.read_csv(survey_file)
    metadata_df_raw = pd.read_csv(metadata_file)
    survey_df, school_info, survey_error = process_survey(survey_df_raw)
    metadata_df, meta_error = process_metadata(metadata_df_raw)

    if survey_error:
        st.error(f"Survey error: {survey_error}")
    elif meta_error:
        st.error(f"Metadata error: {meta_error}")
    else:
        actual_count = len(school_info)
        total_teachers = metadata_df['teacher_name'].nunique()

        # Update session state metrics (consolidated to a single rerun)
        state_changed = False
        if st.session_state.num_schools != actual_count:
            st.session_state.num_schools = actual_count
            state_changed = True
        if st.session_state.total_teachers != total_teachers:
            st.session_state.total_teachers = total_teachers
            state_changed = True
        if state_changed:
            st.rerun()

        st.success(f"Loaded {actual_count} schools and {total_teachers} teachers. Data is valid!")

        # Build school selector mapping (avoids fragile string parsing)
        school_ids = school_info['school_id_no'].tolist()
        id_to_label = {}
        for sid in school_ids:
            name = school_info[school_info['school_id_no'] == sid]['school_name'].values[0]
            id_to_label[sid] = f"ID {sid}: {name}"
        label_to_id = {v: k for k, v in id_to_label.items()}

        selected_school_label = st.selectbox("Select school", list(id_to_label.values()), index=0)
        selected_school_id = label_to_id[selected_school_label]
        selected_school_name = id_to_label[selected_school_id].split(": ", 1)[1]

        # --- Baseline from Uploaded Data ---
        st.markdown("<h2 style='text-align: center;'>Baseline from Uploaded Data</h2>",
                     unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Research Outputs (Recent)")
        df_show = metadata_df[metadata_df['school_id_no'] == selected_school_id].sort_values(
            'upload_date', ascending=False
        )
        if not df_show.empty:
            st.dataframe(df_show[['teacher_name', 'year_undertaken', 'title', 'theme',
                                   'status', 'utilized_by_school']].head(10))
        else:
            st.info("No research outputs for this school.")

        # Radar chart
        latest = get_latest_survey(survey_df, selected_school_id)
        if latest is not None:
            col_left, col_right = st.columns([1, 5])
            with col_left:
                st.markdown("**Legend:**")
                legend_items = "\n".join(
                    f"- **{v} ({MILESTONE_SHORT[i]})** -> {MILESTONE_NAMES[i]}"
                    for i, v in enumerate(VARIABLES)
                )
                st.markdown(f'<div style="font-size: 12px;">{legend_items}</div>',
                            unsafe_allow_html=True)
            with col_right:
                survey_tuple = tuple(latest[v] for v in VARIABLES)
                radar_fig = build_radar_chart(survey_tuple, selected_school_name, dark_mode)
                st.plotly_chart(radar_fig, use_container_width=True)
                get_figure_download_link(radar_fig, "radar_chart.html", "Download Radar Chart")
                st.caption("The distance from the centre (0) to each milestone point represents the "
                           "strength of that milestone. The golden arrow indicates the clockwise "
                           "milestone cycle direction.")
        else:
            st.info("No survey data for current quarter.")

        # Research Outputs Dashboard (data cached, rendering theme-aware)
        with st.expander("Research Outputs Dashboard (for selected school)"):
            metrics = _compute_research_metrics(metadata_df, selected_school_id)
            figs = render_research_dashboard(metrics, selected_school_name, dark_mode)
            if figs:
                for name, fig in figs.items():
                    st.plotly_chart(fig, use_container_width=True)
                    get_figure_download_link(fig, f"{name}.html", f"Download {name}")
                st.metric("School-level Research Utilisation Rate",
                          f"{metrics.get('util_rate', 0):.1f}% -> "
                          f"{metrics.get('util_level', 'N/A')} level",
                          help=metrics.get('util_desc', ''))
                if metrics.get('top_teacher') != "N/A":
                    st.caption(f"Top teacher: {metrics['top_teacher']}")
            else:
                st.info("No data available for this school.")

        # Baseline Synopsis
        if baseline_btn:
            latest_row = get_latest_survey(survey_df, selected_school_id)
            if latest_row is not None:
                st.session_state.baseline_synopsis = generate_baseline_synopsis(
                    latest_row, selected_school_name, metadata_df
                )
                st.session_state.baseline_survey_row = latest_row.to_dict()
                # Compute baseline std devs for significance testing (BUG FIX: was missing)
                school_survey = survey_df[survey_df['school_id_no'] == selected_school_id]
                st.session_state.baseline_std_devs = {
                    v: (school_survey[v].std() if len(school_survey) > 1 else 0.1)
                    for v in VARIABLES
                }
            else:
                st.warning("No survey data available to generate baseline.")

        if 'baseline_synopsis' in st.session_state:
            bs = st.session_state.baseline_synopsis
            st.markdown("### Baseline Synopsis")
            bg_color = '#2E2E2E' if dark_mode else '#E3F2FD'
            text_col = DARK_TEXT if dark_mode else 'inherit'
            st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 5px solid {USTP_GOLD};
                        padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
            <b>School: {selected_school_name}</b><br>
            <b>Baseline RCSI:</b> {bs['baseline_rcsi']:.3f}<br>
            <b>Strengths (>=0.6):</b> {', '.join(bs['strengths']) if bs['strengths'] else 'None'}<br>
            <b>Critical Gaps (<=0.3):</b> {', '.join(bs['gaps']) if bs['gaps'] else 'None'}<br>
            <b>Moderate (0.3-0.6):</b> {', '.join(bs['moderate']) if bs['moderate'] else 'None'}<br>
            <b>Actionable Recommendations:</b><br>
            {'<br>'.join(bs['recommendations'])}
            </div>
            """, unsafe_allow_html=True)

        # Baseline Heatmap
        baseline_heatmap(survey_df, metadata_df, dark_mode)


        # --- Phase 2: Coefficient Calibration ---
        if 'calibrated_coeff' not in st.session_state:
            with st.spinner("Calibrating model coefficients from survey data..."):
                coeff, calib_msg = calibrate_coefficients(survey_df)
                if coeff:
                    st.success(calib_msg)
                    st.session_state.calibrated_coeff = coeff
                else:
                    st.warning(calib_msg)
                    st.session_state.calibrated_coeff = None

        # --- Initialize simulation (Phase 2: calibratable) ---
        if 'sim' not in st.session_state:
            st.session_state.sim = init_calibratable_simulation(
                school_ids, survey_df, metadata_df,
                st.session_state.get('calibrated_coeff'),
                random_events,
            )
            st.session_state.current_month = 0
            st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)

        # --- Run / Step / Reset (Phase 2: uses CalibratableSimulation) ---
        if run_btn:
            st.session_state.sim = init_calibratable_simulation(
                school_ids, survey_df, metadata_df,
                st.session_state.get('calibrated_coeff'),
                random_events,
            )
            st.session_state.current_month = 0
            st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)

            progress = st.progress(0, text="Running simulation...")
            for m in range(1, duration + 1):
                target_month = m
                if use_survey:
                    apply_survey_override(st.session_state.sim.agents, survey_df, target_month)
                st.session_state.sim.step(levers, target_month)
                st.session_state.current_month = target_month
                st.session_state.total_months = target_month
                record_history(st.session_state.history, st.session_state.sim.agents, target_month)
                progress.progress(m / duration, text=f"Month {m}/{duration}")

            if mc_enabled:
                agent_params = get_agent_params(
                    school_ids, survey_df, metadata_df,
                    st.session_state.get('calibrated_coeff'),
                )
                with st.spinner(f"Running {mc_runs} Monte Carlo simulations..."):
                    mc_data = monte_carlo_sim(
                        mc_runs, agent_params, levers, duration,
                        use_survey, survey_df, metadata_df,
                        selected_school_id, school_ids,
                    )
                    st.session_state.mc_data = mc_data
            else:
                st.session_state.pop('mc_data', None)

            progress.empty()
            st.rerun()

        if step_btn:
            target_month = st.session_state.current_month + 1
            if use_survey:
                apply_survey_override(st.session_state.sim.agents, survey_df, target_month)
            st.session_state.sim.step(levers, target_month)
            st.session_state.current_month = target_month
            st.session_state.total_months = target_month
            record_history(st.session_state.history, st.session_state.sim.agents, target_month)
            st.rerun()

        if reset_btn:
            st.session_state.sim = init_calibratable_simulation(
                school_ids, survey_df, metadata_df,
                st.session_state.get('calibrated_coeff'),
                random_events,
            )
            st.session_state.current_month = 0
            st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)
            st.session_state.pop('mc_data', None)
            st.rerun()
        # --- Display Simulation Results ---
        if st.session_state.total_months > 0:
            st.markdown("<h2 style='text-align: center;'>Simulated Data</h2>",
                         unsafe_allow_html=True)
            st.markdown("---")
            hist = st.session_state.history.get(selected_school_id)
            agent = next(
                (a for a in st.session_state.sim.agents if a.real_id == selected_school_id),
                None
            )

            if hist and agent:
                # Main 4-panel chart
                fig1 = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Variable Evolution", "Milestone Progress",
                                    "Research Culture Sustainability Index (RCSI)",
                                    "Improvement per Completed Cycle")
                )
                for i, var in enumerate(VARIABLES):
                    fig1.add_trace(go.Scatter(
                        x=hist['month'], y=hist[var], mode='lines', name=var,
                        line=dict(color=VAR_COLORS[i])
                    ), row=1, col=1)
                fig1.add_trace(go.Scatter(
                    x=hist['month'], y=hist['milestone'], mode='lines', name='Milestone',
                    line=dict(color=DEPED_RED, width=3)
                ), row=1, col=2)
                fig1.add_trace(go.Scatter(
                    x=hist['month'], y=hist['running_outcome'], mode='lines', name='RCSI',
                    line=dict(color=USTP_GOLD, width=3)
                ), row=2, col=1)
                if agent.cycle_improvements:
                    fig1.add_trace(go.Bar(
                        x=[c.cycle_number for c in agent.cycle_improvements],
                        y=[c.total_improvement for c in agent.cycle_improvements],
                        name='RCSI per cycle', marker_color=USTP_DARK_BLUE
                    ), row=2, col=2)
                else:
                    fig1.add_annotation(
                        text="No cycles completed yet",
                        xref="x2 domain", yref="y2 domain",
                        x=0.5, y=0.5, showarrow=False, row=2, col=2
                    )

                text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
                template = 'plotly_dark' if dark_mode else 'plotly_white'
                fig1.update_layout(height=800, showlegend=True,
                                    font=dict(color=text_color), template=template)
                fig1.update_xaxes(title_text="Month", row=1, col=1)
                fig1.update_yaxes(title_text="Value (0-1)", row=1, col=1)
                fig1.update_xaxes(title_text="Month", row=1, col=2)
                fig1.update_yaxes(title_text="Milestone", row=1, col=2)
                fig1.update_xaxes(title_text="Month", row=2, col=1)
                fig1.update_yaxes(title_text="RCSI", row=2, col=1)
                fig1.update_xaxes(title_text="Cycle Number", row=2, col=2)
                fig1.update_yaxes(title_text="RCSI", row=2, col=2)
                st.plotly_chart(fig1, use_container_width=True)
                get_figure_download_link(fig1, "simulation_overview.html",
                                         "Download Simulation Charts")

                with st.expander("Cycle vs Research Outputs"):
                    cycle_research_correlation(agent, metadata_df, selected_school_id, dark_mode)

                with st.expander("Division-Level Analysis"):
                    div_metrics = division_level_analysis(
                        metadata_df, st.session_state.history,
                        st.session_state.sim.agents, dark_mode
                    )

                with st.expander("Comparative School Analysis"):
                    selected_comparison = st.multiselect(
                        "Select schools to compare (choose at least two)",
                        options=school_ids, default=[],
                        format_func=lambda x: id_to_label.get(x, f"ID {x}")
                    )
                    school_comparison_dashboard(
                        survey_df, st.session_state.history, school_info,
                        selected_comparison, dark_mode
                    )

                # RCSI Interpretation Table

                # --- Phase 2: Sensitivity Analysis ---
                with st.expander("Sensitivity Analysis (Tornado Chart)"):
                    with st.spinner("Computing sensitivity of final RCSI to each policy lever..."):
                        agent_params_sens = get_agent_params(
                            school_ids, survey_df, metadata_df,
                            st.session_state.get('calibrated_coeff'),
                        )
                        fig_tornado = run_sensitivity(
                            agent_params_sens, levers, duration,
                            use_survey, survey_df, metadata_df,
                            selected_school_id, school_ids,
                        )
                        st.plotly_chart(fig_tornado, use_container_width=True)
                        get_figure_download_link(fig_tornado, "sensitivity_tornado.html",
                                                 "Download Sensitivity Analysis")
                        st.caption("Each policy lever is varied +-10% while all others remain "
                                   "at their current slider values. The bar length shows the "
                                   "resulting change in final RCSI.")

                # --- Phase 2: Monte Carlo Uncertainty Bands ---
                if 'mc_data' in st.session_state:
                    with st.expander("Monte Carlo Uncertainty Bands"):
                        mc_data = st.session_state.mc_data
                        fig_mc = plot_monte_carlo_bands(mc_data, dark_mode)
                        st.plotly_chart(fig_mc, use_container_width=True)
                        get_figure_download_link(fig_mc, "monte_carlo_bands.html",
                                                 "Download Monte Carlo Chart")
                        st.caption(
                            f"Shaded area represents the P10-P90 range over "
                            f"{mc_runs} Monte Carlo simulations. The solid line "
                            f"is the median outcome."
                        )

                        # --- Phase 2: Causal Impact Analysis ---
                        causal_coeffs = causal_analysis(mc_data)
                        if causal_coeffs:
                            r_sq = causal_coeffs.pop('_r_squared', None)
                            st.markdown("**Causal Impact Analysis**")
                            st.caption(
                                "Estimates how much a one-unit increase in each "
                                "baseline variable increases the final RCSI, based "
                                "on variation across Monte Carlo runs."
                            )
                            causal_df = pd.DataFrame(
                                list(causal_coeffs.items()),
                                columns=['Variable', 'Impact on Final RCSI'],
                            )
                            causal_df['Variable'] = causal_df['Variable'].map(
                                lambda v: VAR_FULL_NAMES.get(v, v)
                            )
                            st.dataframe(causal_df)
                            if r_sq is not None:
                                st.caption(f"Model R-squared: {r_sq:.3f}")
                        else:
                            if not SKLEARN_AVAILABLE:
                                st.info("scikit-learn is required for causal analysis.")
                            else:
                                st.info("Not enough Monte Carlo runs or insufficient "
                                        "variance for causal analysis (need >10 runs).")
                st.markdown("### Research Culture Sustainability Index (RCSI) Interpretation Table")
                st.markdown("""
                | RCSI Range | Level | Description |
                |------------|-------|-------------|
                | 0.0 - 0.2 | Very Low | Little to no accumulated research culture strength. |
                | 0.2 - 0.4 | Low | Minimal ecosystem vitality; research culture still weak. |
                | 0.4 - 0.6 | Moderate | Noticeable strength; research culture developing. |
                | 0.6 - 0.8 | High | Strong ecosystem; research culture becoming sustainable. |
                | 0.8 - 1.0 | Very High | Excellent vitality; research culture fully embedded. |
                """)

                # --- Simulation Synopsis (selected school) ---
                rcsi_val = agent.running_total_outcome
                rcsi_level = classify_rcsi(rcsi_val)
                milestone_name = MILESTONE_NAMES.get(
                    agent.current_milestone, f"Milestone {agent.current_milestone}"
                )

                if agent.cycle_count >= 2:
                    cycle_text = (f"has completed {agent.cycle_count} full cycles, "
                                  f"indicating a self-sustaining research culture.")
                elif agent.cycle_count == 1:
                    cycle_text = "has completed one full cycle, demonstrating initial sustainability."
                else:
                    cycle_text = "has not yet completed any full cycle."

                if agent.current_milestone == 0:
                    milestone_progress = "is at the very beginning of the journey."
                elif agent.current_milestone <= 2:
                    milestone_progress = ("has moved beyond initial readiness but remains "
                                          "in early capacity-building phases.")
                elif agent.current_milestone <= 4:
                    milestone_progress = ("has established structured support and is embedding "
                                          "research into institutional practice.")
                else:
                    milestone_progress = ("is realising tangible impact and is approaching or "
                                          "has achieved cyclical sustainability.")

                key_R = hist['R'][-1] if hist['R'] else 0
                key_M = hist['M'][-1] if hist['M'] else 0

                output_trend_text = ""
                tl = metrics.get('output_timeline')
                if tl is not None and len(tl) >= 2:
                    if tl.iloc[-1]['count'] > tl.iloc[-2]['count']:
                        output_trend_text = "Research output is increasing over time."
                    elif tl.iloc[-1]['count'] < tl.iloc[-2]['count']:
                        output_trend_text = "Research output is declining over time."
                    else:
                        output_trend_text = "Research output has remained stable."
                    avg_output = tl['count'].mean()
                    output_trend_text += f" On average, the school produces {avg_output:.1f} outputs per quarter."

                theme_util_text = ""
                tu = metrics.get('theme_util_df')
                if tu is not None and not tu.empty:
                    max_util = tu.loc[tu['Utilisation Rate'].idxmax()]
                    min_util = tu.loc[tu['Utilisation Rate'].idxmin()]
                    theme_util_text = (f"The most utilised theme is '{max_util['Theme']}' "
                                       f"({max_util['Utilisation Rate']:.0%}), while "
                                       f"'{min_util['Theme']}' has the lowest adoption "
                                       f"({min_util['Utilisation Rate']:.0%}).")

                top_teacher_text = (
                    f"The school's top researcher is {metrics['top_teacher']}."
                    if metrics.get('top_teacher') != "N/A" else ""
                )

                bg_color = '#2E2E2E' if dark_mode else '#E3F2FD'
                text_col = DARK_TEXT if dark_mode else 'inherit'
                coherent_text = f"""
                After {st.session_state.total_months} months, {selected_school_name}
                (ID {selected_school_id}) has reached {milestone_name} and {cycle_text}
                The school's Research Culture Sustainability Index (RCSI) is
                <b>{rcsi_val:.3f}</b>, which falls into the <b>{rcsi_level}</b> level.
                Key indicators: Readiness (R) = {key_R:.2f}, Impact (M) = {key_M:.2f},
                and current Milestone = {agent.current_milestone}.
                This combination suggests that {milestone_progress}
                The RCSI level <b>{rcsi_level.lower()}</b> reinforces this assessment.
                {output_trend_text}
                {theme_util_text}
                {top_teacher_text}
                Overall, the school is on a path toward research culture sustainability,
                but further policy support may be needed.
                """
                st.markdown(f"""
                <div style="background-color: {bg_color}; border-left: 5px solid {USTP_GOLD};
                            padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                <b>School {selected_school_id} ({selected_school_name}) - Simulation Synopsis</b><br>
                {coherent_text}
                </div>
                """, unsafe_allow_html=True)

                # Baseline vs Simulation Comparison
                if ('baseline_synopsis' in st.session_state and
                        'baseline_survey_row' in st.session_state):
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
                            status = ("Improving" if diff > 0.01
                                      else ("Regressing" if diff < -0.01 else "Stable"))
                            std_dev = baseline_std_devs.get(var, 0.1)
                            if abs(diff) >= 0.10:
                                significance = "Both statistically and practically significant"
                            elif abs(diff) >= 0.5 * std_dev:
                                significance = ("Statistically significant, but limited "
                                               "practical impact")
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

                # --- Division-Level Synopsis ---
                total_schools = len(st.session_state.sim.agents)
                early_stage_count = sum(
                    1 for a in st.session_state.sim.agents if a.current_milestone <= 2
                )
                advanced_stage_count = sum(
                    1 for a in st.session_state.sim.agents if a.current_milestone >= 4
                )
                # BUG FIX: transitional_percent was undefined in original
                transitional_count = total_schools - early_stage_count - advanced_stage_count
                early_percent = ((early_stage_count / total_schools) * 100
                                 if total_schools > 0 else 0)
                advanced_percent = ((advanced_stage_count / total_schools) * 100
                                    if total_schools > 0 else 0)
                transitional_percent = ((transitional_count / total_schools) * 100
                                        if total_schools > 0 else 0)

                early_text = (f"{early_percent:.1f}% of schools"
                              if early_percent > 0 else "No schools")
                advanced_text = (f"{advanced_percent:.1f}% of schools"
                                 if advanced_percent > 0 else "No schools")

                if early_percent == 100:
                    sustainability_text = ("All schools are in early milestones; "
                                           "foundational capacity-building is the priority.")
                elif early_percent >= 75:
                    sustainability_text = (f"The vast majority ({early_percent:.1f}%) are in "
                                           f"early milestones; urgent interventions needed.")
                elif early_percent >= 50:
                    sustainability_text = (f"More than half ({early_percent:.1f}%) are in early "
                                           f"milestones; targeted policy support may accelerate progress.")
                elif early_percent > 0:
                    sustainability_text = (f"{early_percent:.1f}% remain in early milestones; "
                                           f"continued efforts are required.")
                else:
                    sustainability_text = ("No schools are in early milestones; the division "
                                           "exhibits a strong, advanced research culture.")

                total_outcome = sum(a.running_total_outcome for a in st.session_state.sim.agents)
                avg_rcsi = total_outcome / total_schools if total_schools > 0 else 0
                level_avg = classify_rcsi(avg_rcsi)
                total_cycles = sum(a.cycle_count for a in st.session_state.sim.agents)
                avg_milestone = np.mean([a.current_milestone for a in st.session_state.sim.agents])
                avg_milestone_interp = interpret_avg_milestone(avg_milestone)

                school_ids_in_sim = [a.real_id for a in st.session_state.sim.agents]
                div_metadata = metadata_df[metadata_df['school_id_no'].isin(school_ids_in_sim)]
                total_utilised = (div_metadata['utilized_by_school'].sum()
                                  if 'utilized_by_school' in div_metadata.columns else 0)
                total_research_outputs = len(div_metadata)
                div_util_rate = ((total_utilised / total_research_outputs * 100)
                                 if total_research_outputs > 0 else 0)

                div_insights = div_metrics if 'div_metrics' in dir() else {}
                top_div_teacher = div_insights.get('top_div_teacher', 'N/A')
                top_div_school = div_insights.get('top_div_school', 'N/A')
                top_div_outputs = div_insights.get('top_div_outputs', 0)
                bottleneck_milestone = div_insights.get('bottleneck_milestone', 'N/A')
                bottleneck_time = div_insights.get('bottleneck_time', 0)

                # Division output trend
                output_trend_div = ""
                if not metadata_df.empty and 'upload_date' in metadata_df.columns:
                    div_timeline = metadata_df.groupby(
                        metadata_df['upload_date'].dt.to_period('Q')
                    ).size()
                    if len(div_timeline) >= 2:
                        if div_timeline.iloc[-1] > div_timeline.iloc[-2]:
                            output_trend_div = "The division's research output is increasing over time."
                        elif div_timeline.iloc[-1] < div_timeline.iloc[-2]:
                            output_trend_div = "The division's research output is declining over time."
                        else:
                            output_trend_div = "The division's research output has remained stable."
                        avg_div_output = div_timeline.mean()
                        output_trend_div += (f" On average, the division produces "
                                             f"{avg_div_output:.1f} outputs per quarter.")
                    else:
                        output_trend_div = "Division output trend data is limited."

                full_bottleneck = MILESTONE_NAMES.get(
                    int(bottleneck_milestone.replace('M', ''))
                    if isinstance(bottleneck_milestone, str) and bottleneck_milestone.startswith('M')
                    else 0,
                    bottleneck_milestone
                )
                bottleneck_insight = (
                    f"Schools spend the most time on average in {full_bottleneck} "
                    f"({bottleneck_time:.1f} months). This is the critical bottleneck."
                    if bottleneck_milestone != "N/A" else ""
                )
                top_teacher_insight = (
                    f"The division's top researcher is {top_div_teacher} from "
                    f"{top_div_school} with {top_div_outputs} outputs."
                    if top_div_teacher != "N/A" else ""
                )

                bg_color_div = '#2E2E2E' if dark_mode else '#E8F5E9'
                st.markdown(f"""
                <div style="background-color: {bg_color_div}; border-left: 5px solid {USTP_GOLD};
                            padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                <b>Division-Level Sustainability Synopsis (all {total_schools} schools)</b><br>
                - Average milestone = {avg_milestone:.1f} - {avg_milestone_interp}<br>
                - Total completed cycles = {total_cycles}<br>
                - Average RCSI = <b>{avg_rcsi:.3f}</b> - <b>{level_avg}</b> level.<br>
                - Average research utilisation rate = <b>{div_util_rate:.1f}%</b>.<br>
                - Stage distribution: {early_text} are in early stages (M<=2),
                  {transitional_percent:.1f}% transitional (M3),
                  and {advanced_text} are advanced (M>=4).<br>
                <i>Division-wide sustainability assessment:</i> {sustainability_text}<br><br>
                <b>Productivity:</b> {output_trend_div}<br>
                <b>Bottleneck:</b> {bottleneck_insight}<br>
                <b>Top Division Researcher:</b> {top_teacher_insight}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Graph Interpretations"):
                    st.markdown("""
                    - **Variable Evolution:** Shows how R, A, C, S, I, P, M change over time.
                      Higher values (closer to 1) mean stronger readiness, awareness, capacity, etc.
                    - **Milestone Progress:** The school moves through milestones 0-6. Reaching
                      milestone 6 and cycling back indicates a full sustainable cycle.
                    - **Research Culture Sustainability Index (RCSI):** Cumulative strength of
                      the research ecosystem, derived from Impact Realization (M) and
                      Collaboration (P).
                    - **Improvement per Completed Cycle:** Each bar shows the RCSI contributed by
                      one cycle. Higher bars in later cycles indicate increasing effectiveness.
                    - **Radar Chart:** Current snapshot of the seven milestone-linked variables.
                    - **Research Outputs Dashboard:** Tracks themes, publication status,
                      utilisation, teacher productivity, experience vs output, timeline,
                      top teachers, and breakdown by rank and attainment.
                    - **Division-Level Analysis:** Milestone transition bottlenecks and
                      teacher leaderboard.
                    - **Comparative Analysis:** Overlay multiple schools' RCSI and milestone
                      progress.
                    - **Cycle vs Research Outputs:** Shows how research output accumulation
                      relates to cycle progression.
                    """)

            # Export button
            if export_btn:
                all_data = []
                for agent in st.session_state.sim.agents:
                    h = st.session_state.history[agent.real_id]
                    for t in range(len(h['month'])):
                        row = {
                            'school_id': agent.real_id, 'month': h['month'][t],
                            'milestone': h['milestone'][t],
                            'running_outcome': h['running_outcome'][t]
                        }
                        for var in VARIABLES:
                            row[var] = h[var][t]
                        all_data.append(row)
                df_hist = pd.DataFrame(all_data)

                cycle_records = []
                for agent in st.session_state.sim.agents:
                    for rec in agent.cycle_improvements:
                        cycle_records.append({
                            'school_id': agent.real_id,
                            'cycle_number': rec.cycle_number,
                            'total_improvement': rec.total_improvement,
                            'completion_month': rec.completion_month
                        })
                df_cycles = pd.DataFrame(cycle_records)

                # BUG FIX: proper keyword args (missing closing paren in original)
                st.download_button(
                    "Download simulation history",
                    data=df_hist.to_csv(index=False).encode('utf-8'),
                    file_name="simulation_history.csv",
                    mime="text/csv"
                )
                st.download_button(
                    "Download cycle improvements",
                    data=df_cycles.to_csv(index=False).encode('utf-8'),
                    file_name="cycle_improvements.csv",
                    mime="text/csv"
                )
else:
    st.info("Please upload quarterly survey and research metadata CSV files to begin.")
