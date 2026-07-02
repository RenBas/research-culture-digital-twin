# ============================================================
# constants.py – all global configuration
# ============================================================

# --- Colours ---
USTP_DARK_BLUE = "#0D2B5E"
USTP_GOLD = "#F5A623"
DEPED_RED = "#D32F2F"
DEPED_MAROON = "#8B0000"
LIGHT_BG = "#F8F9FA"
DARK_BG = "#1E1E1E"
DARK_TEXT = "#FFFFFF"
LIGHT_TEXT = "#000000"

# --- Simulation ---
BASE_YEAR = 2026
RANDOM_EVENT_PROB = 0.00417
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

# --- Milestone mappings ---
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
    'R': 'Readiness (R)', 'A': 'Awareness (A)', 'C': 'Capacity (C)',
    'S': 'Structured Support (S)', 'I': 'Institutional Anchoring (I)',
    'P': 'Community of Practice (P)', 'M': 'Impact Realization (M)',
}
VAR_INTERPRETATION = {
    'R': ['Very Low Readiness', 'Low Readiness', 'Moderate Readiness', 'High Readiness', 'Very High Readiness'],
    'A': ['Very Low Awareness', 'Low Awareness', 'Moderate Awareness', 'High Awareness', 'Very High Awareness'],
    'C': ['Very Low Capacity', 'Low Capacity', 'Moderate Capacity', 'High Capacity', 'Very High Capacity'],
    'S': ['Very Low Support', 'Low Support', 'Moderate Support', 'High Support', 'Very High Support'],
    'I': ['Very Low Anchoring', 'Low Anchoring', 'Moderate Anchoring', 'High Anchoring', 'Very High Anchoring'],
    'P': ['Very Low CoP', 'Low CoP', 'Moderate CoP', 'High CoP', 'Very High CoP'],
    'M': ['Very Low Impact', 'Low Impact', 'Moderate Impact', 'High Impact', 'Very High Impact'],
}
RCSI_INTERPRETATION = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

# --- Milestone transition thresholds ---
MILESTONE_THRESHOLDS = {
    0: ('A', 0.8, 1),
    1: ('C', 0.7, 2),
    2: ('S', 0.7, 3),
    3: ('I', 0.8, 4),
    4: ('P', 0.8, 5),
    5: ('M', 0.7, 6),
}

# --- RCSI level definitions ---
RCSI_LEVELS = [
    (0.0, 0.2, "Very Low"),
    (0.2, 0.4, "Low"),
    (0.4, 0.6, "Moderate"),
    (0.6, 0.8, "High"),
    (0.8, 1.0, "Very High"),
]

# --- CSV required columns ---
REQUIRED_SURVEY_COLS = ['month', 'school_id_no'] + VARIABLES
OPTIONAL_SURVEY_COLS = ['school_name']
REQUIRED_META_COLS = ['upload_date', 'teacher_name', 'school_id_no']
OPTIONAL_META_COLS = {
    'document_type': 'abstract', 'title': '', 'theme': 'Uncategorized',
    'status': 'unpublished', 'publication_link': '', 'utilized_by_school': False,
    'utilization_date': '', 'year_undertaken': 2025, 'years_of_service': None,
    'teacher_rank': None, 'educational_attainment': None,
}

# --- Colours for line charts ---
VAR_COLORS = ['#1E88E5', USTP_GOLD, '#8E44AD', '#2ECC71', '#E67E22', DEPED_RED, '#1ABC9C']
