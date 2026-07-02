# ============================================================
# utils.py – helper functions (classification, interpretation, date parsing)
# ============================================================

from typing import Tuple, Any
import numpy as np
from .constants import RCSI_LEVELS, BASE_YEAR

def classify_rcsi(value: float) -> str:
    for low, high, lev in RCSI_LEVELS:
        if low <= value < high:
            return lev
    return "Very High"

def get_rcsi_interpretation_table() -> str:
    return """
| RCSI Score Range | Risk/Stability Level | School-Level Policy Implication |
| :--- | :--- | :--- |
| **0.000 – 0.010** | **Optimal / Extremely Stable** | Maintain current strategies; focus on qualitative improvements (instruction, morale) rather than structural fixes. |
| **0.011 – 0.030** | Low Risk | Routine monitoring; allocate 5-10% of discretionary budget to preemptive padding. |
| **0.031 – 0.070** | Moderate Risk | Requires targeted intervention; conduct a root-cause analysis on the top 2 contributing variables. |
| **0.071 – 0.100** | High Risk | Immediate corrective action required; escalate to Division Office for shared resource support. |
| **> 0.100** | Critical / Unstable | Full operational review triggered; Division-level contingency protocols are activated. |
"""

def interpret_avg_milestone(avg_milestone: float) -> str:
    desc = [
        (0.5, "between M0 and M1"),
        (1.5, "between M1 and M2"),
        (2.5, "between M2 and M3"),
        (3.5, "between M3 and M4"),
        (4.5, "between M4 and M5"),
        (5.5, "between M5 and M6"),
        (float('inf'), "at or beyond M6"),
    ]
    for threshold, d in desc:
        if avg_milestone < threshold:
            return f"{avg_milestone:.1f} → {d}"
    return f"{avg_milestone:.1f} → at or beyond M6"

def interpret_utilisation_rate(rate: float) -> Tuple[str, str]:
    if rate < 20: return "Very Low", "Rarely adopted."
    elif rate < 40: return "Low", "Limited adoption."
    elif rate < 60: return "Moderate", "Half adopted."
    elif rate < 80: return "High", "Strong translation."
    else: return "Very High", "Excellent utilisation."

def classify_utilisation(rate: float) -> str:
    if rate < 20: return "Very Low"
    elif rate < 40: return "Low"
    elif rate < 60: return "Moderate"
    elif rate < 80: return "High"
    else: return "Very High"

def month_str_to_num(month_str: Any) -> int:
    try:
        parts = str(month_str).strip().split('-')
        if len(parts) == 2:
            return (int(parts[0]) - BASE_YEAR) * 12 + int(parts[1])
    except (ValueError, TypeError, AttributeError):
        pass
    return 0

def date_to_month_num(d: Any) -> int:
    try:
        return (d.year - BASE_YEAR) * 12 + d.month
    except (ValueError, TypeError, AttributeError):
        return 0

def create_glossary() -> dict:
    return {
        "R (Readiness)": "Measures the school's preparedness and foundational conditions for research, including infrastructure and mindset.",
        "A (Awareness)": "Indicates the level of research awareness among teachers and leaders. The threshold for M0→M1 is **A ≥ 0.8**.",
        "C (Capacity)": "Captures the research skills, training, and expertise of the teaching staff.",
        "S (Structured Support)": "Reflects the availability of budget, time, mentoring, and other institutional support for research.",
        "I (Institutional Anchoring)": "Measures how deeply research is embedded in school plans, policies, and regular meetings.",
        "P (Community of Practice)": "Evaluates the strength of research collaboration, sharing forums, and peer learning.",
        "M (Impact Realization)": "Tracks the tangible outcomes of research, such as publications, utilizations, and policy changes.",
        "RCSI (Research Culture Sustainability Index)": "A cumulative score that aggregates the seven variables, indicating overall sustainability. Higher is better.",
        "Milestone (M0–M6)": "A sequential progression from Readiness (M0) to Impact Realization (M6). Reaching M6 and cycling back indicates a sustainable research culture.",
        "Utilisation Rate": "Percentage of research outputs that have been used or applied by the school or division.",
        "Sensitivity Analysis": "Shows which policy lever (Training, Mentorship, Budget, Leadership, Collaboration) has the most influence on RCSI.",
        "Monte Carlo Simulation": "Generates multiple scenarios with random variations to estimate the range of possible RCSI outcomes.",
    }
