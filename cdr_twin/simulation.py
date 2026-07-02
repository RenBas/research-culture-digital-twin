# ============================================================
# simulation.py – SchoolAgent, Simulation, and helper functions
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from .constants import (
    VARIABLES, VALUE_FLOOR, VALUE_CEIL, RANDOM_EVENT_PROB,
    LOSS_CHAMPION_PENALTY, FUNDING_BOOST, LEADERSHIP_PENALTY,
    CYCLE_R_BONUS, CYCLE_M_DECAY_FACTOR, CYCLE_M_MIN_AFTER_DECAY,
    CYCLE_BONUS_BASE, CYCLE_BONUS_M_SCALE, MONTHLY_OUTCOME_BASE,
    MILESTONE_MIN_MONTHS, MILESTONE_THRESHOLDS, MILESTONE_NAMES
)


@dataclass
class CycleRecord:
    cycle_number: int
    total_improvement: float
    completion_month: int


class SchoolAgent:
    def __init__(self, unique_id: int,
                 initial_R=0.3, initial_A=0.2, initial_C=0.2,
                 initial_S=0.1, initial_I=0.1, initial_P=0.1, initial_M=0.0,
                 coeff_dict: Optional[Dict[str, float]] = None,
                 random_events_enabled: bool = False,
                 initial_rcsi: float = 0.0):
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
        # Start cumulative RCSI from the baseline value
        self.running_total_outcome = initial_rcsi
        self.random_events_enabled = random_events_enabled
        self.model_time = 0
        self._rng = np.random.RandomState()
        if coeff_dict is None:
            self.coeff = {
                'R_M': 0.02, 'A_R': 0.04, 'A_train': 0.02, 'A_M': 0.01,
                'C_train': 0.03, 'C_mentor': 0.02, 'S_budget': 0.04, 'S_mentor': 0.02,
                'I_lead': 0.03, 'I_S': 0.02, 'P_collab': 0.04, 'P_I': 0.02,
                'M_C': 0.02, 'M_P': 0.02, 'const_R': -0.01, 'const_A': -0.005,
                'const_C': -0.01, 'const_S': -0.01, 'const_I': -0.005,
                'const_P': -0.01, 'const_M': -0.005
            }
        else:
            self.coeff = coeff_dict

    def apply_random_event(self):
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

    def step_individual(self, levers: Dict[str, float]):
        u_train, u_mentor, u_budget, u_lead, u_collab = levers.values()
        c = self.coeff
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

    def _update_milestone(self):
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

    def _complete_cycle(self):
        old_M = self.M
        self.R = min(VALUE_CEIL, self.R + CYCLE_R_BONUS)
        self.M = max(CYCLE_M_MIN_AFTER_DECAY, self.M * CYCLE_M_DECAY_FACTOR)
        bonus = CYCLE_BONUS_BASE + CYCLE_BONUS_M_SCALE * old_M
        self.running_total_outcome += bonus
        self.current_cycle_accumulator += bonus
        self.cycle_count += 1
        self.cycle_improvements.append(
            CycleRecord(cycle_number=self.cycle_count,
                        total_improvement=self.current_cycle_accumulator,
                        completion_month=self.model_time))
        self.current_cycle_accumulator = 0.0


class Simulation:
    def __init__(self, num_schools=1, random_events=False, agent_params=None):
        if agent_params:
            self.agents = []
            for params in agent_params:
                # Unpack: init variables, coefficients, and initial RCSI
                init_R, init_A, init_C, init_S, init_I, init_P, init_M, coeff, initial_rcsi = params
                self.agents.append(SchoolAgent(
                    unique_id=len(self.agents),
                    initial_R=init_R,
                    initial_A=init_A,
                    initial_C=init_C,
                    initial_S=init_S,
                    initial_I=init_I,
                    initial_P=init_P,
                    initial_M=init_M,
                    coeff_dict=coeff,
                    random_events_enabled=random_events,
                    initial_rcsi=initial_rcsi
                ))
        else:
            self.agents = [SchoolAgent(i, random_events_enabled=random_events) for i in range(num_schools)]

    def step(self, levers, month):
        for agent in self.agents:
            agent.model_time = month
            agent.step_individual(levers)

    def get_agent(self, idx=0):
        return self.agents[idx]


# --- Helpers for history, seeding, overriding ---
def create_empty_history(school_ids):
    return {sid: {var: [] for var in VARIABLES + ['month', 'milestone', 'running_outcome']}
            for sid in school_ids}


def seed_agents_from_metadata(agents, school_ids, metadata_df):
    for agent in agents:
        sm = metadata_df[metadata_df['school_id_no'] == agent.real_id]
        if sm.empty:
            continue
        agent.A = min(VALUE_CEIL, agent.A + len(sm[sm['document_type'] == 'abstract']) * 0.01)
        agent.M = min(VALUE_CEIL, agent.M + len(sm[sm['status'] == 'published']) * 0.02)
        agent.C = min(VALUE_CEIL, agent.C + len(sm[sm['document_type'] == 'full_paper']) * 0.005)
        agent.P = min(VALUE_CEIL, agent.P + sm['theme'].nunique() * 0.01)


def record_history(history, agents, total_months):
    for agent in agents:
        h = history[agent.real_id]
        h['month'].append(total_months)
        for var in VARIABLES:
            h[var].append(getattr(agent, var))
        h['milestone'].append(agent.current_milestone)
        h['running_outcome'].append(agent.running_total_outcome)


def apply_survey_override(agents, survey_df, target_month):
    for agent in agents:
        row = survey_df[(survey_df['school_id_no'] == agent.real_id) & (survey_df['month_num'] == target_month)]
        if not row.empty:
            r = row.iloc[0]
            for var in VARIABLES:
                setattr(agent, var, r[var])


def init_simulation_with_data(school_ids, metadata_df, random_events, agent_params=None):
    if agent_params:
        sim = Simulation(agent_params=agent_params, random_events=random_events)
    else:
        sim = Simulation(num_schools=len(school_ids), random_events=random_events)
    for idx, agent in enumerate(sim.agents):
        agent.real_id = school_ids[idx]
    seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
    return sim
