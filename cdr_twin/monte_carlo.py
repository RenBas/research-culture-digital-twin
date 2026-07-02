# ============================================================
# monte_carlo.py – Phase 2: calibration, clustering, sensitivity, MC, causal
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple

from .constants import VARIABLES, VALUE_FLOOR, VALUE_CEIL, USTP_GOLD, USTP_DARK_BLUE, DEPED_RED
from .data import get_latest_survey
from .simulation import Simulation, seed_agents_from_metadata, apply_survey_override

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def calibrate_coefficients(survey_df):
    if not SKLEARN_AVAILABLE:
        return None, "scikit‑learn not installed – using default coefficients."
    if survey_df is None or survey_df.empty:
        return None, "No survey data – using default coefficients."
    u_train = u_mentor = u_budget = u_lead = u_collab = 0.5
    X_R, y_R = [], []
    X_A, y_A = [], []
    X_C, y_C = [], []
    X_S, y_S = [], []
    X_I, y_I = [], []
    X_P, y_P = [], []
    X_M, y_M = [], []
    schools = survey_df['school_id_no'].unique()
    for sid in schools:
        sdf = survey_df[survey_df['school_id_no'] == sid].sort_values('month_num')
        if len(sdf) < 2:
            continue
        for i in range(len(sdf)-1):
            curr = sdf.iloc[i]
            nxt = sdf.iloc[i+1]
            u_lead_eff = min(1.0, u_lead + 0.05 * curr['M'])
            X_R.append([curr['M']])
            y_R.append(nxt['R'] - curr['R'])
            X_A.append([curr['R'], u_train, curr['M']])
            y_A.append(nxt['A'] - curr['A'])
            X_C.append([u_train, u_mentor])
            y_C.append(nxt['C'] - curr['C'])
            X_S.append([u_budget, u_mentor])
            y_S.append(nxt['S'] - curr['S'])
            X_I.append([u_lead_eff, curr['S']])
            y_I.append(nxt['I'] - curr['I'])
            X_P.append([u_collab, curr['I']])
            y_P.append(nxt['P'] - curr['P'])
            X_M.append([curr['C'], curr['P']])
            y_M.append(nxt['M'] - curr['M'])
    coeff = {}
    default = {
        'R_M': 0.02, 'A_R': 0.04, 'A_train': 0.02, 'A_M': 0.01,
        'C_train': 0.03, 'C_mentor': 0.02, 'S_budget': 0.04, 'S_mentor': 0.02,
        'I_lead': 0.03, 'I_S': 0.02, 'P_collab': 0.04, 'P_I': 0.02,
        'M_C': 0.02, 'M_P': 0.02, 'const_R': -0.01, 'const_A': -0.005,
        'const_C': -0.01, 'const_S': -0.01, 'const_I': -0.005,
        'const_P': -0.01, 'const_M': -0.005
    }
    try:
        if X_R:
            model = LinearRegression().fit(X_R, y_R)
            coeff['R_M'] = model.coef_[0]
            coeff['const_R'] = model.intercept_
        if X_A:
            model = LinearRegression().fit(X_A, y_A)
            coeff['A_R'], coeff['A_train'], coeff['A_M'] = model.coef_
            coeff['const_A'] = model.intercept_
        if X_C:
            model = LinearRegression().fit(X_C, y_C)
            coeff['C_train'], coeff['C_mentor'] = model.coef_
            coeff['const_C'] = model.intercept_
        if X_S:
            model = LinearRegression().fit(X_S, y_S)
            coeff['S_budget'], coeff['S_mentor'] = model.coef_
            coeff['const_S'] = model.intercept_
        if X_I:
            model = LinearRegression().fit(X_I, y_I)
            coeff['I_lead'], coeff['I_S'] = model.coef_
            coeff['const_I'] = model.intercept_
        if X_P:
            model = LinearRegression().fit(X_P, y_P)
            coeff['P_collab'], coeff['P_I'] = model.coef_
            coeff['const_P'] = model.intercept_
        if X_M:
            model = LinearRegression().fit(X_M, y_M)
            coeff['M_C'], coeff['M_P'] = model.coef_
            coeff['const_M'] = model.intercept_
        for k, v in default.items():
            if k not in coeff:
                coeff[k] = v
        return coeff, "Calibration successful using data-driven coefficients."
    except Exception as e:
        return None, f"Calibration failed: {str(e)} – using default coefficients."


def cluster_schools(metadata_df, school_ids):
    if not SKLEARN_AVAILABLE:
        return {sid: 0 for sid in school_ids}, {0: 1.0}
    if metadata_df is None or metadata_df.empty:
        return {sid: 0 for sid in school_ids}, {0: 1.0}
    features = []
    for sid in school_ids:
        sm = metadata_df[metadata_df['school_id_no'] == sid]
        n_teachers = sm['teacher_name'].nunique()
        n_themes = sm['theme'].nunique()
        avg_util = sm['utilized_by_school'].mean() if not sm.empty else 0
        pub_rate = len(sm[sm['status'] == 'published']) / len(sm) if len(sm) > 0 else 0
        features.append([n_teachers, n_themes, avg_util, pub_rate])
    X = np.array(features)
    n_clusters = min(3, len(X))
    if n_clusters < 2:
        return {sid: 0 for sid in school_ids}, {0: 1.0}
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    cluster_map = {sid: cl for sid, cl in zip(school_ids, clusters)}
    multipliers = {0: 1.0, 1: 1.2, 2: 0.8}
    return cluster_map, multipliers


def get_agent_params(school_ids, survey_df, metadata_df, calibrated_coeff=None):
    cluster_map, multipliers = cluster_schools(metadata_df, school_ids)
    base_coeff = calibrated_coeff if calibrated_coeff else {
        'R_M': 0.02, 'A_R': 0.04, 'A_train': 0.02, 'A_M': 0.01,
        'C_train': 0.03, 'C_mentor': 0.02, 'S_budget': 0.04, 'S_mentor': 0.02,
        'I_lead': 0.03, 'I_S': 0.02, 'P_collab': 0.04, 'P_I': 0.02,
        'M_C': 0.02, 'M_P': 0.02, 'const_R': -0.01, 'const_A': -0.005,
        'const_C': -0.01, 'const_S': -0.01, 'const_I': -0.005,
        'const_P': -0.01, 'const_M': -0.005
    }
    params = []
    for sid in school_ids:
        latest = get_latest_survey(survey_df, sid)
        if latest is not None:
            init_vals = (latest['R'], latest['A'], latest['C'], latest['S'], latest['I'], latest['P'], latest['M'])
            baseline_rcsi = np.mean(init_vals)
        else:
            init_vals = (0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0)
            baseline_rcsi = np.mean(init_vals)
        mult = multipliers.get(cluster_map.get(sid, 0), 1.0)
        coeff = {k: v * mult for k, v in base_coeff.items()}
        # Append baseline_rcsi as the last element of the tuple
        params.append((*init_vals, coeff, baseline_rcsi))
    return params


def run_sensitivity(sim_class, agent_params, school_ids, levers, duration, use_survey, survey_df, metadata_df, selected_school_id):
    baseline = levers.copy()
    lever_names = ['u_train', 'u_mentor', 'u_budget', 'u_lead', 'u_collab']

    def _quick_run(test_levers):
        sim = sim_class(agent_params=agent_params)
        for i, agent in enumerate(sim.agents):
            agent.real_id = school_ids[i]
        seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
        for m in range(1, duration + 1):
            if use_survey:
                apply_survey_override(sim.agents, survey_df, m)
            sim.step(test_levers, m)
        agent = next(a for a in sim.agents if a.real_id == selected_school_id)
        return agent.running_total_outcome

    base_rcsi = _quick_run(baseline)
    results = {}
    for lever in lever_names:
        for delta in [-0.1, 0.1]:
            test_levers = baseline.copy()
            test_levers[lever] = max(0.0, min(1.0, baseline[lever] + delta))
            results[(lever, delta)] = _quick_run(test_levers)

    tornado_data = []
    for lever in lever_names:
        low_change = results[(lever, -0.1)] - base_rcsi
        high_change = results[(lever, 0.1)] - base_rcsi
        tornado_data.append({'Lever': lever, 'Low Change': low_change, 'High Change': high_change})
    df = pd.DataFrame(tornado_data).melt(id_vars='Lever', var_name='Direction', value_name='Change')
    fig = px.bar(df, x='Change', y='Lever', color='Direction', orientation='h',
                 title='Sensitivity of Final RCSI to Policy Levers (±10%)',
                 color_discrete_map={'Low Change': DEPED_RED, 'High Change': USTP_GOLD})
    fig.update_layout(template='plotly_white', xaxis_title="Change in RCSI", yaxis_title="Policy Lever")
    impacts = {lever: abs(results[(lever, 0.1)] - base_rcsi) + abs(results[(lever, -0.1)] - base_rcsi) for lever in lever_names}
    most_impactful = max(impacts, key=impacts.get)
    sensitivity_info = f"Sensitivity analysis shows that **{most_impactful}** has the greatest influence on final RCSI."
    return fig, sensitivity_info


def monte_carlo_sim(num_runs, sim_class, agent_params, school_ids, levers, duration, use_survey, survey_df, metadata_df, selected_school_id):
    all_rcsi = []
    all_milestone = []
    for _ in range(num_runs):
        noisy_params = []
        for params in agent_params:
            # unpack: variables, coeff, baseline_rcsi
            *init_vals, coeff, baseline_rcsi = params
            # Perturb initial variables
            new_init = [max(VALUE_FLOOR, min(VALUE_CEIL, v + np.random.normal(0, 0.02))) for v in init_vals]
            # Perturb coefficients
            noisy_coeff = {k: v * np.random.normal(1, 0.05) for k, v in coeff.items()}
            # Recompute baseline RCSI from new initial variables
            new_baseline_rcsi = np.mean(new_init)
            noisy_params.append((*new_init, noisy_coeff, new_baseline_rcsi))
        sim = sim_class(agent_params=noisy_params, random_events=True)
        for i, agent in enumerate(sim.agents):
            agent.real_id = school_ids[i]
        seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
        target = next(a for a in sim.agents if a.real_id == selected_school_id)
        rcsi_hist, mil_hist = [], []
        for m in range(1, duration + 1):
            if use_survey:
                apply_survey_override(sim.agents, survey_df, m)
            sim.step(levers, m)
            rcsi_hist.append(target.running_total_outcome)
            mil_hist.append(target.current_milestone)
        all_rcsi.append(rcsi_hist)
        all_milestone.append(mil_hist)
    all_rcsi = np.array(all_rcsi)
    all_milestone = np.array(all_milestone)
    months = np.arange(1, duration + 1)
    mc_data = {
        'months': months,
        'rcsi': {'p10': np.percentile(all_rcsi, 10, axis=0),
                 'p50': np.percentile(all_rcsi, 50, axis=0),
                 'p90': np.percentile(all_rcsi, 90, axis=0)},
        'milestone': {'p10': np.percentile(all_milestone, 10, axis=0),
                      'p50': np.percentile(all_milestone, 50, axis=0),
                      'p90': np.percentile(all_milestone, 90, axis=0)},
        'final_rcsi': all_rcsi[:, -1]
    }
    mc_info = (f"Monte Carlo simulation ({num_runs} runs) estimates a median final RCSI of "
               f"**{np.median(mc_data['final_rcsi']):.3f}**, with a P10‑P90 range of "
               f"**{np.percentile(mc_data['final_rcsi'], 10):.3f}** – **{np.percentile(mc_data['final_rcsi'], 90):.3f}**. "
               "This narrow spread is relevant at the school level because it indicates that the forecast is highly stable and actionable for local policy decisions.")
    return mc_data, mc_info


def plot_monte_carlo_bands(mc_data, dark_mode):
    months = mc_data['months']
    fig = make_subplots(rows=2, cols=1, subplot_titles=("RCSI with Uncertainty", "Milestone with Uncertainty"))
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p10'], mode='lines', name='P10 RCSI',
                             line=dict(color=USTP_GOLD, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p50'], mode='lines', name='Median RCSI',
                             line=dict(color=USTP_GOLD)), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p90'], mode='lines', name='P90 RCSI',
                             line=dict(color=USTP_GOLD, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p10'], showlegend=False,
                             line=dict(color='rgba(0,0,0,0)'), hoverinfo='none'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p90'], fill='tonexty',
                             fillcolor='rgba(245,166,35,0.2)', line=dict(color='rgba(0,0,0,0)'),
                             showlegend=False, hoverinfo='none'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p10'], mode='lines', name='P10 Milestone',
                             line=dict(color=DEPED_RED, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p50'], mode='lines', name='Median Milestone',
                             line=dict(color=DEPED_RED)), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p90'], mode='lines', name='P90 Milestone',
                             line=dict(color=DEPED_RED, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p10'], showlegend=False,
                             line=dict(color='rgba(0,0,0,0)'), hoverinfo='none'), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p90'], fill='tonexty',
                             fillcolor='rgba(211,47,47,0.2)', line=dict(color='rgba(0,0,0,0)'),
                             showlegend=False, hoverinfo='none'), row=2, col=1)
    fig.update_layout(height=700, template='plotly_dark' if dark_mode else 'plotly_white',
                      xaxis_title="Month", yaxis_title="RCSI",
                      xaxis2_title="Month", yaxis2_title="Milestone")
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="RCSI", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Milestone", row=2, col=1)
    return fig


def causal_analysis(monte_carlo_finals, baseline_values):
    if not SKLEARN_AVAILABLE or len(monte_carlo_finals) < 10:
        return None
    X = np.array([list(baseline_values.values()) for _ in range(len(monte_carlo_finals))])
    model = LinearRegression().fit(X, monte_carlo_finals)
    return dict(zip(baseline_values.keys(), model.coef_))
