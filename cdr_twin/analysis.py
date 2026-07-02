# ============================================================
# analysis.py – higher-level analytical functions (synopsis, heatmap, etc.)
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from .constants import VARIABLES, VAR_FULL_NAMES, MILESTONE_NAMES, USTP_GOLD, USTP_DARK_BLUE, DEPED_RED
from .utils import classify_rcsi, interpret_avg_milestone, interpret_utilisation_rate, date_to_month_num
from .gauges import create_rcsi_gauge, create_gauge

def generate_baseline_synopsis(survey_row, school_name, _metadata_df):
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
        recommendations.append("All variables are at moderate or high levels. Maintain current policies.")
    return {'strengths': strengths, 'gaps': gaps, 'moderate': moderate,
            'baseline_rcsi': baseline_rcsi, 'recommendations': recommendations, 'values': values}

def baseline_heatmap(survey_df, metadata_df, dark_mode):
    st.markdown("### Historical Correlation Matrix (Diagnostic)")
    if survey_df is None or metadata_df is None:
        st.info("Insufficient data.")
        return
    survey_agg = survey_df.groupby(['school_id_no', 'month_num'])[VARIABLES].mean().reset_index()
    meta = metadata_df.copy()
    meta['month_num'] = meta['upload_date'].apply(date_to_month_num)
    output_counts = meta.groupby(['school_id_no', 'month_num']).size().reset_index(name='output_count')
    merged = survey_agg.merge(output_counts, on=['school_id_no', 'month_num'], how='inner')
    if merged.empty:
        st.info("Insufficient data.")
        return
    corr = merged[VARIABLES + ['output_count']].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale='Blues',
                    height=700, width=950)
    fig.update_layout(template='plotly_dark' if dark_mode else 'plotly_white',
                      xaxis_title="Variables", yaxis_title="Variables")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Correlation between the seven variables and research output count in historical data.")

def cycle_research_correlation(agent, metadata_df, school_id, dark_mode):
    if not agent.cycle_improvements:
        st.info("No cycles completed.")
        return
    sm = metadata_df[metadata_df['school_id_no'] == school_id].copy()
    sm['month_num'] = sm['upload_date'].apply(date_to_month_num)
    cumulative = [len(sm[sm['month_num'] <= rec.completion_month]) for rec in agent.cycle_improvements]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c.cycle_number for c in agent.cycle_improvements], y=cumulative,
                             mode='markers+lines', marker=dict(size=10, color=USTP_GOLD),
                             line=dict(color=USTP_DARK_BLUE)))
    fig.update_layout(template='plotly_dark' if dark_mode else 'plotly_white',
                      title="Cycle vs Cumulative Research Outputs",
                      xaxis_title="Cycle Number", yaxis_title="Cumulative Outputs")
    st.plotly_chart(fig, use_container_width=True)

def division_level_analysis(metadata_df, history_per_school, sim_agents, dark_mode):
    st.markdown("### Division-Level Analysis")
    if not metadata_df.empty:
        ts = metadata_df.groupby(['teacher_name', 'school_id_no']).size().reset_index(name='total_outputs')
        ts = ts.sort_values('total_outputs', ascending=False).head(20)
        st.dataframe(ts[['teacher_name', 'school_id_no', 'total_outputs']])
        top_div_teacher = ts.iloc[0]['teacher_name'] if not ts.empty else "N/A"
        top_div_school = ts.iloc[0]['school_id_no'] if not ts.empty else "N/A"
        top_div_outputs = ts.iloc[0]['total_outputs'] if not ts.empty else 0
    else:
        top_div_teacher = top_div_school = "N/A"; top_div_outputs = 0
    all_durations = {m: [] for m in range(7)}
    for sid, hist in history_per_school.items():
        if 'milestone' not in hist or not hist['milestone']:
            continue
        milestones = hist['milestone']
        for i in range(1, len(milestones)):
            if milestones[i] != milestones[i - 1]:
                start = milestones.index(milestones[i - 1], 0, i) if milestones[i - 1] in milestones[:i] else i - 1
                all_durations[milestones[i - 1]].append(i - start)
        if milestones:
            last = milestones[-1]
            start = milestones.index(last, 0, len(milestones)) if last in milestones else len(milestones) - 1
            all_durations[last].append(len(milestones) - start)
    avg_dur = {m: np.mean(v) if v else np.nan for m, v in all_durations.items()}
    df_dur = pd.DataFrame({'Milestone': [f'M{i}' for i in range(7)],
                           'Avg Months': [avg_dur.get(i, np.nan) for i in range(7)]}).dropna()
    bottleneck = "N/A"; bottleneck_time = 0
    if not df_dur.empty:
        max_row = df_dur.loc[df_dur['Avg Months'].idxmax()]
        bottleneck = max_row['Milestone']; bottleneck_time = max_row['Avg Months']
        fig_dur = px.bar(df_dur, x='Milestone', y='Avg Months', title="Average Months per Milestone",
                         color='Avg Months', color_continuous_scale=['#F5A623', '#0D2B5E'])
        fig_dur.update_layout(template='plotly_dark' if dark_mode else 'plotly_white',
                              xaxis_title="Milestone", yaxis_title="Average Months")
        st.plotly_chart(fig_dur, use_container_width=True)
        st.caption(f"Bottleneck: {bottleneck} ({bottleneck_time:.1f} months).")
    else:
        st.info("Not enough transition data.")
    return {'top_div_teacher': top_div_teacher, 'top_div_school': top_div_school,
            'top_div_outputs': top_div_outputs, 'bottleneck_milestone': bottleneck,
            'bottleneck_time': bottleneck_time}

def school_comparison_gauge(school_ids, school_info, history_per_school, dark_mode):
    st.markdown("### Comparative School Analysis (Gauges)")
    if len(school_ids) < 2:
        st.info("Select at least two schools.")
        return
    if len(school_ids) > 3:
        st.warning("Limit to 3 schools. Only first 3 will be shown.")
        school_ids = school_ids[:3]

    col_names = []
    for sid in school_ids:
        name = school_info[school_info['school_id_no'] == sid]['school_name'].values[0]
        col_names.append(name)

    cols = st.columns(len(school_ids))
    for idx, sid in enumerate(school_ids):
        with cols[idx]:
            hist = history_per_school.get(sid)
            if hist and hist.get('running_outcome'):
                rcsi = hist['running_outcome'][-1] if hist['running_outcome'] else 0.0
                milestone = hist['milestone'][-1] if hist['milestone'] else 0
                milestone_name = MILESTONE_NAMES.get(milestone, f"M{milestone}")
                fig = create_rcsi_gauge(rcsi, f"{col_names[idx]}\nRCSI", dark_mode)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"<div style='text-align: center;'><b>Milestone: {milestone_name}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center;'><b>Level: {classify_rcsi(rcsi)}</b></div>", unsafe_allow_html=True)
            else:
                st.write(f"No simulation data for {col_names[idx]}")
