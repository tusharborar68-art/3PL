"""Tab 1 — Descriptive analytics."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import theme as T


def _aed(x):
    if x >= 1e9: return f"AED {x/1e9:.2f}B"
    if x >= 1e6: return f"AED {x/1e6:.2f}M"
    if x >= 1e3: return f"AED {x/1e3:.1f}K"
    return f"AED {x:,.0f}"


def render(df: pd.DataFrame):
    T.section("Highest-level KPIs", "Portfolio snapshot after auto-cleaning.")
    churn = (df["Churned"] == "Yes").mean() * 100
    prem = (df["Premium_Tier_Adoption"] == "Yes").mean() * 100
    T.kpi_cards([
        {"label": "Active Merchants", "value": f"{len(df):,}", "tone": "teal"},
        {"label": "Monthly Orders", "value": f"{df['Monthly_Order_Volume'].sum()/1e6:.2f}M", "tone": "indigo"},
        {"label": "Avg Order Value", "value": _aed(df['Avg_Order_Value_AED'].mean()), "tone": "sky"},
        {"label": "Monthly Logistics Spend", "value": _aed(df['Monthly_Logistics_Spend_AED'].sum()), "tone": "amber"},
        {"label": "Avg Satisfaction", "value": f"{df['Satisfaction_Score'].mean():.1f}/10", "tone": "violet"},
        {"label": "Avg NPS", "value": f"{df['NPS_Rating'].mean():.1f}", "tone": "green"},
        {"label": "Churn Rate", "value": f"{churn:.1f}%", "tone": "red", "sub": "of accounts", "up": False},
        {"label": "Premium Adoption", "value": f"{prem:.1f}%", "tone": "teal"},
    ])

    # ---- Time series ----
    T.section("Time Series &amp; Trends")
    c1, c2 = st.columns([1, 3])
    with c1:
        metric = st.selectbox("Trend metric", [
            "New merchants / month", "Cumulative merchants",
            "Onboarded monthly spend (AED)", "Avg satisfaction by cohort"], key="ts_metric")
        chart = st.radio("Style", ["Area", "Line", "Bars"], horizontal=True, key="ts_style")
    g = df.dropna(subset=["Signup_Month"]).groupby("Signup_Month")
    if metric == "New merchants / month":
        s = g.size(); ylab = "New merchants"
    elif metric == "Cumulative merchants":
        s = g.size().cumsum(); ylab = "Cumulative merchants"
    elif metric == "Onboarded monthly spend (AED)":
        s = g["Monthly_Logistics_Spend_AED"].sum(); ylab = "Spend (AED)"
    else:
        s = g["Satisfaction_Score"].mean(); ylab = "Avg satisfaction"
    ts = s.reset_index(); ts.columns = ["Month", "Value"]
    if chart == "Area":
        fig = px.area(ts, x="Month", y="Value", markers=True)
        fig.update_traces(line_color=T.TEAL, fillcolor="rgba(20,184,166,.18)")
    elif chart == "Line":
        fig = px.line(ts, x="Month", y="Value", markers=True); fig.update_traces(line_color=T.INDIGO)
    else:
        fig = px.bar(ts, x="Month", y="Value"); fig.update_traces(marker_color=T.TEAL)
    fig.update_layout(height=330, yaxis_title=ylab, xaxis_title=None)
    with c2:
        st.plotly_chart(fig, use_container_width=True)

    # ---- Categorical breakdown ----
    T.section("Categorical &amp; Distribution Breakdown")
    cc1, cc2, cc3 = st.columns([1, 2, 2])
    with cc1:
        dim = st.selectbox("Dimension", ["Industry_Category", "Emirate", "Company_Size",
                                          "Current_Provider", "Contract_Type", "Primary_Sales_Channel",
                                          "Price_Sensitivity"], key="cat_dim")
        meas = st.selectbox("Measure", ["Merchant count", "Monthly spend (AED)", "Monthly orders"], key="cat_meas")
    if meas == "Merchant count":
        agg = df[dim].value_counts().reset_index(); agg.columns = [dim, "Value"]
    elif meas == "Monthly spend (AED)":
        agg = df.groupby(dim)["Monthly_Logistics_Spend_AED"].sum().sort_values(ascending=False).reset_index()
        agg.columns = [dim, "Value"]
    else:
        agg = df.groupby(dim)["Monthly_Order_Volume"].sum().sort_values(ascending=False).reset_index()
        agg.columns = [dim, "Value"]
    bar = px.bar(agg, x="Value", y=dim, orientation="h", color="Value",
                 color_continuous_scale=["#9ee7df", T.TEAL, T.NAVY])
    bar.update_layout(height=360, yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, xaxis_title=meas, yaxis_title=None)
    with cc2:
        st.plotly_chart(bar, use_container_width=True)
    donut = px.pie(agg, names=dim, values="Value", hole=.58, color_discrete_sequence=T.SEQ)
    donut.update_traces(textposition="inside", textinfo="percent")
    donut.update_layout(height=360, legend=dict(orientation="v", font=dict(size=10)))
    with cc3:
        st.plotly_chart(donut, use_container_width=True)

    # ---- Pareto ----
    T.section("Pareto Analysis", "The vital few that drive the most volume — 80/20 lens.")
    p1, p2 = st.columns([1, 3])
    with p1:
        pdim = st.selectbox("Group by", ["Current_Provider", "Industry_Category", "Primary_Pain_Point",
                                         "Emirate", "Contract_Type"], key="pareto_dim")
        pmeas = st.selectbox("Measure", ["Monthly_Logistics_Spend_AED", "Monthly_Order_Volume"], key="pareto_meas")
    pg = df.groupby(pdim)[pmeas].sum().sort_values(ascending=False).reset_index()
    pg["cum_pct"] = pg[pmeas].cumsum() / pg[pmeas].sum() * 100
    fig = go.Figure()
    fig.add_bar(x=pg[pdim], y=pg[pmeas], name=pmeas, marker_color=T.INDIGO)
    fig.add_trace(go.Scatter(x=pg[pdim], y=pg["cum_pct"], name="Cumulative %", yaxis="y2",
                             mode="lines+markers", line=dict(color=T.AMBER, width=3)))
    fig.add_hline(y=80, yref="y2", line=dict(color=T.RED, dash="dash"),
                  annotation_text="80%", annotation_position="top left")
    fig.update_layout(height=360, yaxis=dict(title=pmeas),
                      yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                      legend=dict(orientation="h", y=1.12))
    with p2:
        st.plotly_chart(fig, use_container_width=True)

    # ---- Cohort retention ----
    T.section("Cohort Retention", "Signup-month cohorts vs. months since signup, from first to last active order.")
    coh = _cohort_matrix(df)
    if coh is not None and not coh.empty:
        fig = px.imshow(coh, color_continuous_scale=["#fee2e2", "#fde68a", T.TEAL, T.NAVY],
                        aspect="auto", labels=dict(x="Months since signup", y="Cohort", color="Retention %"))
        fig.update_layout(height=420)
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough dated history to build a cohort matrix.")

    # ---- Cross tabulation ----
    T.section("Cross Tabulation")
    x1, x2, x3 = st.columns(3)
    dims = ["Company_Size", "Industry_Category", "Emirate", "Contract_Type",
            "Tech_Integration_Level", "Price_Sensitivity", "Current_Provider"]
    with x1:
        rdim = st.selectbox("Rows", dims, index=0, key="ct_r")
    with x2:
        cdim = st.selectbox("Columns", dims, index=3, key="ct_c")
    with x3:
        val = st.selectbox("Value", ["Count", "Churn rate %", "Avg monthly spend", "Avg satisfaction"], key="ct_v")
    if rdim == cdim:
        st.warning("Pick two different dimensions.")
    else:
        if val == "Count":
            ct = pd.crosstab(df[rdim], df[cdim])
        elif val == "Churn rate %":
            ct = pd.crosstab(df[rdim], df[cdim], values=(df["Churned"] == "Yes"), aggfunc="mean").round(3) * 100
        elif val == "Avg monthly spend":
            ct = pd.crosstab(df[rdim], df[cdim], values=df["Monthly_Logistics_Spend_AED"], aggfunc="mean").round(0)
        else:
            ct = pd.crosstab(df[rdim], df[cdim], values=df["Satisfaction_Score"], aggfunc="mean").round(2)
        st.dataframe(ct.style.background_gradient(cmap="BuGn", axis=None).format(precision=1),
                     use_container_width=True)

    # ---- Granular table ----
    T.section("Granular Data Table", "Filter and inspect cleaned account-level records.")
    show_cols = st.multiselect("Columns", list(df.columns),
                               default=["Customer_ID", "Company_Name", "Industry_Category", "Emirate",
                                        "Company_Size", "Monthly_Order_Volume", "Monthly_Logistics_Spend_AED",
                                        "Satisfaction_Score", "NPS_Rating", "Churned"], key="grid_cols")
    st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=360)


def _cohort_matrix(df, max_cohorts=18, max_months=12):
    d = df.dropna(subset=["Signup_Month", "Tenure_Months", "Active_Until_Months"]).copy()
    if d.empty:
        return None
    cohorts = sorted(d["Signup_Month"].unique())[-max_cohorts:]
    rows = {}
    for c in cohorts:
        sub = d[d["Signup_Month"] == c]
        vals = []
        for m in range(0, max_months + 1):
            denom = (sub["Tenure_Months"] >= m).sum()
            if denom == 0:
                vals.append(np.nan)
            else:
                numer = ((sub["Tenure_Months"] >= m) & (sub["Active_Until_Months"] >= m)).sum()
                vals.append(round(numer / denom * 100, 0))
        rows[pd.Timestamp(c).strftime("%Y-%m")] = vals
    mat = pd.DataFrame(rows, index=[f"M{m}" for m in range(0, max_months + 1)]).T
    return mat
