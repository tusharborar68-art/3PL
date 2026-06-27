"""Tab 2 — Diagnostic analytics: variance, drills, correlation, anomalies, clustering, RFM."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import theme as T

CLUSTER_FEATURES = [
    "Monthly_Order_Volume", "Avg_Order_Value_AED", "Number_of_SKUs",
    "Monthly_Logistics_Spend_AED", "Number_of_Sales_Channels", "Recency_Days",
    "Order_Frequency_Per_Month", "Return_Rate_Pct", "Satisfaction_Score",
]
SCALERS = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}


def _scale(df, feats, scaler):
    X = df[list(feats)].astype(float).values
    return SCALERS[scaler]().fit_transform(X)


@st.cache_data(show_spinner=False)
def _kmeans_sweep(df, feats, scaler):
    X = _scale(df, feats, scaler)
    ks = list(range(2, 9))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    return ks, inertia, sil


def render(df: pd.DataFrame):
    # ---- Deviation & variance ----
    T.section("Deviation &amp; Variance", "Service KPIs vs. operational targets.")
    targets = {"On_Time_Delivery_Rate_Pct": ("On-time delivery", 95, "%", True),
               "Order_Accuracy_Pct": ("Order accuracy", 98, "%", True),
               "Avg_Delivery_Time_Days": ("Delivery time", 2.0, " days", False),
               "Satisfaction_Score": ("Satisfaction", 8.0, "/10", True)}
    cols = st.columns(len(targets))
    for (col, (lbl, tgt, unit, higher)), cc in zip(targets.items(), cols):
        actual = df[col].mean()
        delta = actual - tgt
        good = delta >= 0 if higher else delta <= 0
        cc.metric(lbl, f"{actual:.1f}{unit}", f"{delta:+.1f} vs target {tgt}{unit}",
                  delta_color="normal" if good else "inverse")

    cw1, cw2 = st.columns(2)
    with cw1:
        st.markdown("<div class='hint'>Monthly spend at risk — contribution of churned accounts by segment</div>",
                    unsafe_allow_html=True)
        risk = (df[df["Churned"] == "Yes"].groupby("Company_Size")["Monthly_Logistics_Spend_AED"]
                .sum().reindex(["Startup", "SME", "Mid-Market", "Enterprise"]).fillna(0))
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["relative"] * len(risk) + ["total"],
            x=list(risk.index) + ["Total at risk"], y=list(risk.values) + [0],
            connector=dict(line=dict(color="#cbd5e1")),
            increasing=dict(marker=dict(color=T.RED)), totals=dict(marker=dict(color=T.NAVY))))
        wf.update_layout(height=330, yaxis_title="AED / month")
        st.plotly_chart(wf, use_container_width=True)
    with cw2:
        st.markdown("<div class='hint'>Avg satisfaction deviation from portfolio mean, by provider</div>",
                    unsafe_allow_html=True)
        mean_sat = df["Satisfaction_Score"].mean()
        dev = (df.groupby("Current_Provider")["Satisfaction_Score"].mean() - mean_sat).sort_values()
        fig = go.Figure(go.Bar(x=dev.values, y=dev.index, orientation="h",
                               marker_color=[T.RED if v < 0 else T.GREEN for v in dev.values]))
        fig.add_vline(x=0, line=dict(color="#94a3b8"))
        fig.update_layout(height=330, xaxis_title="Δ vs mean satisfaction")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Dimensional contribution / drills ----
    T.section("Dimensional Contribution &amp; Drills")
    d1, d2 = st.columns(2)
    with d1:
        tm = px.treemap(df, path=[px.Constant("All"), "Emirate", "Industry_Category", "Company_Size"],
                        values="Monthly_Logistics_Spend_AED", color="Monthly_Logistics_Spend_AED",
                        color_continuous_scale=["#d1faf3", T.TEAL, T.NAVY])
        tm.update_layout(height=420, margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(tm, use_container_width=True)
    with d2:
        sb = (df.groupby(["Company_Size", "Contract_Type"]).size().reset_index(name="n"))
        order = ["Startup", "SME", "Mid-Market", "Enterprise"]
        fig = px.bar(sb, x="Company_Size", y="n", color="Contract_Type", barmode="stack",
                     category_orders={"Company_Size": order}, color_discrete_sequence=T.SEQ)
        fig.update_layout(height=420, yaxis_title="Merchants", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Correlation & multimetric mapping ----
    T.section("Correlation &amp; Multimetric Mapping")
    num = ["Monthly_Order_Volume", "Avg_Order_Value_AED", "Number_of_SKUs",
           "Monthly_Logistics_Spend_AED", "Return_Rate_Pct", "On_Time_Delivery_Rate_Pct",
           "Order_Accuracy_Pct", "Avg_Support_Response_Hours", "Complaints_Last_Quarter",
           "Damaged_Shipment_Pct", "Satisfaction_Score", "NPS_Rating", "Recency_Days"]
    e1, e2 = st.columns([3, 2])
    with e1:
        corr = df[num].corr()
        hm = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                       labels=dict(color="r"))
        hm.update_layout(height=440, margin=dict(l=10, t=10))
        st.plotly_chart(hm, use_container_width=True)
    with e2:
        xv = st.selectbox("X", num, index=0, key="sc_x")
        yv = st.selectbox("Y", num, index=3, key="sc_y")
        cv = st.selectbox("Colour", ["Company_Size", "Churned", "Price_Sensitivity", "Emirate"], key="sc_c")
        sc = px.scatter(df, x=xv, y=yv, color=cv, opacity=.65, trendline="ols",
                        trendline_scope="overall", color_discrete_sequence=T.SEQ)
        sc.update_layout(height=380, legend=dict(font=dict(size=9)))
        st.plotly_chart(sc, use_container_width=True)

    # ---- Anomaly / exception / root cause ----
    T.section("Anomaly, Exception &amp; Root Cause")
    a1, a2 = st.columns([1, 2])
    with a1:
        ametric = st.selectbox("Metric for outlier scan (IQR)", [
            "Monthly_Logistics_Spend_AED", "Avg_Order_Value_AED", "Monthly_Order_Volume",
            "Complaints_Last_Quarter", "Return_Rate_Pct"], key="anom_metric")
    q1, q3 = df[ametric].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = df[(df[ametric] < lo) | (df[ametric] > hi)]
    with a2:
        st.markdown(f"<div class='hint'>{len(out)} outliers outside [{lo:,.0f}, {hi:,.0f}] on "
                    f"<b>{ametric}</b></div>", unsafe_allow_html=True)
    box = px.box(df, y=ametric, points="outliers", color_discrete_sequence=[T.INDIGO])
    box.update_layout(height=300)
    ra1, ra2 = st.columns([2, 3])
    with ra1:
        st.plotly_chart(box, use_container_width=True)
    with ra2:
        st.markdown("<div class='hint'>Root-cause: complaints &amp; churn by primary pain point</div>",
                    unsafe_allow_html=True)
        rc = (df.groupby("Primary_Pain_Point")
              .agg(Merchants=("Customer_ID", "count"),
                   Avg_Complaints=("Complaints_Last_Quarter", "mean"),
                   Churn_Rate=("Churned", lambda s: (s == "Yes").mean() * 100))
              .round(2).sort_values("Churn_Rate", ascending=False))
        st.dataframe(rc.style.background_gradient(cmap="OrRd", subset=["Churn_Rate"]), use_container_width=True)
    with st.expander(f"View {len(out)} flagged outlier accounts"):
        st.dataframe(out[["Customer_ID", "Company_Name", "Company_Size", ametric,
                          "Churned"]].sort_values(ametric, ascending=False), use_container_width=True)

    # ---- Clustering ----
    T.section("Customer Segmentation — Clustering", "Scale features, then compare centroid, density &amp; hierarchical methods.")
    s1, s2 = st.columns([1, 3])
    with s1:
        scaler = st.selectbox("Scaler", list(SCALERS.keys()), key="clu_scaler")
        feats = st.multiselect("Features", CLUSTER_FEATURES,
                               default=["Monthly_Order_Volume", "Avg_Order_Value_AED",
                                        "Number_of_SKUs", "Monthly_Logistics_Spend_AED",
                                        "Number_of_Sales_Channels"], key="clu_feats")
    if len(feats) < 2:
        st.warning("Select at least two features.")
        return
    X = _scale(df, feats, scaler)

    ks, inertia, sil = _kmeans_sweep(df, tuple(feats), scaler)
    cc1, cc2 = st.columns(2)
    with cc1:
        el = go.Figure(go.Scatter(x=ks, y=inertia, mode="lines+markers", line=dict(color=T.INDIGO, width=3)))
        el.update_layout(height=300, title="Elbow method (inertia)", xaxis_title="k", yaxis_title="Inertia")
        st.plotly_chart(el, use_container_width=True)
    with cc2:
        best_k = ks[int(np.argmax(sil))]
        si = go.Figure(go.Scatter(x=ks, y=sil, mode="lines+markers", line=dict(color=T.TEAL, width=3)))
        si.add_vline(x=best_k, line=dict(color=T.AMBER, dash="dash"),
                     annotation_text=f"best k={best_k}")
        si.update_layout(height=300, title="Silhouette score by k", xaxis_title="k", yaxis_title="Silhouette")
        st.plotly_chart(si, use_container_width=True)

    method = st.radio("Algorithm", ["K-Means (centroid)", "DBSCAN (density)", "Agglomerative (hierarchical)"],
                      horizontal=True, key="clu_method")
    pca = PCA(n_components=2, random_state=42).fit_transform(X)
    proj = pd.DataFrame({"PC1": pca[:, 0], "PC2": pca[:, 1]})

    if method == "K-Means (centroid)":
        k = st.slider("Clusters (k)", 2, 8, int(best_k), key="km_k")
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    elif method == "DBSCAN (density)":
        m1, m2 = st.columns(2)
        eps = m1.slider("eps", 0.2, 3.0, 1.2, 0.1, key="db_eps")
        ms = m2.slider("min_samples", 3, 30, 8, key="db_ms")
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
    else:
        k = st.slider("Clusters (k)", 2, 8, int(best_k), key="ag_k")
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)

    proj["Cluster"] = [f"C{l}" if l >= 0 else "Noise" for l in labels]
    n_clusters = len({l for l in labels if l >= 0})
    sil_txt = "—"
    if n_clusters > 1:
        mask = labels >= 0
        if mask.sum() > n_clusters:
            sil_txt = f"{silhouette_score(X[mask], labels[mask]):.3f}"
    st.markdown(f"<span class='badge'>Clusters: {n_clusters}</span>"
                f"<span class='badge'>Noise: {(labels < 0).sum()}</span>"
                f"<span class='badge best'>Silhouette: {sil_txt}</span>", unsafe_allow_html=True)

    g1, g2 = st.columns([3, 2])
    with g1:
        sc = px.scatter(proj, x="PC1", y="PC2", color="Cluster", opacity=.7,
                        color_discrete_sequence=T.SEQ)
        sc.update_layout(height=420, title="Clusters projected on 2 principal components")
        st.plotly_chart(sc, use_container_width=True)
    with g2:
        prof = df.copy(); prof["Cluster"] = proj["Cluster"].values
        profile = (prof[prof["Cluster"] != "Noise"].groupby("Cluster")[feats].mean().round(1))
        profile["Merchants"] = prof[prof["Cluster"] != "Noise"].groupby("Cluster").size()
        st.markdown("<div class='hint'>Cluster profiles (feature means)</div>", unsafe_allow_html=True)
        st.dataframe(profile.style.background_gradient(cmap="BuGn", axis=0), use_container_width=True)

    with st.expander("Hierarchical dendrogram (60-account sample)"):
        samp = df[feats].sample(min(60, len(df)), random_state=1)
        Xs = SCALERS[scaler]().fit_transform(samp.values)
        dfig = ff.create_dendrogram(Xs, color_threshold=None,
                                    linkagefun=lambda x: linkage(x, "ward"))
        dfig.update_layout(height=380, title="Ward linkage", xaxis_title="accounts")
        st.plotly_chart(dfig, use_container_width=True)

    # ---- RFM ----
    T.section("RFM Framework", "Recency · Frequency · Monetary scoring and segmentation.")
    rfm = _rfm(df)
    r1, r2 = st.columns([2, 3])
    with r1:
        seg = rfm["RFM_Segment"].value_counts().reset_index()
        seg.columns = ["Segment", "Merchants"]
        fig = px.bar(seg, x="Merchants", y="Segment", orientation="h", color="Merchants",
                     color_continuous_scale=["#cdeee8", T.TEAL, T.NAVY])
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        piv = rfm.pivot_table(index="R", columns="F", values="M", aggfunc="mean")
        hm = px.imshow(piv, color_continuous_scale=["#fee2e2", "#fde68a", T.TEAL],
                       labels=dict(x="Frequency score", y="Recency score", color="Avg M"), aspect="auto")
        hm.update_layout(height=400, title="Avg Monetary score across R × F")
        st.plotly_chart(hm, use_container_width=True)
    with st.expander("RFM scored table"):
        st.dataframe(rfm[["Customer_ID", "Company_Name", "Recency_Days", "Order_Frequency_Per_Month",
                          "Monthly_Logistics_Spend_AED", "R", "F", "M", "RFM_Score", "RFM_Segment"]]
                     .sort_values("RFM_Score", ascending=False), use_container_width=True, height=320)


def _rfm(df):
    r = df.copy()
    r["R"] = pd.qcut(r["Recency_Days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    r["F"] = pd.qcut(r["Order_Frequency_Per_Month"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["M"] = pd.qcut(r["Monthly_Logistics_Spend_AED"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["RFM_Score"] = r["R"] + r["F"] + r["M"]

    def seg(row):
        s = row["RFM_Score"]
        if s >= 13: return "Champions"
        if s >= 11: return "Loyal"
        if s >= 9: return "Potential Loyalist"
        if s >= 7: return "Needs Attention"
        if s >= 5: return "At Risk"
        return "Hibernating"
    r["RFM_Segment"] = r.apply(seg, axis=1)
    return r
