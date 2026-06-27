"""Tab 3 — Predictive analytics: classification, ARIMA, propensity, LTV, recommendations."""
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, r2_score,
                             mean_absolute_error)
from sklearn.decomposition import NMF

import theme as T
import cleaning as C

warnings.filterwarnings("ignore")
SCALERS = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}
NUM_FEATS = C.NUMERIC_COLS + C.FLAG_COLS
CAT_FEATS = C.CAT_COLS


def _prep(df, target, scaler, num_drop=()):
    num = [c for c in NUM_FEATS if c in df and c not in num_drop and c != target]
    cat = [c for c in CAT_FEATS if c in df and c != target]
    pre = ColumnTransformer([
        ("num", SCALERS[scaler](), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return pre, num, cat


@st.cache_data(show_spinner=True)
def compute_classification(df, target, scaler, tune, test_size):
    y = (df[target] == "Yes").astype(int).values
    pre, num, cat = _prep(df, target, scaler)
    Xdf = df[num + cat]
    Xtr, Xte, ytr, yte = train_test_split(Xdf, y, test_size=test_size, stratify=y, random_state=42)
    pre.fit(Xtr)
    Xtr_t, Xte_t = pre.transform(Xtr), pre.transform(Xte)
    if hasattr(Xtr_t, "toarray"):
        Xtr_t, Xte_t = Xtr_t.toarray(), Xte_t.toarray()
    feat_names = list(pre.get_feature_names_out())

    models = {
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 11, 21]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {"max_depth": [4, 6, 10]}),
        "Random Forest": (RandomForestClassifier(random_state=42, n_estimators=200),
                          {"max_depth": [None, 10]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                              {"learning_rate": [0.05, 0.1]}),
        "Logistic Regression": (LogisticRegression(max_iter=2000), {"C": [0.1, 1, 10]}),
    }
    rows, roc_data, importances, fitted = [], {}, None, {}
    for name, (mdl, grid) in models.items():
        if tune:
            gs = GridSearchCV(mdl, grid, cv=3, scoring="f1", n_jobs=-1).fit(Xtr_t, ytr)
            est = gs.best_estimator_
        else:
            est = mdl.fit(Xtr_t, ytr)
        fitted[name] = est
        ptr, pte = est.predict(Xtr_t), est.predict(Xte_t)
        proba = est.predict_proba(Xte_t)[:, 1]
        rows.append({
            "Model": name,
            "Train Acc": round(accuracy_score(ytr, ptr), 3),
            "Test Acc": round(accuracy_score(yte, pte), 3),
            "Precision": round(precision_score(yte, pte, zero_division=0), 3),
            "Recall": round(recall_score(yte, pte, zero_division=0), 3),
            "F1": round(f1_score(yte, pte, zero_division=0), 3),
            "AUC-ROC": round(roc_auc_score(yte, proba), 3),
        })
        fpr, tpr, _ = roc_curve(yte, proba)
        roc_data[name] = (fpr.tolist(), tpr.tolist())

    res = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    best = res.iloc[0]["Model"]
    best_est = fitted[best]
    pte_best = best_est.predict(Xte_t)
    cm = confusion_matrix(yte, pte_best).tolist()
    # importances
    if hasattr(best_est, "feature_importances_"):
        imp = best_est.feature_importances_
    elif hasattr(best_est, "coef_"):
        imp = np.abs(best_est.coef_[0])
    else:
        imp = np.zeros(len(feat_names))
    importances = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)[:15]
    # propensity (logistic) on full data — reuse the train-fitted preprocessor
    Xall = pre.transform(Xdf)
    if hasattr(Xall, "toarray"):
        Xall = Xall.toarray()
    lr = LogisticRegression(max_iter=2000).fit(Xtr_t, ytr)
    prop = lr.predict_proba(Xall)[:, 1]
    return res.to_dict("records"), roc_data, cm, importances, best, prop.tolist()


def render(df: pd.DataFrame):
    # ---- Classification ----
    T.section("Classification", "Predict churn or premium adoption — stratified split, optional tuning &amp; scaling.")
    c1, c2, c3, c4 = st.columns(4)
    target = c1.selectbox("Target", ["Churned", "Premium_Tier_Adoption"], key="clf_target")
    scaler = c2.selectbox("Scaler", list(SCALERS.keys()), key="clf_scaler")
    test_size = c3.slider("Test size", 0.15, 0.4, 0.25, 0.05, key="clf_ts")
    tune = c4.toggle("Hyperparameter tuning", value=False, key="clf_tune")

    res, roc_data, cm, importances, best, prop = compute_classification(
        df, target, scaler, tune, test_size)
    res_df = pd.DataFrame(res)
    st.markdown(f"<span class='badge best'>Best model: {best}</span>"
                f"<span class='badge'>Ranked by AUC-ROC · positive class = '{target}=Yes'</span>",
                unsafe_allow_html=True)
    st.dataframe(res_df.style
                 .background_gradient(cmap="Greens", subset=["Test Acc", "F1", "AUC-ROC"])
                 .apply(lambda r: ["background-color:#e8fff4" if r["Model"] == best else "" for _ in r], axis=1),
                 use_container_width=True)

    g1, g2, g3 = st.columns([2, 1, 2])
    with g1:
        fig = go.Figure()
        for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
            auc = res_df.loc[res_df.Model == name, "AUC-ROC"].values[0]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc})",
                                     line=dict(width=3 if name == best else 1.5,
                                               color=T.SEQ[i % len(T.SEQ)])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                 line=dict(color="#94a3b8", dash="dash")))
        fig.update_layout(height=380, title="ROC curves", xaxis_title="FPR", yaxis_title="TPR",
                          legend=dict(font=dict(size=9), y=.02, x=.4))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        cmf = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                        labels=dict(x="Predicted", y="Actual"), x=["No", "Yes"], y=["No", "Yes"])
        cmf.update_layout(height=380, title=f"Confusion — {best}", coloraxis_showscale=False)
        st.plotly_chart(cmf, use_container_width=True)
    with g3:
        imp_df = pd.DataFrame(importances, columns=["Feature", "Importance"])
        imp_df["Feature"] = imp_df["Feature"].str.replace("num__", "").str.replace("cat__", "")
        fig = px.bar(imp_df.iloc[::-1], x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#cdeee8", T.TEAL, T.NAVY])
        fig.update_layout(height=380, title=f"Top drivers — {best}", coloraxis_showscale=False, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Propensity ----
    T.section("Propensity Modelling", f"Logistic propensity that an account is '{target}=Yes'.")
    pser = pd.Series(prop, index=df.index)
    pp1, pp2 = st.columns([2, 3])
    with pp1:
        h = px.histogram(pser, nbins=30, color_discrete_sequence=[T.INDIGO])
        h.update_layout(height=320, title="Propensity score distribution",
                        xaxis_title="P(Yes)", yaxis_title="Merchants", showlegend=False)
        st.plotly_chart(h, use_container_width=True)
    with pp2:
        topn = st.slider("Show top-N highest-propensity accounts", 5, 40, 15, key="prop_n")
        tp = df.assign(Propensity=pser.round(3)).nlargest(topn, "Propensity")
        st.dataframe(tp[["Customer_ID", "Company_Name", "Company_Size", "Current_Provider",
                         "Satisfaction_Score", "Propensity"]], use_container_width=True, height=300)

    # ---- ARIMA forecasting ----
    T.section("Time-Series Forecasting (ARIMA)")
    f1, f2 = st.columns([1, 3])
    with f1:
        series_kind = st.selectbox("Series", ["New merchants / month", "Onboarded monthly spend"], key="arima_series")
        horizon = st.slider("Forecast horizon (months)", 3, 12, 6, key="arima_h")
        p = st.slider("p", 0, 3, 1, key="ar_p"); d = st.slider("d", 0, 2, 1, key="ar_d"); q = st.slider("q", 0, 3, 1, key="ar_q")
    g = df.dropna(subset=["Signup_Month"]).groupby("Signup_Month")
    s = g.size() if series_kind.startswith("New") else g["Monthly_Logistics_Spend_AED"].sum()
    s = s.asfreq("MS").fillna(0)
    with f2:
        _arima_plot(s, (p, d, q), horizon)

    # ---- LTV regression ----
    T.section("Lifetime-Value Prediction (Regression)")
    l1, l2 = st.columns([2, 3])
    ltv = _ltv_model(df)
    with l1:
        T.kpi_cards([
            {"label": "R² (test)", "value": f"{ltv['r2']:.3f}", "tone": "teal"},
            {"label": "MAE (AED)", "value": f"{ltv['mae']:,.0f}", "tone": "amber"},
        ])
        st.markdown("<div class='hint'>Gradient-boosted regressor predicting Customer_Lifetime_Value_AED.</div>",
                    unsafe_allow_html=True)
    with l2:
        sc = px.scatter(x=ltv["y_test"], y=ltv["y_pred"], opacity=.6,
                        labels={"x": "Actual CLV", "y": "Predicted CLV"}, color_discrete_sequence=[T.TEAL])
        lim = max(max(ltv["y_test"]), max(ltv["y_pred"]))
        sc.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", showlegend=False,
                                line=dict(color="#94a3b8", dash="dash")))
        sc.update_layout(height=340, title="Predicted vs actual CLV")
        st.plotly_chart(sc, use_container_width=True)

    # ---- Recommendation engine ----
    T.section("Service Recommendation Engine", "Matrix factorisation (NMF) on the merchant × service usage matrix.")
    rec = _recommend(df)
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        cid = st.selectbox("Select a merchant", df["Customer_ID"].tolist(), key="rec_cid")
    cur, recs = rec(cid)
    with rc2:
        st.markdown(f"**Currently uses:** {', '.join(cur) if cur else 'No add-on services'}")
        if recs:
            rdf = pd.DataFrame(recs, columns=["Recommended service", "Affinity score"])
            fig = px.bar(rdf, x="Affinity score", y="Recommended service", orientation="h",
                         color="Affinity score", color_continuous_scale=["#cdeee8", T.TEAL, T.NAVY])
            fig.update_layout(height=240, coloraxis_showscale=False, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This merchant already uses all available services.")
    st.caption("Collaborative-filtering upgrade path: swap NMF for implicit-ALS or LightFM with merchant & service "
               "side-features for cold-start handling.")


def _arima_plot(s, order, horizon):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        if len(s) < 8:
            st.info("Not enough monthly history for ARIMA.")
            return
        model = ARIMA(s, order=order).fit()
        fc = model.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name="History",
                                 line=dict(color=T.NAVY)))
        fig.add_trace(go.Scatter(x=mean.index, y=mean.values, mode="lines+markers", name="Forecast",
                                 line=dict(color=T.TEAL, width=3)))
        fig.add_trace(go.Scatter(x=list(ci.index) + list(ci.index[::-1]),
                                 y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                                 fill="toself", fillcolor="rgba(20,184,166,.15)",
                                 line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
        fig.update_layout(height=360, title=f"ARIMA{order} forecast — next {horizon} months")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"ARIMA could not fit with these parameters ({e}). Try different p,d,q.")


@st.cache_data(show_spinner=False)
def _ltv_model(df):
    from sklearn.model_selection import train_test_split as tts
    pre, num, cat = _prep(df, "x", "Standard", num_drop=["Customer_Lifetime_Value_AED"])
    y = df["Customer_Lifetime_Value_AED"].values
    Xdf = df[num + cat]
    Xtr, Xte, ytr, yte = tts(Xdf, y, test_size=0.25, random_state=42)
    pre.fit(Xtr)
    Xtr_t = pre.transform(Xtr); Xte_t = pre.transform(Xte)
    if hasattr(Xtr_t, "toarray"):
        Xtr_t, Xte_t = Xtr_t.toarray(), Xte_t.toarray()
    m = GradientBoostingRegressor(random_state=42).fit(Xtr_t, ytr)
    pred = m.predict(Xte_t)
    return {"r2": r2_score(yte, pred), "mae": mean_absolute_error(yte, pred),
            "y_test": yte.tolist(), "y_pred": pred.tolist()}


def _recommend(df):
    services = C.FLAG_COLS
    M = df[services].astype(float).values
    nmf = NMF(n_components=4, init="nndsvda", random_state=42, max_iter=400)
    W = nmf.fit_transform(M); H = nmf.components_
    R = W @ H
    label = {s: s.replace("Uses_", "").replace("_", " ") for s in services}
    idx = {c: i for i, c in enumerate(df["Customer_ID"].tolist())}

    def rec(cid):
        i = idx[cid]
        cur = [label[s] for s, v in zip(services, M[i]) if v == 1]
        scores = [(label[s], round(float(R[i, j]), 3)) for j, s in enumerate(services) if M[i, j] == 0]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:4]
        return cur, scores
    return rec
