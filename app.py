"""3PL D2C Analytics Dashboard — main entry point.

Upload the survey dataset (CSV/XLSX) or use the bundled sample. Data is auto-cleaned,
then explored across Descriptive, Diagnostic and Predictive tabs.

Run locally:   streamlit run app.py
"""
import io
import pandas as pd
import streamlit as st

import theme as T
import cleaning as C
import descriptive
import diagnostic
import predictive

st.set_page_config(page_title="3PL D2C Analytics", page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")
T.apply_plotly_theme()
T.inject_css()


@st.cache_data(show_spinner=True)
def load_raw(file_bytes, name):
    if name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet = "Raw_Survey_Data" if "Raw_Survey_Data" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=sheet)
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=True)
def get_clean(file_bytes, name):
    raw = load_raw(file_bytes, name)
    clean, report = C.clean_data(raw)
    return raw, clean, report


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 📦 3PL D2C Analytics")
    st.caption("Upload your survey export — it is cleaned automatically before any analysis.")
    up = st.file_uploader("Dataset (CSV or XLSX)", type=["csv", "xlsx", "xls"])
    if up is not None:
        file_bytes, fname = up.getvalue(), up.name
    else:
        with open("sample_data.csv", "rb") as f:
            file_bytes, fname = f.read(), "sample_data.csv"
        st.info("Using bundled sample dataset (1,178 raw records).")

raw, df_clean, report = get_clean(file_bytes, fname)

with st.sidebar:
    st.markdown("---")
    st.markdown("#### 🔎 Global filters")
    em = st.multiselect("Emirate", sorted(df_clean["Emirate"].unique()),
                        default=sorted(df_clean["Emirate"].unique()))
    ind = st.multiselect("Industry", sorted(df_clean["Industry_Category"].unique()),
                         default=sorted(df_clean["Industry_Category"].unique()))
    size = st.multiselect("Company size", ["Startup", "SME", "Mid-Market", "Enterprise"],
                          default=["Startup", "SME", "Mid-Market", "Enterprise"])
    churn_view = st.radio("Account status", ["All", "Active only", "Churned only"], horizontal=False)

df = df_clean[df_clean["Emirate"].isin(em) & df_clean["Industry_Category"].isin(ind)
              & df_clean["Company_Size"].isin(size)].copy()
if churn_view == "Active only":
    df = df[df["Churned"] != "Yes"]
elif churn_view == "Churned only":
    df = df[df["Churned"] == "Yes"]

with st.sidebar:
    st.markdown("---")
    st.metric("Records in view", f"{len(df):,}", f"{len(df)-len(df_clean):+,} vs full")
    st.caption("Built with Streamlit · scikit-learn · statsmodels · Plotly")

# ---------------- Header + cleaning report ----------------
T.hero("3PL D2C Logistics — Market Intelligence Dashboard",
       "Descriptive · Diagnostic · Predictive analytics for the UAE D2C fulfilment market")

with st.expander("🧼 Auto-cleaning report — what happened to your raw data", expanded=False):
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Raw rows", f"{len(raw):,}")
    cc2.metric("Clean rows", f"{len(df_clean):,}", f"{len(df_clean)-len(raw):+,}")
    cc3.metric("Missing cells removed", f"{int(raw.isna().sum().sum()):,} → 0")
    st.dataframe(pd.DataFrame(report).rename(columns={"step": "Cleaning step", "n": "Records affected"}),
                 use_container_width=True, hide_index=True)

if df.empty:
    st.warning("No records match the current filters. Widen your selection in the sidebar.")
    st.stop()

# ---------------- Tabs ----------------
t1, t2, t3 = st.tabs(["📊  Descriptive", "🔬  Diagnostic", "🔮  Predictive"])
with t1:
    descriptive.render(df)
with t2:
    diagnostic.render(df)
with t3:
    predictive.render(df)
