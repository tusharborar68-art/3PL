# 📦 3PL D2C Logistics — Market Intelligence Dashboard

An interactive **Streamlit** analytics dashboard for a data-driven 3PL startup serving
UAE direct-to-consumer (D2C) brands. Upload a survey export and the app **auto-cleans
the data** before running a full **Descriptive → Diagnostic → Predictive** analytics suite.

---

## ✨ What it does

When you upload a dataset (or use the bundled sample), the app first runs an automatic
cleaning pipeline — trimming whitespace, removing duplicates, standardising category
labels (`DXB`/`dubai` → `Dubai`), normalising mixed booleans, parsing currency strings
(`"AED 250"`) and free text (`"~500"`), fixing mixed date formats, rescaling fractional
percentages, correcting impossible negatives, nulling out-of-range values, and imputing
missing data. A transparent report shows exactly what changed.

### 📊 Descriptive tab
KPI flashcards · time-series & trend charts (area/line/bar) · categorical breakdowns
(bar + donut) · Pareto (80/20) analysis · cohort-retention heatmap · cross-tabulation ·
filterable granular data table.

### 🔬 Diagnostic tab
Deviation/variance delta cards + waterfall · dimensional contribution treemap & stacked
bars · correlation heatmap + multimetric scatter (with trendline) · anomaly/outlier
detection (IQR) + root-cause table · **clustering**: K-Means (centroid), DBSCAN
(density) and Agglomerative (hierarchical) with **Elbow** and **Silhouette** diagnostics,
PCA cluster projection, dendrogram, and a full **RFM** framework with segmentation.

### 🔮 Predictive tab
**Classification** (KNN, Decision Tree, Random Forest, Gradient Boosting, Logistic
Regression) with stratified train/test split, selectable scaling (Standard / MinMax /
Robust), optional hyperparameter tuning, and a results table of Train/Test Accuracy,
Precision, Recall, F1 and AUC-ROC — with the best model highlighted plus ROC curves,
confusion matrix and feature importances. Also: **ARIMA** time-series forecasting,
**propensity** modelling, **LTV** regression, and an **NMF matrix-factorisation**
service recommendation engine.

All controls are interactive; charts and models recompute live.

---

## 🗂 Project structure

```
3pl-d2c-dashboard/
├── app.py              # Entry point: sidebar, upload, cleaning report, tabs
├── cleaning.py         # Automatic data-cleaning pipeline
├── descriptive.py      # Tab 1
├── diagnostic.py       # Tab 2 (clustering, RFM)
├── predictive.py       # Tab 3 (ML, ARIMA, recommender)
├── theme.py            # Palette, CSS, Plotly template, KPI cards
├── sample_data.csv     # Bundled demo dataset (loads by default)
├── requirements.txt
└── .streamlit/config.toml
```

---

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL it prints (usually http://localhost:8501).

---

## ☁️ Deploy on Streamlit Community Cloud (free)

1. Push this folder to a **public GitHub repository** (these files at the repo root).
   ```bash
   git init
   git add .
   git commit -m "3PL D2C analytics dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app**, choose your repo/branch, set **Main file path** to `app.py`,
   and click **Deploy**. Streamlit installs `requirements.txt` automatically.
4. Once live, upload your own dataset via the sidebar, or use the bundled sample.

---

## 📥 Data format

The app expects the survey schema (47 columns). It reads the `Raw_Survey_Data` sheet from
the XLSX deliverable, or any CSV with the same columns. Missing/dirty values are fine —
they are cleaned automatically. Targets `Churned` and `Premium_Tier_Adoption` should be
`Yes`/`No`.

---

## 🔧 Notes & extensions

- The recommender uses `scikit-learn`'s NMF (install-free). To upgrade to implicit-ALS or
  **LightFM** with side-features for cold-start, add the package and swap `_recommend()`.
- Heavy steps are cached with `st.cache_data`, so re-interaction is fast.
