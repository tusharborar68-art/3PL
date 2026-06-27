"""Automatic data-cleaning pipeline for the 3PL D2C survey dataset.

Handles every error class injected into the raw file: whitespace, duplicate rows,
inconsistent category labels, mixed booleans, currency strings, free text in numeric
fields, mixed date formats, out-of-range values, fraction/percent scale mix-ups,
impossible negatives, and missing values. Returns the clean frame plus a report.
"""
import re
import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-06-27")

FLAG_COLS = [
    "Uses_Warehousing", "Uses_Same_Day_Delivery", "Uses_Next_Day_Delivery",
    "Uses_Reverse_Logistics", "Uses_COD_Collection", "Uses_International_Shipping",
    "Uses_Qcommerce_Fulfillment", "Uses_Inventory_Management", "Uses_Custom_Packaging",
]

NUMERIC_COLS = [
    "Company_Age_Years", "Number_of_Sales_Channels", "Monthly_Order_Volume",
    "Avg_Order_Value_AED", "Number_of_SKUs", "Total_Lifetime_Orders", "Return_Rate_Pct",
    "COD_Order_Pct", "Avg_Delivery_Time_Days", "Peak_Season_Multiplier",
    "Monthly_Logistics_Spend_AED", "Recency_Days", "Order_Frequency_Per_Month",
    "Customer_Lifetime_Value_AED", "On_Time_Delivery_Rate_Pct", "Order_Accuracy_Pct",
    "Avg_Support_Response_Hours", "Complaints_Last_Quarter", "Damaged_Shipment_Pct",
    "Satisfaction_Score", "NPS_Rating", "Willingness_To_Switch",
]

CAT_COLS = [
    "Industry_Category", "Company_Size", "Emirate", "Primary_Sales_Channel",
    "Current_Provider", "Contract_Type", "Preferred_Billing_Cycle",
    "Tech_Integration_Level", "Price_Sensitivity", "Primary_Pain_Point",
]

# bounds: (min, max) inclusive — values outside become NaN before imputation
BOUNDS = {
    "Satisfaction_Score": (1, 10), "NPS_Rating": (0, 10), "Willingness_To_Switch": (1, 5),
    "On_Time_Delivery_Rate_Pct": (0, 100), "Order_Accuracy_Pct": (0, 100),
    "Return_Rate_Pct": (0, 100), "COD_Order_Pct": (0, 100), "Damaged_Shipment_Pct": (0, 100),
}
NON_NEGATIVE = ["Monthly_Order_Volume", "Company_Age_Years", "Number_of_SKUs",
                "Avg_Order_Value_AED", "Monthly_Logistics_Spend_AED", "Total_Lifetime_Orders"]

EMIRATE_MAP = {
    "dubai": "Dubai", "dxb": "Dubai", "abu dhabi": "Abu Dhabi", "auh": "Abu Dhabi",
    "sharjah": "Sharjah", "shj": "Sharjah", "ras al khaimah": "Ras Al Khaimah",
    "rak": "Ras Al Khaimah", "ajman": "Ajman", "fujairah": "Fujairah",
    "umm al quwain": "Umm Al Quwain",
}
PROVIDER_MAP = {
    "aramex": "Aramex", "imile": "iMile", "i-mile": "iMile", "in house": "In-house",
    "inhouse": "In-house", "in-house": "In-house", "fetchr": "Fetchr", "quiqup": "Quiqup",
    "smsa express": "SMSA Express", "shipa": "Shipa", "none": "None",
}
BOOL_TRUE = {"1", "1.0", "yes", "y", "true", "t"}
BOOL_FALSE = {"0", "0.0", "no", "n", "false", "f"}


def _to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).lower().replace("aed", "").replace(",", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else np.nan


def _to_bool(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in BOOL_TRUE:
        return 1
    if s in BOOL_FALSE:
        return 0
    return np.nan


def _norm_cat(x, mapping=None):
    if pd.isna(x):
        return np.nan
    s = re.sub(r"\s+", " ", str(x)).strip()
    if mapping is not None:
        return mapping.get(s.lower(), s.title() if s else np.nan)
    return s


def _parse_dates(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")


def clean_data(raw: pd.DataFrame):
    """Return (clean_df, report list[dict])."""
    df = raw.copy()
    report = []

    # 1) IDs: strip whitespace
    if "Customer_ID" in df:
        df["Customer_ID"] = df["Customer_ID"].astype(str).str.strip()
        report.append({"step": "Trim whitespace in Customer_ID", "n": "all rows"})

    # 2) Duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if "Customer_ID" in df:
        df = df.drop_duplicates(subset="Customer_ID", keep="first")
    report.append({"step": "Removed duplicate rows / IDs", "n": before - len(df)})

    # 3) Categoricals — standardise labels
    if "Emirate" in df:
        n = df["Emirate"].apply(lambda v: 0 if pd.isna(v) else 1).sum()
        df["Emirate"] = df["Emirate"].apply(lambda v: _norm_cat(v, EMIRATE_MAP))
        report.append({"step": "Standardised Emirate labels (DXB→Dubai, etc.)", "n": int(n)})
    if "Current_Provider" in df:
        df["Current_Provider"] = df["Current_Provider"].apply(lambda v: _norm_cat(v, PROVIDER_MAP))
        report.append({"step": "Standardised provider names", "n": "—"})
    for c in ["Industry_Category", "Company_Size", "Primary_Sales_Channel", "Contract_Type",
              "Preferred_Billing_Cycle", "Tech_Integration_Level", "Price_Sensitivity",
              "Primary_Pain_Point"]:
        if c in df:
            df[c] = df[c].apply(_norm_cat)

    # 4) Boolean flags
    fixed = 0
    for c in FLAG_COLS:
        if c in df:
            orig = df[c].copy()
            df[c] = df[c].apply(_to_bool)
            fixed += (orig.astype(str) != df[c].astype(str)).sum()
    report.append({"step": "Normalised service flags to 0/1 (Yes/No/TRUE/Y → 1/0)", "n": int(fixed)})

    # 5) Numeric coercion (currency strings, free text, ~approx)
    coerced = 0
    for c in NUMERIC_COLS:
        if c in df:
            orig_na = df[c].isna().sum()
            df[c] = df[c].apply(_to_num)
            coerced += int(df[c].isna().sum() - orig_na)
    report.append({"step": "Parsed numbers from text/currency ('AED 250', '~500')", "n": "applied"})

    # 6) Return rate fraction fix (0.14 → 14)
    if "Return_Rate_Pct" in df:
        frac = df["Return_Rate_Pct"].between(0, 1, inclusive="right") & (df["Return_Rate_Pct"] > 0)
        df.loc[frac, "Return_Rate_Pct"] = df.loc[frac, "Return_Rate_Pct"] * 100
        report.append({"step": "Rescaled fractional Return_Rate (0.xx → %)", "n": int(frac.sum())})

    # 7) Impossible negatives → absolute
    negfix = 0
    for c in NON_NEGATIVE:
        if c in df:
            neg = df[c] < 0
            negfix += int(neg.sum())
            df.loc[neg, c] = df.loc[neg, c].abs()
    report.append({"step": "Fixed impossible negatives (|value|)", "n": negfix})

    # 8) Out-of-range → NaN (then imputed)
    oor = 0
    for c, (lo, hi) in BOUNDS.items():
        if c in df:
            bad = ~df[c].between(lo, hi) & df[c].notna()
            oor += int(bad.sum())
            df.loc[bad, c] = np.nan
    report.append({"step": "Out-of-range values set to NaN (sat>10, on-time>100…)", "n": oor})

    # 9) Dates
    for c in ["Signup_Date", "Last_Order_Date"]:
        if c in df:
            df[c] = _parse_dates(df[c])
    report.append({"step": "Parsed mixed date formats to datetime", "n": "—"})

    # 10) Imputation
    num_imp = 0
    for c in NUMERIC_COLS:
        if c in df and df[c].isna().any():
            num_imp += int(df[c].isna().sum())
            df[c] = df[c].fillna(df[c].median())
    for c in FLAG_COLS:
        if c in df and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True).iloc[0] if df[c].notna().any() else 0).astype(int)
    cat_imp = 0
    for c in CAT_COLS:
        if c in df and df[c].isna().any():
            cat_imp += int(df[c].isna().sum())
            mode = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "Unknown")
    report.append({"step": "Imputed missing numeric (median) + categorical (mode)",
                   "n": f"{num_imp} num / {cat_imp} cat"})

    # 11) Derived temporal features for cohort / time-series
    if "Signup_Date" in df:
        df["Signup_Month"] = df["Signup_Date"].dt.to_period("M").dt.to_timestamp()
        df["Tenure_Months"] = ((SNAPSHOT - df["Signup_Date"]).dt.days / 30.44).round(1)
    if "Signup_Date" in df and "Last_Order_Date" in df:
        df["Active_Until_Months"] = ((df["Last_Order_Date"] - df["Signup_Date"]).dt.days / 30.44)
        df["Active_Until_Months"] = df["Active_Until_Months"].clip(lower=0)

    # round target flags to clean labels
    for c in ["Churned", "Premium_Tier_Adoption"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip().str.title()

    df = df.reset_index(drop=True)
    return df, report
