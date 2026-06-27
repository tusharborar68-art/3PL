"""Visual theme: palette, CSS injection, Plotly template, KPI card helper."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- Palette ----
NAVY   = "#102A43"
INK    = "#1B2A3A"
TEAL   = "#14B8A6"
INDIGO = "#6366F1"
AMBER  = "#F59E0B"
SKY    = "#0EA5E9"
VIOLET = "#8B5CF6"
PINK   = "#EC4899"
GREEN  = "#10B981"
RED    = "#EF4444"
SLATE  = "#64748B"
BG     = "#F4F6FA"
CARD   = "#FFFFFF"

SEQ = [TEAL, INDIGO, AMBER, SKY, VIOLET, PINK, GREEN, RED, "#0891B2", "#9333EA"]


def apply_plotly_theme():
    pio.templates["threepl"] = go.layout.Template(
        layout=dict(
            font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=13),
            colorway=SEQ,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(font=dict(size=16, color=NAVY)),
            xaxis=dict(gridcolor="#E6EBF2", zerolinecolor="#E6EBF2", linecolor="#CBD5E1"),
            yaxis=dict(gridcolor="#E6EBF2", zerolinecolor="#E6EBF2", linecolor="#CBD5E1"),
            legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="#E6EBF2", borderwidth=1),
            margin=dict(l=40, r=20, t=50, b=40),
        )
    )
    pio.templates.default = "threepl"


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
    .stApp {{ background:{BG}; }}
    .block-container {{ padding-top:1.6rem; padding-bottom:2rem; max-width:1500px; }}

    /* Hero */
    .hero {{
        background:linear-gradient(120deg,{NAVY} 0%,#173d63 55%,{TEAL} 140%);
        border-radius:18px; padding:22px 28px; margin-bottom:18px;
        box-shadow:0 10px 30px rgba(16,42,67,.18);
    }}
    .hero h1 {{ color:#fff; font-size:1.55rem; font-weight:800; margin:0; letter-spacing:-.02em; }}
    .hero p {{ color:#cfe6f0; margin:.3rem 0 0; font-size:.92rem; }}

    /* KPI cards */
    .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:.2rem 0 1rem; }}
    .kpi {{
        flex:1; min-width:150px; background:{CARD}; border-radius:14px; padding:15px 17px;
        box-shadow:0 4px 14px rgba(16,42,67,.06); border-left:5px solid {TEAL};
        transition:transform .15s ease, box-shadow .15s ease;
    }}
    .kpi:hover {{ transform:translateY(-3px); box-shadow:0 10px 22px rgba(16,42,67,.12); }}
    .kpi .lbl {{ color:{SLATE}; font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }}
    .kpi .val {{ color:{NAVY}; font-size:1.5rem; font-weight:800; margin-top:3px; line-height:1.1; }}
    .kpi .sub {{ font-size:.74rem; font-weight:600; margin-top:3px; }}
    .kpi.t-teal {{ border-left-color:{TEAL}; }} .kpi.t-indigo {{ border-left-color:{INDIGO}; }}
    .kpi.t-amber {{ border-left-color:{AMBER}; }} .kpi.t-violet {{ border-left-color:{VIOLET}; }}
    .kpi.t-sky {{ border-left-color:{SKY}; }} .kpi.t-red {{ border-left-color:{RED}; }}
    .kpi.t-green {{ border-left-color:{GREEN}; }}
    .up {{ color:{GREEN}; }} .down {{ color:{RED}; }}

    /* Section headers */
    .sec {{ color:{NAVY}; font-size:1.05rem; font-weight:700; margin:1.1rem 0 .2rem;
            padding-bottom:.3rem; border-bottom:2px solid #E3E9F1; }}
    .sec span {{ color:{TEAL}; }}
    .hint {{ color:{SLATE}; font-size:.82rem; margin:.1rem 0 .6rem; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
    .stTabs [data-baseweb="tab"] {{
        background:#E8EEF6; border-radius:10px 10px 0 0; padding:9px 18px; font-weight:600; color:{NAVY};
    }}
    .stTabs [aria-selected="true"] {{ background:{NAVY}; color:#fff; }}

    [data-testid="stSidebar"] {{ background:#0E2438; }}
    [data-testid="stSidebar"] * {{ color:#dbe7f0; }}
    [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stMultiSelect label {{ color:#9fc3d8 !important; font-weight:600; }}

    .badge {{ display:inline-block; background:{TEAL}1a; color:{TEAL}; font-weight:700;
              padding:3px 10px; border-radius:20px; font-size:.78rem; margin-right:6px; }}
    .badge.best {{ background:{GREEN}1a; color:{GREEN}; }}
    </style>
    """, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def section(title, hint=None):
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)
    if hint:
        st.markdown(f"<div class='hint'>{hint}</div>", unsafe_allow_html=True)


def kpi_cards(cards):
    """cards: list of dict(label, value, sub=None, tone='teal', up=None)."""
    html = "<div class='kpi-row'>"
    for c in cards:
        tone = c.get("tone", "teal")
        sub = ""
        if c.get("sub"):
            cls = "up" if c.get("up") is True else ("down" if c.get("up") is False else "")
            arrow = "▲ " if c.get("up") is True else ("▼ " if c.get("up") is False else "")
            sub = f"<div class='sub {cls}'>{arrow}{c['sub']}</div>"
        html += (f"<div class='kpi t-{tone}'><div class='lbl'>{c['label']}</div>"
                 f"<div class='val'>{c['value']}</div>{sub}</div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
