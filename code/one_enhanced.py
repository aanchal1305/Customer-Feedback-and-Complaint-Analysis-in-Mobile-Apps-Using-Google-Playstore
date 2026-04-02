# -*- coding: utf-8 -*-
"""
Google Play Analytics Dashboard — Enhanced Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# WordCloud — install with: pip install wordcloud
try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Complaint Pattern Mining & Analysis using Google Playstore Dataset",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL STYLES  — dark, modern, card-based design
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,opsz,wght@0,18..144,300..900;1,18..144,300..900&display=swap');

/* Play Store pastel palette:
   Cyan  #E0F7FA  accent #00ACC1
   Green #E8F5E9  accent #2E7D32
   Red   #FBE9E7  accent #BF360C
   Blue  #E3F2FD  accent #1565C0
   Yellow#FFFDE7  accent #F9A825  */

html, body, [class*="css"] {
    font-family: Merriweather, serif;
}

/* ---- page background — very light cyan tint ---- */
.stApp {
    background: #F0FAFA;
    color: #1a1a2e;
}

/* ---- hide default streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- sidebar — pastel green tint ---- */
[data-testid="stSidebar"] {
    background: #E8F5E9;
    border-right: 2px solid #A5D6A7;
}
[data-testid="stSidebar"] * { color: #1B5E20 !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: rgba(0,172,193,0.08) !important;
    border: 1px solid rgba(0,172,193,0.35) !important;
    border-radius: 10px !important;
    color: #006064 !important;
    font-family: Merriweather, serif !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
    margin-bottom: 6px !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,172,193,0.18) !important;
    border-color: #00ACC1 !important;
    transform: translateX(4px) !important;
}

/* ---- KPI cards — CHANGE 3: bigger font, black text, no hover glow ---- */
.kpi-row { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 160px;
    background: #ffffff;
    border: 1.5px solid #B2EBF2;
    border-radius: 16px;
    padding: 22px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,172,193,0.10);
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #00ACC1, #43A047, #FFB300, #E53935);
}
/* CHANGE 3: no hover effect — removed transform and glow */
.kpi-title {
    font-size: 12px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
    color: #00838F;          /* readable teal — not grey */
    margin-bottom: 10px;
}
.kpi-value {
    font-family: Merriweather, serif;
    font-size: 32px;         /* CHANGE 3: bigger */
    font-weight: 900;
    color: #000000;          /* CHANGE 3: pure black */
    line-height: 1;
}
.kpi-sub { font-size: 12px; color: #374151; margin-top: 6px; }

/* ---- section headers — CHANGE 4: black, not grey ---- */
.section-header {
    font-family: Merriweather, serif;
    font-size: 20px; font-weight: 700;
    color: #000000;          /* CHANGE 4: pure black */
    border-left: 5px solid #00ACC1;
    padding-left: 14px;
    margin: 28px 0 18px;
}

/* ---- chart card wrapper ---- */
.chart-card {
    background: #ffffff;
    border: 1px solid #B2EBF2;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}

/* ---- tab styling — pastel cyan ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #E0F7FA;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #B2EBF2;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #37474F !important;
    font-family: Merriweather, serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #00ACC1 !important;
    color: #ffffff !important;
}

/* ---- dataframe ---- */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ---- inputs / selects ---- */
.stSelectbox > div > div, .stSlider > div {
    background: #E0F7FA !important;
    border: 1px solid #80DEEA !important;
    border-radius: 10px !important;
    color: #000000 !important;
}

/* ---- login card ---- */
.login-card {
    max-width: 440px; margin: 60px auto;
    background: #ffffff;
    border: 1.5px solid #B2EBF2;
    border-radius: 24px; padding: 40px;
}
.login-title {
    font-family: Merriweather, serif;
    font-size: 28px; font-weight: 800;
    color: #006064; text-align: center; margin-bottom: 8px;
}
.login-sub { font-size: 13px; color: #374151; text-align: center; margin-bottom: 28px; }

/* ---- warning / info boxes ---- */
.stAlert { border-radius: 12px !important; }

/* ---- home hero — Play Store gradient cyan→green ---- */
.hero-title {
    font-family: Merriweather, serif;
    font-size: 44px; font-weight: 900;
    background: linear-gradient(135deg, #00ACC1 0%, #43A047 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 12px;
}
.hero-sub { font-size: 16px; color: #374151; max-width: 600px; line-height: 1.7; }
.feature-pill {
    display: inline-block;
    background: #E0F7FA;
    border: 1px solid #80DEEA;
    border-radius: 20px; padding: 6px 14px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
    color: #006064; margin: 4px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB DARK THEME  (used for all mpl/sns charts)
# ============================================================

plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#fafafa",
    "axes.edgecolor":    "#e5e7eb",
    "axes.labelcolor":   "#374151",
    "axes.titlecolor":   "#1a1a2e",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#374151",
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.6,
    "figure.titlesize":  14,
    "font.family":       "DejaVu Sans",
})

# Play Store pastel accent colours
ACCENT   = "#00ACC1"   # cyan  (Play Store icon)
ACCENT2  = "#43A047"   # green (Play Store icon)
ACCENT3  = "#FFB300"   # yellow/amber (Play Store icon)
WARN     = "#E53935"   # red   (Play Store icon)
PALETTES = {
    "playstore": ["#00ACC1","#43A047","#FFB300","#E53935","#1E88E5","#8E24AA"],
    "gradient10": px.colors.sequential.Teal,
}


# ============================================================
# LOGIN SYSTEM
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}


def login():
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")
    if st.button("Sign In", use_container_width=True):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.success("Welcome back! Redirecting…")
            st.rerun()
        else:
            st.error("Invalid username or password.")


def register():
    new_user = st.text_input("Choose a Username", placeholder="New username")
    new_pass = st.text_input("Choose a Password", type="password", placeholder="New password")
    if st.button("Create Account", use_container_width=True):
        if new_user in st.session_state.users:
            st.warning("Username already taken.")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("Account created! Switch to the Login tab.")


if not st.session_state.logged_in:
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px;'>
        <div style='font-family:Merriweather,serif; font-size:42px; font-weight:800;
                    background:linear-gradient(135deg,#c4b5fd,#93c5fd);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Customer Complaint Pattern Mining & Analysis using Google Playstore Dataset
        </div>
        <div style='color:#4b5563; font-size:14px; margin-top:8px;'>
            Sign in to explore the dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        tab_l, tab_r = st.tabs(["  Login  ", "  Register  "])
        with tab_l:
            login()
        with tab_r:
            register()
    st.stop()


# ============================================================
# LOAD & CLEAN DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    df = df.drop_duplicates()
    df = df[df["Rating"].notna()]

    df["Installs"] = (df["Installs"].astype(str)
                      .str.replace("+", "", regex=False)
                      .str.replace(",", "", regex=False))
    df = df[df["Installs"].str.isnumeric()]
    df["Installs"] = df["Installs"].astype(int)

    df["Price"] = df["Price"].str.replace("$", "", regex=False).astype(float)

    def clean_size(x):
        if "M" in str(x):  return float(x.replace("M", ""))
        elif "k" in str(x): return float(x.replace("k", "")) / 1000
        return np.nan

    df["Size"] = df["Size"].apply(clean_size)
    df["Size"].fillna(df["Size"].median(), inplace=True)

    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
    df = df[df["Reviews"].notna()]
    df["Reviews"] = df["Reviews"].astype(int)

    # Derived columns
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")
    df["Update Year"]  = df["Last Updated"].dt.year
    df["Install Tier"] = pd.cut(
        df["Installs"],
        bins=[0, 1_000, 10_000, 100_000, 1_000_000, 1_000_000_000],
        labels=["< 1K", "1K–10K", "10K–100K", "100K–1M", "1M+"]
    )
    return df


@st.cache_data
def load_reviews():
    rv = pd.read_csv("clean_googleplay_reviews.csv")
    rv["clean_review"] = rv["clean_review"].fillna("")
    return rv


df         = load_data()
reviews_df = load_reviews()


# ============================================================
# TRAIN MODEL — runs once at startup, cached globally
# Defined here (not inside the page block) so the LabelEncoders
# are stable across reruns and predict_proba always works correctly.
# ============================================================

@st.cache_resource
def train_model(_dataframe):
    pred_df = _dataframe.copy()
    pred_df["Success"] = np.where(pred_df["Installs"] > 1_000_000, 1, 0)

    le_cat  = LabelEncoder()
    le_cr   = LabelEncoder()
    le_type = LabelEncoder()
    pred_df["Category_enc"]      = le_cat.fit_transform(pred_df["Category"])
    pred_df["ContentRating_enc"] = le_cr.fit_transform(pred_df["Content Rating"])
    pred_df["Type_enc"]          = le_type.fit_transform(pred_df["Type"])

    # Reviews excluded: it has 74% feature importance but reflects
    # existing popularity, not a pre-launch developer decision.
    # Removing it gives varied, meaningful predictions for new apps.
    features = pred_df[[
        "Category_enc", "Rating", "Size",
        "Price", "ContentRating_enc", "Type_enc"
    ]]
    target = pred_df["Success"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred_test  = model.predict(X_test)
    proba_test = model.predict_proba(X_test)[:, 1]

    return model, le_cat, le_cr, le_type, X_test, y_test, pred_test, proba_test


# Train once — unpack into module-level variables
model, le_cat, le_cr, le_type, X_test, y_test, pred_test, proba_test = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/888/888857.png' width='64'/>
        <div style='font-family:Merriweather,serif; font-size:16px; font-weight:700;
                    color:#006064; margin-top:10px;'>Play Analytics</div>
        <div style='font-size:11px; color:#00838F;'>Dashboard v2.0</div>
    </div>
    <hr style='border-color:rgba(139,92,246,0.2); margin:12px 0;'/>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    pages = {
        "Home":                    "Home",
        "App Analytics":           "App Analytics",
        "Review Intelligence":     "User Review Intelligence",
        "Success Prediction":      "App Success Prediction",
    }
    
    for label, key in pages.items():
        if st.button(label):
            st.session_state.page = key

    st.markdown("<hr style='border-color:rgba(139,92,246,0.2); margin:16px 0;'/>", unsafe_allow_html=True)
    st.markdown(
    "<div style=\"font-size:12px; color:#1B5E20; text-align:center;\">"
    "<b>Contact Us: +91 9004905131</b>"
    "</div>",
    unsafe_allow_html=True)

    # Quick dataset stats
    st.markdown(f"""
    <div style='font-size:11px; color:#6b7280; text-align:center;'>
        <div style='color:#00838F; font-weight:600; margin-bottom:6px;'>DATASET STATS</div>
        {len(df):,} apps &nbsp;·&nbsp; {df['Category'].nunique()} categories<br>
        {len(reviews_df):,} reviews
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

page = st.session_state.page


# ============================================================
# HELPER — reusable KPI row
# ============================================================

def kpi_row(cards):
    """cards = list of (title, value, subtitle_optional)
    Renders a horizontal row of styled KPI cards.
    Builds the entire block as one HTML string so Streamlit renders
    it correctly (passing it in fragments across columns breaks rendering).
    """
    html = '<div class="kpi-row">'
    for c in cards:
        sub = f'<div class="kpi-sub">{c[2]}</div>' if len(c) > 2 else ""
        html += (
            '<div class="kpi-card">'
            f'<div class="kpi-title">{c[0]}</div>'
            f'<div class="kpi-value">{c[1]}</div>'
            f'{sub}'
            '</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown("""
    <div style='padding: 40px 0 20px;'>
        <div class='hero-title'>Google Play<br>Store Analytics</div>
        <div class='hero-sub'>
            Deep-dive into 10,000+ apps across every category on the Google Play Store —
            uncover market trends, decode user sentiment, and predict what makes an app succeed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 3 Objectives — using st.columns so no HTML rendering issues ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Project Objectives")

    obj1, obj2, obj3 = st.columns(3, gap="large")

    with obj1:
        st.markdown(
            "<div style=\"background:#E0F7FA;border:2px solid #00ACC1;"
            "border-radius:16px;padding:28px;height:100%;\">"
            "<div style=\"font-size:12px;font-weight:700;letter-spacing:1.2px;"
            "text-transform:uppercase;color:#00838F;margin-bottom:10px;\">Objective 1</div>"
            "<div style=\"font-size:17px;font-weight:700;color:#000000;margin-bottom:12px;\">"
            "Understand the App Market</div>"
            "<div style=\"font-size:13px;color:#374151;line-height:1.8;\">"
            "Analyse over 10,000 Google Play Store apps to identify key trends in ratings, "
            "installs, pricing, and category performance — giving developers and marketers "
            "a data-driven view of the competitive landscape."
            "</div></div>",
            unsafe_allow_html=True
        )

    with obj2:
        st.markdown(
            "<div style=\"background:#E8F5E9;border:2px solid #43A047;"
            "border-radius:16px;padding:28px;height:100%;\">"
            "<div style=\"font-size:12px;font-weight:700;letter-spacing:1.2px;"
            "text-transform:uppercase;color:#2E7D32;margin-bottom:10px;\">Objective 2</div>"
            "<div style=\"font-size:17px;font-weight:700;color:#000000;margin-bottom:12px;\">"
            "Decode User Sentiment</div>"
            "<div style=\"font-size:13px;color:#374151;line-height:1.8;\">"
            "Mine 37,000+ user reviews using NLP-based sentiment analysis to surface "
            "what users love, what frustrates them, and which apps generate the most "
            "complaints — broken down by category and keyword."
            "</div></div>",
            unsafe_allow_html=True
        )

    with obj3:
        st.markdown(
            "<div style=\"background:#FBE9E7;border:2px solid #E53935;"
            "border-radius:16px;padding:28px;height:100%;\">"
            "<div style=\"font-size:12px;font-weight:700;letter-spacing:1.2px;"
            "text-transform:uppercase;color:#BF360C;margin-bottom:10px;\">Objective 3</div>"
            "<div style=\"font-size:17px;font-weight:700;color:#000000;margin-bottom:12px;\">"
            "Predict App Success</div>"
            "<div style=\"font-size:13px;color:#374151;line-height:1.8;\">"
            "Build a machine learning classifier that predicts whether a new app will "
            "exceed 1 million installs — empowering developers to validate their idea "
            "before launch with personalised improvement recommendations."
            "</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards — using st.columns to avoid single-quote/triple-quote conflicts
    fc1, fc2, fc3, fc4 = st.columns(4, gap="small")

    with fc1:
        st.markdown(
            "<div style=\"background:#E0F7FA;border:1px solid #80DEEA;"
            "border-radius:16px;padding:20px;\">"
            "<div style=\"font-size:15px;font-weight:700;color:#000000;margin-bottom:6px;\">App Analytics</div>"
            "<div style=\"font-size:12px;color:#4b5563;line-height:1.6;\">Rating distributions, "
            "category benchmarks, market share treemaps and install-tier breakdowns.</div>"
            "</div>",
            unsafe_allow_html=True
        )
    with fc2:
        st.markdown(
            "<div style=\"background:#E8F5E9;border:1px solid #A5D6A7;"
            "border-radius:16px;padding:20px;\">"
            "<div style=\"font-size:15px;font-weight:700;color:#000000;margin-bottom:6px;\">Review Intelligence</div>"
            "<div style=\"font-size:12px;color:#4b5563;line-height:1.6;\">Sentiment breakdown, "
            "complaint heatmaps, per-app review scoring and keyword search.</div>"
            "</div>",
            unsafe_allow_html=True
        )
    with fc3:
        st.markdown(
            "<div style=\"background:#FFFDE7;border:1px solid #FFE082;"
            "border-radius:16px;padding:20px;\">"
            "<div style=\"font-size:15px;font-weight:700;color:#000000;margin-bottom:6px;\">Success Prediction</div>"
            "<div style=\"font-size:12px;color:#4b5563;line-height:1.6;\">Random Forest classifier "
            "with confusion matrix, feature importance and per-class metrics.</div>"
            "</div>",
            unsafe_allow_html=True
        )
    with fc4:
        st.markdown(
            "<div style=\"background:#FBE9E7;border:1px solid #FFAB91;"
            "border-radius:16px;padding:20px;\">"
            "<div style=\"font-size:15px;font-weight:700;color:#000000;margin-bottom:6px;\">Top Apps Explorer</div>"
            "<div style=\"font-size:12px;color:#4b5563;line-height:1.6;\">Filter by category, type "
            "and installs to surface the highest-rated apps in any niche.</div>"
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# APP ANALYTICS
# ============================================================

elif page == "App Analytics":

    st.markdown('<div class="hero-title" style="font-size:36px;color:#000000;-webkit-text-fill-color:#000000;">App Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset Overview",
        "EDA",
        "Market Insights",
        "Top Apps",
    ])

    # ----------------------------------------------------------
    # TAB 1 — Dataset Overview
    # ----------------------------------------------------------
    with tab1:
        section_header("Dataset Overview")

        kpi_row([
            ("Rows",         f"{df.shape[0]:,}"),
            ("Columns",      df.shape[1]),
            ("Categories",   df["Category"].nunique()),
            ("Free Apps",    f"{(df['Type']=='Free').sum():,}"),
            ("Paid Apps",    f"{(df['Type']=='Paid').sum():,}"),
            ("Missing Vals", int(df.isnull().sum().sum())),
        ])

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("**Sample Records**")
            st.dataframe(df.head(10), use_container_width=True, height=280)
        with col_b:
            st.markdown("**Column Types**")
            dtype_df = pd.DataFrame({
                "Column": df.dtypes.index,
                "Type": df.dtypes.astype(str).values
            })
            st.dataframe(dtype_df, use_container_width=True, height=280)

        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(2), use_container_width=True)


    # ----------------------------------------------------------
    # TAB 2 — EDA
    # ----------------------------------------------------------
    with tab2:
        section_header("Exploratory Data Analysis")

        kpi_row([
            ("Avg Rating",    round(df["Rating"].mean(), 2),       "all apps"),
            ("Median Installs", f"{df['Installs'].median()/1e6:.1f}M", "median"),
            ("Avg Size",      f"{df['Size'].mean():.1f} MB",       ""),
            ("Avg Reviews",   f"{int(df['Reviews'].mean()):,}",    "per app"),
        ])

        # -------------------------------------------------------
        # [EXTRA KPI] -- To activate, remove the #> prefix from each line below.
        # This KPI shows what % of all apps are Free.
        # -------------------------------------------------------
        #> kpi_row([
        #>     ("Free App %",
        #>      f"{(df['Type']=='Free').mean()*100:.1f}%",
        #>      "of all apps"),
        #> ])
        # -------------------------------------------------------

        # -------------------------------------------------------
        # [EXTRA CHART] -- To activate, remove the #> prefix from each line below.
        # This shows a box plot of ratings across top 8 categories.
        # -------------------------------------------------------
        #> st.subheader("Rating Distribution by Category (Box Plot)")
        #> top8_cats = df["Category"].value_counts().head(8).index
        #> fig_extra, ax_extra = plt.subplots(figsize=(9, 4))
        #> sns.boxplot(
        #>     x="Category", y="Rating",
        #>     data=df[df["Category"].isin(top8_cats)],
        #>     palette=["#00ACC1","#43A047","#FFB300","#E53935",
        #>              "#1E88E5","#8E24AA","#00ACC1","#43A047"],
        #>     ax=ax_extra
        #> )
        #> ax_extra.set_xticklabels(
        #>     ax_extra.get_xticklabels(), rotation=30, ha="right", fontsize=8
        #> )
        #> ax_extra.set_title("Rating Distribution - Top 8 Categories")
        #> fig_extra.tight_layout()
        #> st.pyplot(fig_extra)
        # -------------------------------------------------------

        # Row 1
        col1, col2 = st.columns(2)

        with col1:
            # Rating distribution — histogram + KDE
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df["Rating"], bins=25, kde=True, color="#00ACC1",
                         edgecolor="none", alpha=0.7, ax=ax)
            ax.axvline(df["Rating"].mean(), color="#E53935", linewidth=1.5,
                       linestyle="--", label=f"Mean: {df['Rating'].mean():.2f}")
            ax.legend(fontsize=9)
            ax.set_title("Rating Distribution")
            ax.set_xlabel("Rating"); ax.set_ylabel("Count")
            fig.tight_layout()
            st.pyplot(fig)

        with col2:
            # Top 10 categories by app count — horizontal bar
            top_cats = df["Category"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(top_cats)))
            bars = ax.barh(top_cats.index[::-1], top_cats.values[::-1],
                           color=colors[::-1], edgecolor="none", height=0.65)
            for bar, val in zip(bars, top_cats.values[::-1]):
                ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                        f"{val}", va="center", fontsize=8, color="#9ca3af")
            ax.set_title("Top 10 Categories by App Count")
            ax.set_xlabel("Number of Apps")
            ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)

        # Row 2
        col3, col4 = st.columns(2)

        with col3:
            # Free vs Paid — donut chart via Plotly
            type_counts = df["Type"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.55,
                marker_colors=["#00ACC1", "#43A047"],
                textfont_size=13,
            ))
            fig_pie.update_layout(
                title="Free vs Paid Apps",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=320,
                legend=dict(font=dict(color="#000000")),
                margin=dict(t=40, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col4:
            # Content Rating breakdown
            cr_counts = df["Content Rating"].value_counts()
            fig_bar = px.bar(
                x=cr_counts.values, y=cr_counts.index,
                orientation="h",
                color=cr_counts.values,
                color_continuous_scale="Plasma",
                labels={"x": "Apps", "y": "Content Rating"},
                title="Content Rating Breakdown",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=320,
                coloraxis_showscale=False,
                margin=dict(t=40, b=0, l=0, r=0),
                yaxis=dict(gridcolor="#E0F7FA"),
                xaxis=dict(gridcolor="#E0F7FA"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Row 3 — Rating by Category boxplot (Plotly — interactive)
        section_header("Rating Distribution by Category")
        top12_cats = df["Category"].value_counts().head(12).index
        df_box = df[df["Category"].isin(top12_cats)].copy()
        df_box["CAT"] = df_box["Category"].str.replace("_", " ").str.title()
        fig_box = px.box(
            df_box, x="Rating", y="CAT",
            color="CAT",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"Rating": "Rating", "CAT": ""},
            title="Rating Distribution — Top 12 Categories",
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=480, showlegend=False,
            xaxis=dict(gridcolor="#E0F7FA"),
            yaxis=dict(gridcolor="#E0F7FA"),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Row 4 — Size vs Rating scatter
        col5, col6 = st.columns(2)
        with col5:
            fig_sc = px.scatter(
                df.sample(min(2000, len(df)), random_state=42),
                x="Size", y="Rating",
                color="Type",
                color_discrete_map={"Free": "#00ACC1", "Paid": "#43A047"},
                opacity=0.5,
                title="App Size vs Rating",
                labels={"Size": "Size (MB)", "Rating": "Rating"},
            )
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=340,
                legend=dict(font=dict(color="#000000")),
                xaxis=dict(gridcolor="#E0F7FA"),
                yaxis=dict(gridcolor="#E0F7FA"),
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col6:
            # Install tier pie
            tier_counts = df["Install Tier"].value_counts().sort_index()
            fig_tier = px.bar(
                x=tier_counts.index.astype(str),
                y=tier_counts.values,
                color=tier_counts.values,
                color_continuous_scale="Viridis",
                title="Apps by Install Tier",
                labels={"x": "Install Range", "y": "Apps"},
            )
            fig_tier.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=340,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#E0F7FA"),
                yaxis=dict(gridcolor="#E0F7FA"),
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_tier, use_container_width=True)


    # ----------------------------------------------------------
    # TAB 3 — Market Insights
    # ----------------------------------------------------------
    with tab3:
        section_header("Market Insights")

        kpi_row([
            ("Total Installs",    f"{df['Installs'].sum()/1e9:.2f}B",  "cumulative"),
            ("Avg Installs/App",  f"{int(df['Installs'].mean()):,}",   "mean"),
            ("Max Installs",      f"{df['Installs'].max()/1e9:.1f}B",  "single app"),
            ("Paid Revenue Est.", f"${(df[df['Type']=='Paid']['Price']*df[df['Type']=='Paid']['Installs']).sum()/1e6:.0f}M", "installs × price"),
        ])

        col1, col2 = st.columns(2)

        with col1:
            # Treemap — installs by category
            cat_inst = (df.groupby("Category")["Installs"]
                        .sum().reset_index()
                        .sort_values("Installs", ascending=False)
                        .head(20))
            cat_inst["Category_clean"] = cat_inst["Category"].str.replace("_", " ").str.title()
            fig_tm = px.treemap(
                cat_inst,
                path=["Category_clean"],
                values="Installs",
                color="Installs",
                color_continuous_scale="Plasma",
                title="Installs by Category (Treemap)",
            )
            fig_tm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=380,
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_tm, use_container_width=True)

        with col2:
            # Avg rating per category — top 15
            cat_rating = (df.groupby("Category")["Rating"]
                          .mean().sort_values(ascending=False).head(15))
            fig_cr = px.bar(
                x=cat_rating.values,
                y=cat_rating.index.str.replace("_"," ").str.title(),
                orientation="h",
                color=cat_rating.values,
                color_continuous_scale="Viridis",
                title="Avg Rating by Category (Top 15)",
                labels={"x": "Avg Rating", "y": ""},
            )
            fig_cr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=380,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#E0F7FA", range=[3, 5]),
                yaxis=dict(gridcolor="#E0F7FA"),
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_cr, use_container_width=True)

        # Scatter — Reviews vs Installs (log scale)
        section_header("Reviews vs Installs")
        df_sc2 = df[(df["Reviews"] > 0) & (df["Installs"] > 0)].copy()
        df_sc2 = df_sc2.sample(min(3000, len(df_sc2)), random_state=1)
        df_sc2["Category_clean"] = df_sc2["Category"].str.replace("_"," ").str.title()
        fig_rv = px.scatter(
            df_sc2,
            x="Installs", y="Reviews",
            color="Type",
            color_discrete_map={"Free":"#00ACC1","Paid":"#FFB300"},
            hover_name="App",
            hover_data={"Rating": True, "Category_clean": True},
            log_x=True, log_y=True,
            opacity=0.55,
            title="Reviews vs Installs (log scale) — coloured by Type",
            labels={"Installs":"Installs (log)","Reviews":"Reviews (log)"},
            trendline="ols",
        )
        fig_rv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=420,
            legend=dict(font=dict(color="#000000")),
            xaxis=dict(gridcolor="#E0F7FA"),
            yaxis=dict(gridcolor="#E0F7FA"),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_rv, use_container_width=True)

        # Price distribution for paid apps
        col3, col4 = st.columns(2)
        with col3:
            paid_df = df[(df["Type"] == "Paid") & (df["Price"] < 30)]
            fig_price = px.histogram(
                paid_df, x="Price", nbins=30,
                color_discrete_sequence=["#00ACC1"],
                title="Paid App Price Distribution (< $30)",
                labels={"Price": "Price (USD)"},
            )
            fig_price.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=320,
                bargap=0.05,
                xaxis=dict(gridcolor="#E0F7FA"),
                yaxis=dict(gridcolor="#E0F7FA"),
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_price, use_container_width=True)

        with col4:
            # Average installs: Free vs Paid per category (top 8)
            top8 = df["Category"].value_counts().head(8).index
            fp = (df[df["Category"].isin(top8)]
                  .groupby(["Category", "Type"])["Installs"]
                  .mean().reset_index())
            fp["Category"] = fp["Category"].str.replace("_"," ").str.title()
            fig_fp = px.bar(
                fp, x="Category", y="Installs", color="Type",
                barmode="group",
                color_discrete_map={"Free":"#00ACC1","Paid":"#FFB300"},
                title="Avg Installs: Free vs Paid (Top 8 Categories)",
            )
            fig_fp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=320,
                legend=dict(font=dict(color="#000000")),
                xaxis=dict(gridcolor="#E0F7FA", tickangle=-35),
                yaxis=dict(gridcolor="#E0F7FA"),
                margin=dict(t=40, b=0, l=40, r=0),
            )
            st.plotly_chart(fig_fp, use_container_width=True)


    # ----------------------------------------------------------
    # TAB 4 — Top Apps
    # ----------------------------------------------------------
    with tab4:
        section_header("Top Apps Explorer")

        # ------- FILTERS ROW -------
        # Active filter: Category
        # To add the App Type filter, uncomment the block marked [APP TYPE FILTER]
        fcol1, fcol2, fcol3 = st.columns([3, 2, 1])

        with fcol1:
            selected_category = st.selectbox(
                "Category",
                ["All"] + sorted(df["Category"].unique())
            )

        with fcol2:
            # FIX 2: Sort By filter — prevents all bars showing the same 5.0 value
            sort_by = st.selectbox(
                "Sort By",
                ["Rating", "Installs", "Reviews"],
                index=0,
                key="top_apps_sort"
            )

        with fcol3:
            top_n = st.selectbox("Show Top N", [10, 20, 30, 50], index=1)

        # -------------------------------------------------------
        # [APP TYPE FILTER] -- To activate, remove the #> prefix from each line below.
        # Also change the columns line above to: fcol1, fcol2, fcol3 = st.columns([3, 2, 1])
        # -------------------------------------------------------
        #> with fcol2:
        #>     selected_type = st.selectbox(
        #>         "App Type",
        #>         ["All", "Free", "Paid"]
        #>     )
        # When activating, also uncomment these lines:
        # if selected_type != "All":
        #     filtered_df = filtered_df[filtered_df["Type"] == selected_type]
        # if selected_type != "All":
        #     chart_title += f" ({selected_type})"
        # -------------------------------------------------------

        # ------- APPLY FILTERS -------
        filtered_df = df.copy()
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["Category"] == selected_category]

        # FIX 2b: sort by user-selected column, not always Rating
        top_apps = filtered_df.sort_values(sort_by, ascending=False).head(top_n)

        if top_apps.empty:
            st.warning("No apps match the selected filters.")
        else:
            kpi_row([
                ("Apps Shown",   len(top_apps)),
                ("Avg Rating",   round(top_apps["Rating"].mean(), 2)),
                ("Avg Installs", f"{int(top_apps['Installs'].mean()):,}"),
                ("Paid in Set",  f"{(top_apps['Type']=='Paid').sum()}"),
            ])

            st.dataframe(
                top_apps[["App","Category","Type","Installs","Rating","Reviews"]],
                use_container_width=True, height=280
            )

            # FIX 2c: title and axes now reflect the Sort By selection
            chart_title = f"Top {len(top_apps)} Apps by {sort_by}"
            if selected_category != "All":
                chart_title += f" — {selected_category.replace('_',' ').title()}"
            # if selected_type != "All":
            #     chart_title += f" ({selected_type})"

            # Plotly horizontal bar — interactive
            fig_ta = px.bar(
                top_apps.sort_values(sort_by),
                x=sort_by, y="App",
                color=sort_by,
                color_continuous_scale="Plasma",
                orientation="h",
                hover_data={"Installs": True, "Reviews": True, "Rating": True, "Type": True},
                title=chart_title,
            )
            # Only hard-cap x-axis to 5.5 when sorting by Rating
            xrange = [0, 5.5] if sort_by == "Rating" else None
            fig_ta.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000",
                height=max(380, len(top_apps) * 28),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#E0F7FA", range=xrange),
                yaxis=dict(gridcolor="#E0F7FA", tickfont=dict(size=10)),
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_ta, use_container_width=True)


# ============================================================
# USER REVIEW INTELLIGENCE
# ============================================================

elif page == "User Review Intelligence":

    st.markdown('<div class="hero-title" style="font-size:36px;color:#000000;-webkit-text-fill-color:#000000;">Review Intelligence</div>', unsafe_allow_html=True)

    # KPIs
    pos_pct  = (reviews_df["Sentiment"] == "Positive").mean() * 100
    neg_pct  = (reviews_df["Sentiment"] == "Negative").mean() * 100
    neu_pct  = (reviews_df["Sentiment"] == "Neutral").mean()  * 100
    comp_pct = reviews_df["complaint"].mean() * 100

    kpi_row([
        ("Total Reviews",    f"{len(reviews_df):,}"),
        ("Positive",         f"{pos_pct:.1f}%",   f"{int(pos_pct/100*len(reviews_df)):,} reviews"),
        ("Negative",         f"{neg_pct:.1f}%",   f"{int(neg_pct/100*len(reviews_df)):,} reviews"),
        ("Neutral",          f"{neu_pct:.1f}%",   f"{int(neu_pct/100*len(reviews_df)):,} reviews"),
        ("Complaint Rate",   f"{comp_pct:.1f}%",  "flagged reviews"),
        ("Avg Length",       f"{int(reviews_df['review_length'].mean())} w", "words per review"),
    ])

    # ---- FIX 3: Sentiment section ----
    # The dataset is imbalanced (64% Positive, 22% Negative, 14% Neutral).
    # Raw count charts are misleading — Positive always dominates visually.
    # Fix: add a Category filter and show PERCENTAGES so every filtered
    # view is fair, regardless of how many reviews exist in that group.

    section_header("Sentiment Distribution")

    sent_cat_filter = st.selectbox(
        "Filter by Category",
        ["All Categories"] + sorted(df["Category"].unique()),
        key="sent_cat_filter"
    )

    # Filter reviews to selected category
    if sent_cat_filter == "All Categories":
        sent_reviews = reviews_df.copy()
    else:
        apps_in_sent_cat = df[df["Category"] == sent_cat_filter]["App"].unique()
        sent_reviews = reviews_df[reviews_df["App"].isin(apps_in_sent_cat)]

    # Compute percentage share — not raw counts
    sent_order = ["Positive", "Neutral", "Negative"]
    sent_pct = (
        sent_reviews["Sentiment"].value_counts(normalize=True)
        .mul(100).reindex(sent_order).fillna(0).reset_index()
    )
    sent_pct.columns = ["Sentiment", "Percentage"]
    sent_raw = (
        sent_reviews["Sentiment"].value_counts()
        .reindex(sent_order).fillna(0).astype(int).reset_index()
    )
    sent_raw.columns = ["Sentiment", "Count"]
    sent_combined = sent_pct.merge(sent_raw, on="Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        # Donut shows percentage slices — fair across any filter
        fig_sent = go.Figure(go.Pie(
            labels=sent_combined["Sentiment"],
            values=sent_combined["Percentage"],
            hole=0.55,
            marker_colors=["#10b981", "#f59e0b", "#ef4444"],
            textfont_size=14,
            pull=[0.04, 0.04, 0.04],
            texttemplate="%{label}<br>%{value:.1f}%",
        ))
        fig_sent.update_layout(
            title=f"Sentiment Share — {sent_cat_filter} ({len(sent_reviews):,} reviews)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=360,
            legend=dict(font=dict(color="#000000")),
            margin=dict(t=50, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        # Percentage bar chart — clearly shows relative proportions
        fig_comp = px.bar(
            sent_combined,
            x="Sentiment", y="Percentage",
            color="Sentiment",
            color_discrete_map={"Positive":"#43A047","Negative":"#E53935","Neutral":"#FFB300"},
            text=sent_combined["Percentage"].apply(lambda x: f"{x:.1f}%"),
            title="Sentiment % (not raw counts)",
            labels={"Percentage": "% of Reviews"},
        )
        fig_comp.update_traces(textposition="outside")
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=360, showlegend=False,
            yaxis=dict(gridcolor="#E0F7FA", range=[0, sent_combined["Percentage"].max() + 10]),
            xaxis=dict(gridcolor="#E0F7FA"),
            margin=dict(t=50, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ---- Row 2: Review length distribution by sentiment ----
    section_header("Review Length Analysis")
    fig_rl = px.violin(
        reviews_df[reviews_df["review_length"] < 100],
        x="Sentiment", y="review_length",
        color="Sentiment",
        color_discrete_map={"Positive":"#43A047","Negative":"#E53935","Neutral":"#FFB300"},
        box=True, points=False,
        title="Review Length Distribution by Sentiment (words)",
        labels={"review_length": "Review Length (words)"},
    )
    fig_rl.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#000000", height=380, showlegend=False,
        xaxis=dict(gridcolor="#E0F7FA"),
        yaxis=dict(gridcolor="#E0F7FA"),
        margin=dict(t=40, b=0, l=0, r=0),
    )
    st.plotly_chart(fig_rl, use_container_width=True)

    # ---- Row 3: Top apps by avg sentiment score ----
    section_header("Top Apps by Sentiment Score")

    # Encode sentiment numerically for scoring
    sent_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    reviews_df["sent_score"] = reviews_df["Sentiment"].map(sent_map)
    app_scores = (reviews_df.groupby("App")
                  .agg(avg_score=("sent_score","mean"),
                       total=("sent_score","count"))
                  .reset_index())
    app_scores = app_scores[app_scores["total"] >= 5]  # min 5 reviews
    top_pos = app_scores.nlargest(10, "avg_score")
    top_neg = app_scores.nsmallest(10, "avg_score")

    col3, col4 = st.columns(2)
    with col3:
        fig_tp = px.bar(
            top_pos.sort_values("avg_score"),
            x="avg_score", y="App",
            orientation="h",
            color="avg_score",
            color_continuous_scale="Greens",
            title="Top 10 Most Positive Apps",
            labels={"avg_score": "Avg Sentiment Score"},
        )
        fig_tp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=340,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#E0F7FA"),
            yaxis=dict(gridcolor="#E0F7FA", tickfont=dict(size=9)),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_tp, use_container_width=True)

    with col4:
        fig_tn = px.bar(
            top_neg.sort_values("avg_score", ascending=False),
            x="avg_score", y="App",
            orientation="h",
            color="avg_score",
            color_continuous_scale="Reds_r",
            title="Top 10 Most Negative Apps",
            labels={"avg_score": "Avg Sentiment Score"},
        )
        fig_tn.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=340,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#E0F7FA"),
            yaxis=dict(gridcolor="#E0F7FA", tickfont=dict(size=9)),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_tn, use_container_width=True)

    # ---- Top Complained Apps by Category ----
    section_header("Top Complained Apps by Category")

    sel_cat = st.selectbox("Select Category", sorted(df["Category"].unique()), key="rev_cat")
    apps_in_cat = df[df["Category"] == sel_cat]["App"].unique()
    filt_rev = reviews_df[reviews_df["App"].isin(apps_in_cat)]

    complaints = (filt_rev.groupby("App")["complaint"]
                  .sum().sort_values(ascending=False).head(10))

    if complaints.empty:
        st.info("No complaint data for this category.")
    else:
        fig_cl = px.bar(
            x=complaints.values,
            y=complaints.index,
            orientation="h",
            color=complaints.values,
            color_continuous_scale="Reds",
            title=f"Top Complained Apps — {sel_cat.replace('_',' ').title()}",
            labels={"x": "Complaint Count", "y": "App"},
        )
        fig_cl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=360,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#E0F7FA"),
            yaxis=dict(gridcolor="#E0F7FA", tickfont=dict(size=10)),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_cl, use_container_width=True)

    # ---- Word Cloud ----
    section_header("Word Cloud by Category")
    st.markdown(
        "<div style='font-size:13px;color:#374151;margin-bottom:16px;'>"
        "The word cloud shows the most frequently used words in reviews "
        "for the selected category. Larger words appear more often.</div>",
        unsafe_allow_html=True
    )

    wc_col1, wc_col2 = st.columns([2, 1])
    with wc_col1:
        wc_category = st.selectbox(
            "Select Category for Word Cloud",
            sorted(df["Category"].unique()),
            key="wc_cat"
        )
    with wc_col2:
        wc_sentiment = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive", "Negative", "Neutral"],
            key="wc_sent"
        )

    # Gather reviews for chosen category and sentiment
    wc_apps = df[df["Category"] == wc_category]["App"].unique()
    wc_reviews = reviews_df[reviews_df["App"].isin(wc_apps)]
    if wc_sentiment != "All":
        wc_reviews = wc_reviews[wc_reviews["Sentiment"] == wc_sentiment]

    wc_text = " ".join(wc_reviews["clean_review"].dropna().tolist())

    if not wc_text.strip():
        st.info("No reviews found for this combination. Try a different category or sentiment.")
    elif not WORDCLOUD_AVAILABLE:
        st.warning(
            "WordCloud library is not installed. "
            "Run  pip install wordcloud  in your terminal, then restart the app."
        )
    else:
        # Colour function — Play Store palette
        import random
        playstore_colors = ["#00ACC1", "#43A047", "#FFB300", "#E53935", "#1E88E5", "#8E24AA"]
        def playstore_color_func(word, font_size, position, orientation,
                                 random_state=None, **kwargs):
            return random.choice(playstore_colors)

        wc = WordCloud(
            width=900,
            height=420,
            background_color="white",
            stopwords=STOPWORDS,
            color_func=playstore_color_func,
            max_words=120,
            collocations=False,
            prefer_horizontal=0.85,
        ).generate(wc_text)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 4.5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title(
            f"Word Cloud — {wc_category.replace('_', ' ').title()} "
            f"({'All Sentiments' if wc_sentiment == 'All' else wc_sentiment})",
            fontsize=13, pad=12
        )
        fig_wc.patch.set_facecolor("white")
        fig_wc.tight_layout()
        st.pyplot(fig_wc)

        # Top 10 most frequent words as a bar chart alongside
        from collections import Counter
        stop = STOPWORDS | {"app", "apps", "game", "games", "one", "use",
                             "get", "also", "like", "good", "great", "well",
                             "really", "very", "much", "even", "still", "would"}
        words = [w.lower() for w in wc_text.split() if len(w) > 2 and w.lower() not in stop]
        freq = pd.DataFrame(Counter(words).most_common(15),
                            columns=["Word", "Count"])
        fig_freq = px.bar(
            freq.sort_values("Count"),
            x="Count", y="Word",
            orientation="h",
            color="Count",
            color_continuous_scale="Teal",
            title=f"Top 15 Words — {wc_category.replace('_',' ').title()}",
            labels={"Count": "Frequency", "Word": ""},
        )
        fig_freq.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=420,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#E0F7FA"),
            yaxis=dict(gridcolor="#E0F7FA", tickfont=dict(size=10)),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("---")

    # ---- Keyword Search ----
    section_header("Search Reviews by Keyword")

    keyword = st.text_input("Enter keyword", placeholder="e.g. crash, update, ads…")
    if keyword:
        filt_kw = reviews_df[
            reviews_df["clean_review"].str.contains(keyword, case=False, na=False)
        ]
        if filt_kw.empty:
            st.warning("No reviews found for that keyword.")
        else:
            # Sentiment breakdown of matching reviews
            kw_sent = filt_kw["Sentiment"].value_counts()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Matching Reviews", len(filt_kw))
            k2.metric("Positive", f"{(filt_kw['Sentiment']=='Positive').sum()}")
            k3.metric("Negative", f"{(filt_kw['Sentiment']=='Negative').sum()}")
            k4.metric("Neutral",  f"{(filt_kw['Sentiment']=='Neutral').sum()}")
            st.dataframe(
                filt_kw[["App","review","Sentiment","complaint"]].head(30),
                use_container_width=True
            )


# ============================================================
# APP SUCCESS PREDICTION
# ============================================================

elif page == "App Success Prediction":

    st.markdown('<div class="hero-title" style="font-size:36px;color:#000000;-webkit-text-fill-color:#000000;">Will Your App Succeed?</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#374151;font-size:15px;margin-bottom:24px;">Fill in your app details below and our Random Forest model will predict whether your app is likely to exceed <strong>1 Million installs</strong> — and tell you exactly what to improve.</div>', unsafe_allow_html=True)

    # Model + encoders loaded from module-level globals (trained once at startup)
    acc    = accuracy_score(y_test, pred_test)
    report = classification_report(y_test, pred_test, output_dict=True)

    # ============================================================
    # SECTION 1 — INTERACTIVE PREDICTOR
    # ============================================================
    section_header("Predict Your App's Success")

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("**Tell us about your app:**")
        st.markdown("<br>", unsafe_allow_html=True)

        user_category = st.selectbox(
            "App Category",
            sorted(df["Category"].unique()),
            help="Which Play Store category will your app be listed under?"
        )
        user_type = st.radio(
            "App Type",
            ["Free", "Paid"],
            horizontal=True,
            help="Will your app be free or paid?"
        )
        user_price = 0.0
        if user_type == "Paid":
            user_price = st.slider(
                "Price (USD)", 0.99, 29.99, 2.99, step=0.50
            )

        user_rating = st.slider(
            "Expected Rating", 1.0, 5.0, 4.0, step=0.1,
            help="What rating do you realistically expect to achieve?"
        )
        user_size = st.slider(
            "App Size (MB)", 1.0, 100.0, 20.0, step=1.0,
            help="Approximate size of your app"
        )
        user_content = st.selectbox(
            "Content Rating",
            sorted(df["Content Rating"].unique()),
            help="Who is your target audience?"
        )

        predict_btn = st.button("Predict Now", use_container_width=True)

    with right_col:
        if not predict_btn:
            # Placeholder before prediction
            st.markdown("""
            <div style='background:#E0F7FA;border:1.5px solid #80DEEA;border-radius:16px;
                        padding:32px;text-align:center;margin-top:32px;'>
                <div style='font-size:48px;margin-bottom:12px;'></div>
                <div style='font-family:Merriweather,serif;font-size:18px;
                            font-weight:700;color:#006064;margin-bottom:8px;'>
                    Ready to check your app?
                </div>
                <div style='font-size:14px;color:#374151;line-height:1.7;'>
                    Fill in your app details on the left and click
                    <strong>Predict Now</strong> to get your success score,
                    confidence level, and personalised improvement tips.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ---- Encode user inputs safely ----
            def safe_encode(encoder, value):
                """Return encoded integer; falls back to median class if unseen."""
                classes = list(encoder.classes_)
                if value in classes:
                    return encoder.transform([value])[0]
                return len(classes) // 2   # safe fallback

            cat_enc  = safe_encode(le_cat,  user_category)
            cr_enc   = safe_encode(le_cr,   user_content)
            type_enc = safe_encode(le_type, user_type)

            FEATURE_COLS = ["Category_enc","Rating","Size","Price","ContentRating_enc","Type_enc"]
            user_input = pd.DataFrame([[
                cat_enc,
                float(user_rating),
                float(user_size),
                float(user_price),
                cr_enc,
                type_enc,
            ]], columns=FEATURE_COLS)

            prob_success = float(model.predict_proba(user_input)[0][1])
            prediction   = int(prob_success >= 0.5)

            # ---- Result card ----
            if prediction == 1:
                card_bg    = "#E8F5E9"
                card_border= "#A5D6A7"
                icon       = ""
                verdict    = "HIGH SUCCESS POTENTIAL"
                verdict_color = "#1B5E20"
                msg = "Your app has a strong chance of exceeding 1 million installs based on the features you provided."
            else:
                card_bg    = "#FBE9E7"
                card_border= "#FFAB91"
                icon       = ""
                verdict    = "NEEDS IMPROVEMENT"
                verdict_color = "#BF360C"
                msg = "Based on current inputs, your app may struggle to cross 1 million installs. See tips below."

            # Confidence bar fill
            bar_pct  = int(prob_success * 100)
            bar_color= "#43A047" if prediction == 1 else "#E53935"

            st.markdown(f"""
            <div style='background:{card_bg};border:2px solid {card_border};
                        border-radius:16px;padding:28px;margin-bottom:20px;'>
                <div style='font-size:36px;margin-bottom:6px;'>{icon}</div>
                <div style='font-family:Merriweather,serif;font-size:20px;
                            font-weight:900;color:{verdict_color};margin-bottom:8px;'>
                    {verdict}
                </div>
                <div style='font-size:14px;color:#374151;line-height:1.6;margin-bottom:18px;'>
                    {msg}
                </div>
                <div style='font-size:12px;font-weight:700;color:#374151;
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>
                    Success Probability
                </div>
                <div style='background:#e5e7eb;border-radius:99px;height:18px;overflow:hidden;'>
                    <div style='width:{bar_pct}%;background:{bar_color};height:100%;
                                border-radius:99px;transition:width 0.5s ease;'></div>
                </div>
                <div style='font-size:26px;font-weight:900;color:{verdict_color};
                            margin-top:8px;'>{bar_pct}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ---- Personalised Tips ----
            st.markdown("** Personalised Improvement Tips**")

            tips = []
            importance = pd.Series(
                model.feature_importances_,
                index=["Category","Rating","Size","Price","Content Rating","Type"]
            ).sort_values(ascending=False)

            if user_rating < 4.0:
                tips.append(("Improve Your Rating",
                    f"Your target rating of **{user_rating}** is below 4.0. "
                    "Apps rated 4.0+ are significantly more likely to succeed. "
                    "Focus on stability, UX polish and responding to reviews."))

            if user_type == "Paid" and user_price > 4.99:
                tips.append(("Reconsider Pricing",
                    f"At **${user_price}**, your app is in a competitive price bracket. "
                    "Consider launching as Free with in-app purchases, or "
                    "reducing price to under $2.99 to lower the barrier to install."))

            # Category-specific tip
            cat_avg_installs = df[df["Category"] == user_category]["Installs"].mean()
            overall_avg      = df["Installs"].mean()
            if cat_avg_installs < overall_avg * 0.6:
                tips.append(("Category Insight",
                    f"The **{user_category.replace('_',' ').title()}** category has "
                    f"lower average installs ({cat_avg_installs/1e6:.1f}M) than the "
                    f"platform average ({overall_avg/1e6:.1f}M). "
                    "Consider whether a neighbouring category might give better visibility."))

            if user_size > 60:
                tips.append(("Reduce App Size",
                    f"At **{user_size:.0f} MB**, your app may deter users on limited data plans. "
                    "Apps under 30 MB have higher conversion rates — consider lazy loading assets."))

            if not tips:
                tips.append(("Looking Good!",
                    "Your inputs are well-optimised. Focus on marketing, ASO "
                    "(App Store Optimisation) and maintaining your rating post-launch."))

            for title, body in tips:
                with st.expander(title, expanded=True):
                    st.markdown(body)

    st.markdown("---")

    # ============================================================
    # SECTION 2 — MODEL DETAILS (collapsed by default)
    # ============================================================
    with st.expander("View Model Performance Details", expanded=False):

        section_header("Model Performance")

        kpi_row([
            ("Accuracy",   f"{acc*100:.1f}%",                          "overall"),
            ("Precision",  f"{report['1']['precision']*100:.1f}%",     "success class"),
            ("Recall",     f"{report['1']['recall']*100:.1f}%",        "success class"),
            ("F1 Score",   f"{report['1']['f1-score']*100:.1f}%",      "success class"),
            ("Train Size", f"{len(X_test)*4:,}",                       "apps trained on"),
            ("Test Size",  f"{len(X_test):,}",                         "apps evaluated"),
        ])

        col1, col2 = st.columns(2)

        with col1:
            importance_s = pd.Series(
                model.feature_importances_,
                index=["Category","Rating","Size","Price","Content Rating","Type"]
            ).sort_values()
            fig_fi = px.bar(
                x=importance_s.values, y=importance_s.index,
                orientation="h",
                color=importance_s.values,
                color_continuous_scale="Teal",
                title="Feature Importance — what drives success",
                labels={"x": "Importance", "y": "Feature"},
            )
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=360,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#E0F7FA"),
                yaxis=dict(gridcolor="#E0F7FA"),
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        with col2:
            cm = confusion_matrix(y_test, pred_test)
            fig_cm = px.imshow(
                cm, text_auto=True,
                color_continuous_scale="Teal",
                labels=dict(x="Predicted", y="Actual"),
                x=["Not Success", "Success"],
                y=["Not Success", "Success"],
                title="Confusion Matrix",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#000000", height=360,
                coloraxis_showscale=False,
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        section_header("Per-Class Classification Report")
        report_df = pd.DataFrame({
            "Class":     ["Not Success (< 1M installs)", "Success (≥ 1M installs)"],
            "Precision": [f"{report['0']['precision']:.3f}", f"{report['1']['precision']:.3f}"],
            "Recall":    [f"{report['0']['recall']:.3f}",    f"{report['1']['recall']:.3f}"],
            "F1 Score":  [f"{report['0']['f1-score']:.3f}",  f"{report['1']['f1-score']:.3f}"],
            "Support":   [int(report['0']['support']),        int(report['1']['support'])],
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        section_header("Prediction Confidence Distribution")
        fig_prob = px.histogram(
            x=proba_test, nbins=40,
            color_discrete_sequence=["#00ACC1"],
            title="Predicted Probability of Success — test set",
            labels={"x": "P(Success)", "y": "Count"},
        )
        fig_prob.add_vline(
            x=0.5, line_dash="dash", line_color="#E53935",
            annotation_text="Decision boundary (0.5)",
            annotation_font_color="#E53935"
        )
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#000000", height=320,
            bargap=0.04,
            xaxis=dict(gridcolor="#E0F7FA"),
            yaxis=dict(gridcolor="#E0F7FA"),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_prob, use_container_width=True)