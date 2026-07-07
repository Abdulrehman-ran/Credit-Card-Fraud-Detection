"""
╔══════════════════════════════════════════════════════════════╗
║      CREDIT CARD FRAUD DETECTION — STREAMLIT DASHBOARD      ║
║                                                              ║
║  A modern, interactive web application for fraud prediction  ║
║  with real-time model inference and performance analytics.   ║
║                                                              ║
║  Usage:                                                      ║
║    streamlit run app.py                                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_saved_model, load_results

# ──────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  Custom CSS — Dark Glassmorphism Theme
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0F0C29 0%, #1A1A2E 40%, #16213E 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0C29 0%, #1A1A2E 100%) !important;
        border-right: 1px solid rgba(108, 92, 231, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #A29BFE !important;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(108, 92, 231, 0.15);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C5CE7, #A29BFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
        font-weight: 500;
    }

    /* ── Prediction Result Cards ── */
    .prediction-safe {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.12), rgba(0, 212, 170, 0.04));
        border: 2px solid rgba(0, 212, 170, 0.4);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        animation: pulse-safe 2s ease-in-out infinite;
    }
    @keyframes pulse-safe {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 170, 0.1); }
        50% { box-shadow: 0 0 40px rgba(0, 212, 170, 0.25); }
    }
    .prediction-fraud {
        background: linear-gradient(135deg, rgba(255, 76, 106, 0.15), rgba(255, 76, 106, 0.05));
        border: 2px solid rgba(255, 76, 106, 0.5);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        animation: pulse-fraud 1.5s ease-in-out infinite;
    }
    @keyframes pulse-fraud {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 76, 106, 0.15); }
        50% { box-shadow: 0 0 50px rgba(255, 76, 106, 0.35); }
    }

    /* ── Hero Title ── */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C5CE7, #A29BFE, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #8892B0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #CCD6F6;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(108, 92, 231, 0.3);
    }

    /* ── Best Model Badge ── */
    .best-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6C5CE7, #A29BFE);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Tabs styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 8px 20px;
        border: 1px solid rgba(255,255,255,0.08);
        color: #8892B0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(108, 92, 231, 0.2) !important;
        border-color: #6C5CE7 !important;
        color: #A29BFE !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Load Model & Results (cached)
# ──────────────────────────────────────────────

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler from disk."""
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    
    model = load_saved_model(model_path)
    scaler = load_saved_model(scaler_path)
    return model, scaler


@st.cache_data
def load_training_results():
    """Load training results JSON."""
    results_path = os.path.join(PROJECT_ROOT, "models", "training_results.json")
    return load_results(results_path)


def check_model_exists():
    """Check if trained model files exist."""
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    results_path = os.path.join(PROJECT_ROOT, "models", "training_results.json")
    return all(os.path.exists(p) for p in [model_path, scaler_path, results_path])


# ──────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────

def create_gauge_chart(probability, title="Fraud Probability"):
    """Create a semicircular gauge chart for fraud probability."""
    # Determine color based on probability
    if probability < 0.3:
        color = "#00D4AA"  # Green — safe
    elif probability < 0.7:
        color = "#FDCB6E"  # Yellow — warning
    else:
        color = "#FF4C6A"  # Red — danger
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 16, "color": "#8892B0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#2D3561",
                     "tickfont": {"color": "#8892B0"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 2,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0, 212, 170, 0.08)"},
                {"range": [30, 70], "color": "rgba(253, 203, 110, 0.08)"},
                {"range": [70, 100], "color": "rgba(255, 76, 106, 0.08)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EAEAEA"},
    )
    return fig


def create_metrics_radar(metrics, model_name):
    """Create a radar chart for model metrics."""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(108, 92, 231, 0.15)",
        line=dict(color="#6C5CE7", width=2.5),
        marker=dict(size=8, color="#A29BFE"),
        name=model_name,
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#8892B0", size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#CCD6F6", size=12),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_sample_data(fraud=False):
    """
    Return sample feature values for demonstration.
    These are representative values from the actual dataset.
    """
    if fraud:
        # Typical fraud transaction pattern
        return {
            "Time": 80000, "V1": -3.04, "V2": 2.68, "V3": -5.15,
            "V4": 4.54, "V5": -2.11, "V6": -1.63, "V7": -4.12,
            "V8": 1.18, "V9": -3.58, "V10": -6.89, "V11": 3.42,
            "V12": -7.89, "V13": 0.35, "V14": -8.35, "V15": 0.23,
            "V16": -5.47, "V17": -7.86, "V18": -3.28, "V19": 1.12,
            "V20": 0.68, "V21": 0.98, "V22": -0.34, "V23": -0.12,
            "V24": -0.45, "V25": 0.23, "V26": 0.56, "V27": 1.23,
            "V28": 0.45, "Amount": 122.50,
        }
    else:
        # Typical legitimate transaction pattern
        return {
            "Time": 50000, "V1": -1.36, "V2": -0.07, "V3": 2.54,
            "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24,
            "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
            "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
            "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
            "V20": 0.25, "V21": -0.02, "V22": -0.54, "V23": -0.20,
            "V24": -0.07, "V25": 0.18, "V26": -0.19, "V27": 0.12,
            "V28": -0.02, "Amount": 49.90,
        }


# ──────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Fraud Detector")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Predict", "📊 Model Performance", "ℹ️ About"],
        index=0,
        label_visibility="collapsed",
    )
    
    st.markdown("---")


# ──────────────────────────────────────────────
#  Check if model is trained
# ──────────────────────────────────────────────

if not check_model_exists():
    st.markdown('<p class="hero-title">🛡️ Credit Card Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Machine Learning Powered Transaction Security</p>', unsafe_allow_html=True)
    
    st.warning(
        "⚠️ **Model not trained yet!**\n\n"
        "Please run the training pipeline first:\n\n"
        "```bash\n"
        "# 1. Place creditcard.csv in the dataset/ folder\n"
        "# 2. Install dependencies\n"
        "pip install -r requirements.txt\n\n"
        "# 3. Run training\n"
        "python train.py\n"
        "```\n\n"
        "After training completes, refresh this page."
    )
    st.stop()

# Load model and results
model, scaler = load_model_and_scaler()
results = load_training_results()


# ══════════════════════════════════════════════
#  PAGE: Home
# ══════════════════════════════════════════════

if page == "🏠 Home":
    # Hero section
    st.markdown('<p class="hero-title">🛡️ Credit Card Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Transaction Security using Machine Learning</p>', unsafe_allow_html=True)
    
    # Summary metrics
    summary = results.get("dataset_summary", {})
    best_metrics = results.get("best_metrics", {})
    best_name = results.get("best_model_name", "Unknown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('shape', [0])[0]:,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('fraud_percentage', 0):.3f}%</div>
            <div class="metric-label">Fraud Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_metrics.get('F1 Score', 0):.2%}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_metrics.get('ROC-AUC', 0):.2%}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Best model highlight
    st.markdown(f"""
    <div class="glass-card" style="text-align:center; margin-bottom:2rem;">
        <span class="best-badge">🏆 Best Model</span>
        <h2 style="color:#CCD6F6; margin:12px 0 4px 0;">{best_name}</h2>
        <p style="color:#8892B0;">Selected based on highest F1 Score — optimal for imbalanced datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Insights
    st.markdown('<p class="section-header">💡 Key Insights from EDA</p>', unsafe_allow_html=True)
    
    insights = results.get("insights", [])
    for insight in insights:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:10px; padding:14px 20px;">
            <span style="color:#CCD6F6; font-size:0.95rem;">{insight}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # EDA Images
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    eda_images = {
        "Class Distribution": "class_distribution.png",
        "Amount Distribution": "amount_distribution.png",
        "Time Distribution": "time_distribution.png",
        "Top Correlated Features": "top_features.png",
        "Correlation Heatmap": "correlation_heatmap.png",
    }
    
    # Show images in tabs
    eda_tabs = st.tabs(list(eda_images.keys()))
    for tab, (title, filename) in zip(eda_tabs, eda_images.items()):
        with tab:
            img_path = os.path.join(PROJECT_ROOT, "assets", filename)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info(f"📷 {title} plot will appear here after running `python train.py`")


# ══════════════════════════════════════════════
#  PAGE: Predict
# ══════════════════════════════════════════════

elif page == "🔍 Predict":
    st.markdown('<p class="hero-title">🔍 Transaction Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Enter transaction details to check for fraud</p>', unsafe_allow_html=True)
    
    # Quick-fill buttons
    st.markdown('<p class="section-header">⚡ Quick Fill (Demo Data)</p>', unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("✅ Sample Legitimate", use_container_width=True, type="secondary"):
            st.session_state["sample_data"] = get_sample_data(fraud=False)
            st.rerun()
    
    with col_btn2:
        if st.button("🚨 Sample Fraud", use_container_width=True, type="secondary"):
            st.session_state["sample_data"] = get_sample_data(fraud=True)
            st.rerun()
    
    # Get default values (from quick-fill or zeros)
    defaults = st.session_state.get("sample_data", {})
    
    # Input form
    st.markdown('<p class="section-header">📝 Transaction Features</p>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        # Time and Amount in first row
        col_t, col_a = st.columns(2)
        with col_t:
            time_val = st.number_input(
                "⏱️ Time (seconds)", 
                value=float(defaults.get("Time", 0.0)),
                step=1.0, format="%.1f",
                help="Seconds elapsed since first transaction in dataset"
            )
        with col_a:
            amount_val = st.number_input(
                "💰 Amount ($)", 
                value=float(defaults.get("Amount", 0.0)),
                step=0.01, format="%.2f", min_value=0.0,
                help="Transaction amount in USD"
            )
        
        # V1-V28 in grid (4 columns)
        st.markdown("**PCA-Transformed Features (V1–V28)**")
        
        v_values = {}
        cols_per_row = 4
        v_features = [f"V{i}" for i in range(1, 29)]
        
        for row_start in range(0, 28, cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, feature_idx in enumerate(range(row_start, min(row_start + cols_per_row, 28))):
                feature = v_features[feature_idx]
                with cols[col_idx]:
                    v_values[feature] = st.number_input(
                        feature,
                        value=float(defaults.get(feature, 0.0)),
                        step=0.01, format="%.4f",
                        label_visibility="visible",
                    )
        
        submitted = st.form_submit_button(
            "🔍 Analyze Transaction", 
            use_container_width=True,
            type="primary",
        )
    
    # Prediction
    if submitted:
        # Prepare input
        input_data = np.array([[time_val] + [v_values[f"V{i}"] for i in range(1, 29)] + [amount_val]])
        
        # Scale Time and Amount
        time_amount = np.array([[time_val, amount_val]])
        time_amount_scaled = scaler.transform(time_amount)
        input_data[0, 0] = time_amount_scaled[0, 0]   # Time scaled
        input_data[0, -1] = time_amount_scaled[0, 1]   # Amount scaled
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        fraud_prob = probability[1]
        legit_prob = probability[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Result display
        col_result, col_gauge = st.columns([1, 1])
        
        with col_result:
            if prediction == 0:
                st.markdown(f"""
                <div class="prediction-safe">
                    <h1 style="color:#00D4AA; margin:0; font-size:3rem;">✅</h1>
                    <h2 style="color:#00D4AA; margin:8px 0;">LEGITIMATE</h2>
                    <p style="color:#8892B0; font-size:1.1rem;">
                        This transaction appears to be <strong style="color:#00D4AA;">safe</strong>.
                    </p>
                    <p style="color:#CCD6F6; font-size:1.8rem; font-weight:700; margin-top:12px;">
                        {legit_prob:.1%} Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-fraud">
                    <h1 style="color:#FF4C6A; margin:0; font-size:3rem;">🚨</h1>
                    <h2 style="color:#FF4C6A; margin:8px 0;">FRAUDULENT</h2>
                    <p style="color:#8892B0; font-size:1.1rem;">
                        This transaction is flagged as <strong style="color:#FF4C6A;">potentially fraudulent</strong>.
                    </p>
                    <p style="color:#CCD6F6; font-size:1.8rem; font-weight:700; margin-top:12px;">
                        {fraud_prob:.1%} Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_gauge:
            st.plotly_chart(
                create_gauge_chart(fraud_prob, "Fraud Probability"),
                use_container_width=True,
            )
        
        # Probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">📊 Probability Breakdown</p>', unsafe_allow_html=True)
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Legitimate", "Fraudulent"],
            y=[legit_prob * 100, fraud_prob * 100],
            marker_color=["#00D4AA", "#FF4C6A"],
            text=[f"{legit_prob:.2%}", f"{fraud_prob:.2%}"],
            textposition="auto",
            textfont=dict(size=16, color="white"),
        ))
        fig_bar.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.05)",
                       color="#8892B0"),
            xaxis=dict(color="#CCD6F6"),
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE: Model Performance
# ══════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.markdown('<p class="hero-title">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Compare ML models trained on the fraud detection dataset</p>', unsafe_allow_html=True)
    
    all_results = results.get("all_results", {})
    best_name = results.get("best_model_name", "")
    
    # Model comparison table
    st.markdown('<p class="section-header">📋 Performance Comparison</p>', unsafe_allow_html=True)
    
    if all_results:
        # Build DataFrame
        df_results = pd.DataFrame(all_results).T
        df_results.index.name = "Model"
        
        # Highlight best values
        styled_df = df_results.style.format("{:.4f}").highlight_max(
            axis=0, color="rgba(108, 92, 231, 0.3)"
        )
        st.dataframe(styled_df, use_container_width=True, height=210)
    
    # Best model card
    best_metrics = results.get("best_metrics", {})
    st.markdown(f"""
    <div class="glass-card" style="text-align:center; margin:1.5rem 0;">
        <span class="best-badge">🏆 Best Model</span>
        <h2 style="color:#CCD6F6; margin:10px 0 6px 0;">{best_name}</h2>
        <p style="color:#8892B0;">
            F1 Score: <strong style="color:#6C5CE7;">{best_metrics.get('F1 Score', 0):.4f}</strong> · 
            ROC-AUC: <strong style="color:#6C5CE7;">{best_metrics.get('ROC-AUC', 0):.4f}</strong> · 
            Accuracy: <strong style="color:#6C5CE7;">{best_metrics.get('Accuracy', 0):.4f}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Radar chart + bar chart
    col_radar, col_bar = st.columns(2)
    
    with col_radar:
        st.markdown('<p class="section-header">🎯 Best Model Radar</p>', unsafe_allow_html=True)
        if best_metrics:
            st.plotly_chart(
                create_metrics_radar(best_metrics, best_name),
                use_container_width=True,
            )
    
    with col_bar:
        st.markdown('<p class="section-header">📊 Model Comparison</p>', unsafe_allow_html=True)
        if all_results:
            fig = go.Figure()
            
            model_colors = {
                "Logistic Regression": "#6C5CE7",
                "Decision Tree": "#00D4AA",
                "Random Forest": "#FDCB6E",
                "XGBoost": "#FF4C6A",
            }
            
            for model_name, metrics in all_results.items():
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=model_colors.get(model_name, "#6C5CE7"),
                    opacity=0.85,
                ))
            
            fig.update_layout(
                barmode="group",
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#CCD6F6")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#8892B0",
                           title="Score"),
                xaxis=dict(color="#CCD6F6"),
                margin=dict(l=40, r=20, t=20, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices and ROC Curves
    st.markdown('<p class="section-header">🔬 Detailed Visualizations</p>', unsafe_allow_html=True)
    
    tab_cm, tab_roc = st.tabs(["Confusion Matrices", "ROC Curves"])
    
    with tab_cm:
        img_cm = os.path.join(PROJECT_ROOT, "assets", "confusion_matrices.png")
        if os.path.exists(img_cm):
            st.image(img_cm, use_container_width=True)
        else:
            st.info("Confusion matrix plots will appear after running `python train.py`")
    
    with tab_roc:
        img_roc = os.path.join(PROJECT_ROOT, "assets", "roc_curves.png")
        if os.path.exists(img_roc):
            st.image(img_roc, use_container_width=True)
        else:
            st.info("ROC curve plots will appear after running `python train.py`")


# ══════════════════════════════════════════════
#  PAGE: About
# ══════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown('<p class="hero-title">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Credit Card Fraud Detection using Machine Learning</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="margin-bottom:1.5rem;">
        <h3 style="color:#A29BFE; margin-top:0;">🎯 Project Overview</h3>
        <p style="color:#CCD6F6;">
            This project implements a complete machine learning pipeline for detecting 
            fraudulent credit card transactions. It uses the 
            <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank" 
               style="color:#6C5CE7;">Kaggle Credit Card Fraud Detection</a> dataset 
            containing 284,807 European cardholder transactions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00D4AA; margin-top:0;">🛠️ Tech Stack</h3>
            <ul style="color:#CCD6F6; line-height:2;">
                <li><strong>Python 3.9+</strong> — Core language</li>
                <li><strong>Pandas & NumPy</strong> — Data manipulation</li>
                <li><strong>Scikit-learn</strong> — ML algorithms</li>
                <li><strong>XGBoost</strong> — Gradient boosting</li>
                <li><strong>Imbalanced-learn</strong> — SMOTE</li>
                <li><strong>Matplotlib & Seaborn</strong> — Static plots</li>
                <li><strong>Plotly</strong> — Interactive charts</li>
                <li><strong>Streamlit</strong> — Web dashboard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#FDCB6E; margin-top:0;">📋 ML Pipeline</h3>
            <ul style="color:#CCD6F6; line-height:2;">
                <li><strong>Data Preprocessing</strong> — Cleaning & scaling</li>
                <li><strong>SMOTE</strong> — Class imbalance handling</li>
                <li><strong>Logistic Regression</strong> — Baseline model</li>
                <li><strong>Decision Tree</strong> — Interpretable model</li>
                <li><strong>Random Forest</strong> — Ensemble method</li>
                <li><strong>XGBoost</strong> — Advanced boosting</li>
                <li><strong>F1 Score</strong> — Primary metric</li>
                <li><strong>ROC-AUC</strong> — Ranking metric</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="margin-top:1.5rem;">
        <h3 style="color:#FF4C6A; margin-top:0;">📄 Dataset Information</h3>
        <p style="color:#CCD6F6;">
            The dataset contains transactions made by European cardholders in September 2013.
            Due to confidentiality, the original features are PCA-transformed into V1–V28.
            Only <strong>Time</strong> and <strong>Amount</strong> retain their original meaning.
            The target variable <strong>Class</strong> is binary: 0 (legitimate) and 1 (fraud).
        </p>
        <p style="color:#8892B0; font-size:0.85rem; margin-top:12px;">
            📌 <em>This project was developed as a semester project for a university Data Science course.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
