import os
import tempfile
from pathlib import Path

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# 1. Page Configuration
# =========================================================
st.set_page_config(
    page_title="UAGC Immunochemotherapy Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. File Paths & Constants
# =========================================================
APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "SVM_rbf.pkl"
X_TRAIN_PATH = APP_DIR / "x_train.csv"
Y_TRAIN_PATH = APP_DIR / "y_train.csv"

FIXED_THRESHOLD = 0.4629352474478095

MODEL_ALIAS = "VE-EF"
TARGET_MODEL_NAME = "SVM_rbf"

APP_TITLE = "Prediction of Response to Immunotherapy Combined With Chemotherapy in Unresectable Advanced Gastric Cancer"
APP_SUBTITLE = "Multimodal venous-phase CT and elasticity radiomics model for individual treatment response estimation"

POSITIVE_LABEL_NAME = "Responder (CR/PR)"
NEGATIVE_LABEL_NAME = "Non-responder (SD/PD)"

SHAP_NSAMPLES = 160
BACKGROUND_N = 30

# =========================================================
# 3. CSS Styling
# =========================================================
st.markdown("""
<style>
    .main { background: linear-gradient(180deg, #f7f9fc 0%, #f4f7fb 100%); }
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }
    h1, h2, h3 { color: #1f2d3d; font-family: "Times New Roman", serif; letter-spacing: 0.2px; }
    .hero-card { background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); border: 1px solid #e6edf6; border-radius: 22px; padding: 1.35rem 1.6rem; box-shadow: 0 8px 24px rgba(31,45,61,0.08); margin-bottom: 1rem; }
    .note-card { background: #f4f8fc; border-left: 5px solid #4a90e2; border-radius: 12px; padding: 0.95rem 1rem; color: #30465d; font-size: 0.96rem; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); border: 1px solid #dfe8f4; border-radius: 18px; padding: 1rem 1.15rem; box-shadow: 0 4px 14px rgba(31,45,61,0.06); text-align: center; min-height: 132px; }
    .metric-title { font-size: 0.92rem; color: #607387; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.75rem; font-weight: 700; color: #163a63; line-height: 1.2; }
    .metric-sub { font-size: 0.86rem; color: #7d8c9c; margin-top: 0.3rem; }
    .result-positive { background: linear-gradient(135deg, #eef9f1 0%, #ffffff 100%); border: 1px solid #b9e5c4; border-radius: 14px; padding: 1rem 1.2rem; color: #1f5d2e; font-weight: 600; margin-top: 0.75rem; }
    .result-negative { background: linear-gradient(135deg, #fff4f4 0%, #ffffff 100%); border: 1px solid #f0c3c3; border-radius: 14px; padding: 1rem 1.2rem; color: #8a2d2d; font-weight: 600; margin-top: 0.75rem; }
    .section-card { background: white; border: 1px solid #e7edf5; border-radius: 18px; padding: 1rem 1rem 0.6rem 1rem; box-shadow: 0 4px 14px rgba(31,45,61,0.05); margin-bottom: 1rem; }
    .footer-note { font-size: 0.82rem; color: #708090; margin-top: 1rem; }
    .small-muted { color: #6d7d8d; font-size: 0.88rem; }
    div[data-testid="stExpander"] { border-radius: 14px !important; border: 1px solid #e6edf5 !important; background: #ffffff !important; }
    div[data-testid="stForm"] { background: white; border: 1px solid #e7edf5; border-radius: 18px; padding: 1rem 1rem 0.2rem 1rem; box-shadow: 0 4px 14px rgba(31,45,61,0.05); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. Utility Functions
# =========================================================
def check_required_files():
    required = [MODEL_PATH, X_TRAIN_PATH, Y_TRAIN_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("Missing required files in repository root:")
        for name in missing:
            st.write(f"- {name}")
        st.stop()

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["ID", "Id", "id", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

def infer_feature_group(feature_name: str) -> str:
    if "Elasticity" in feature_name:
        return "Elasticity Features"
    elif "Venous" in feature_name:
        return "Venous-phase CT Features"
    else:
        return "Other Features"

def format_widget_label(name: str, max_len: int = 150) -> str:
    # No abbreviations. Only truncate if absurdly long to protect UI layout.
    if len(name) > max_len:
        return name[:max_len - 1] + "…"
    return name

def format_force_label(name: str, max_len: int = 150) -> str:
    # No abbreviations for the force plot either.
    if len(name) > max_len:
        return name[:max_len - 1] + "…"
    return name

def safe_stats(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"min": -1.0, "max": 1.0, "median": 0.0, "q1": -0.5, "q3": 0.5}
    return {
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "median": float(np.median(s)),
        "q1": float(np.percentile(s, 25)),
        "q3": float(np.percentile(s, 75))
    }

@st.cache_resource(show_spinner=True)
def load_assets():
    model = joblib.load(MODEL_PATH)
    x_train = pd.read_csv(X_TRAIN_PATH)
    x_train = clean_columns(x_train)

    y_train = pd.read_csv(Y_TRAIN_PATH)
    if "label" not in y_train.columns:
        raise ValueError("y_train.csv missing column 'label'")
    y_train = y_train["label"].astype(int).values

    feature_names = list(x_train.columns)

    feature_meta = {}
    for col in feature_names:
        stats = safe_stats(x_train[col])
        feature_meta[col] = {
            **stats,
            "group": infer_feature_group(col)
        }

    background = x_train.sample(min(BACKGROUND_N, len(x_train)), random_state=42)
    return model, x_train, y_train, feature_names, feature_meta, background

def predict_positive_proba(model, X_df: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X_df)
    return proba[:, 1]

@st.cache_resource(show_spinner=False)
def build_explainer(_model, background_df):
    def f(data):
        data_df = pd.DataFrame(data, columns=background_df.columns)
        return predict_positive_proba(_model, data_df)
    return shap.KernelExplainer(f, background_df.values)

def shap_for_single_case(explainer, input_df, nsamples=160):
    shap_values = explainer.shap_values(input_df.values, nsamples=nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_value = float(np.array(explainer.expected_value).reshape(-1)[0]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

    explanation = shap.Explanation(
        values=np.array(shap_values).reshape(-1),
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )
    return explanation

def subset_explanation(explanation, top_n=None, for_force=False):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    data = np.array(explanation.data)

    if top_n is None:
        top_n = len(values)
        
    order = np.argsort(np.abs(values))[::-1][:top_n]

    if for_force:
        display_names = [format_force_label(n) for n in names[order]]
    else:
        display_names = [format_widget_label(n) for n in names[order]]

    return shap.Explanation(
        values=values[order],
        base_values=float(explanation.base_values),
        data=data[order],
        feature_names=display_names
    )

def build_shap_table(explanation):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    data = np.array(explanation.data)

    order = np.argsort(np.abs(values))[::-1]

    df = pd.DataFrame({
        "Feature Name": names[order],
        "Input Value": data[order],
        "SHAP Value": values[order],
        "Absolute SHAP": np.abs(values[order]),
        "Direction": np.where(values[order] >= 0, "Increase response probability", "Decrease response probability")
    }).sort_values("Absolute SHAP", ascending=False)

    return df

def render_force_plot_html(explanation, height=280):
    feature_values = pd.Series(explanation.data, index=explanation.feature_names)
    force_obj = shap.force_plot(
        base_value=float(explanation.base_values),
        shap_values=np.array(explanation.values),
        features=feature_values,
        matplotlib=False
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        shap.save_html(tmp.name, force_obj)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Enable horizontal scrolling for the force plot since names are long
    html = html.replace(
        "<body>",
        '<body style="margin:0; background:#ffffff; overflow-x:auto; zoom:0.82;">'
    )
    components.html(html, height=height, scrolling=True)

def plot_waterfall(explanation, total_features):
    plt.close("all")
    old_rc = plt.rcParams.copy()
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white"
    })

    # Increased width to 16 to accommodate long feature names
    fig_height = max(7.2, total_features * 0.35)
    fig = plt.figure(figsize=(16, fig_height), dpi=300)
    
    shap.plots.waterfall(explanation, max_display=total_features, show=False)
    fig = plt.gcf()
    
    # Dramatically increased left margin to 0.55 to push the plot right and leave space for names
    fig.subplots_adjust(left=0.55, right=0.97, top=0.98, bottom=0.05)

    plt.rcParams.update(old_rc)
    return fig

def plot_top_contrib_bar(explanation, total_features):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    order = np.argsort(np.abs(values))[::-1]

    vals = values[order]
    labels = names[order]

    # Increased width to 15 to accommodate long feature names
    fig_height = max(6.2, total_features * 0.25)
    fig, ax = plt.subplots(figsize=(15, fig_height), dpi=300)
    
    colors = ["#d9534f" if v > 0 else "#3b82f6" for v in vals]
    ax.barh(range(len(vals)), vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("SHAP value", fontsize=11, fontweight="bold")
    ax.set_title("All Feature Contributions", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # tight_layout automatically calculates boundaries so long text stays visible
    plt.tight_layout()
    return fig

def plot_probability_bar(prob_pos):
    prob_neg = 1 - prob_pos
    labels = [NEGATIVE_LABEL_NAME, POSITIVE_LABEL_NAME]
    probs = [prob_neg, prob_pos]
    colors = ["#b9c6d6", "#2f7ed8"]

    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=300)
    bars = ax.barh(labels, probs, color=colors, edgecolor="none")

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", fontsize=11, fontweight="bold")
    ax.set_title("Predicted Class Probabilities", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, probs):
        x = min(val + 0.015, 0.96)
        ax.text(x, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def make_input_widgets(feature_meta, feature_names):
    group_order = [
        "Elasticity Features",
        "Venous-phase CT Features",
        "Other Features"
    ]

    grouped = {g: [] for g in group_order}
    for feat in feature_names:
        g = feature_meta[feat]["group"]
        if g in grouped:
            grouped[g].append(feat)

    user_values = {}

    for group_name in group_order:
        feats = grouped.get(group_name, [])
        if not feats:
            continue

        with st.expander(f"{group_name} ({len(feats)} features)", expanded=False):
            cols = st.columns(3)
            for idx, feat in enumerate(feats):
                meta = feature_meta[feat]
                col = cols[idx % 3]

                min_v = float(meta["min"])
                max_v = float(meta["max"])
                default_v = float(meta["median"])

                if min_v == max_v:
                    min_v -= 1e-6
                    max_v += 1e-6

                help_text = (f"Full name: {feat}\nRange: {meta['min']:.4f} to {meta['max']:.4f}\nMedian: {meta['median']:.4f}")

                with col:
                    user_values[feat] = st.number_input(
                        label=format_widget_label(feat),
                        min_value=min_v, max_value=max_v, value=default_v,
                        help=help_text, format="%.6f", key=f"input_{feat}"
                    )

    return user_values

def make_interpretation_text(prob_pos, threshold):
    if prob_pos >= threshold:
        return f"The estimated probability of response is {prob_pos:.3f}, which is above the threshold ({threshold:.6f}). The model classifies this patient as a {POSITIVE_LABEL_NAME}."
    else:
        return f"The estimated probability of response is {prob_pos:.3f}, which is below the threshold ({threshold:.6f}). The model classifies this patient as a {NEGATIVE_LABEL_NAME}."

# =========================================================
# 5. Load Assets
# =========================================================
check_required_files()
model, x_train, y_train, feature_names, feature_meta, background = load_assets()
explainer = build_explainer(model, background)
TOTAL_FEATURES = len(feature_names)

# =========================================================
# 6. Hero Section
# =========================================================
st.markdown(f"""
<div class="hero-card">
    <h1 style="margin-bottom:0.35rem;">{APP_TITLE}</h1>
    <div style="font-size:1.02rem; color:#4f647a; line-height:1.5;">
        {APP_SUBTITLE}<br>
        <b>Deployed model:</b> {MODEL_ALIAS} &nbsp;|&nbsp;
        <b>Classifier:</b> {TARGET_MODEL_NAME} &nbsp;|&nbsp;
        <b>Total Features:</b> {TOTAL_FEATURES}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="note-card">The deployment uses a fixed decision threshold of <b>{FIXED_THRESHOLD:.6f}</b>.</div>', unsafe_allow_html=True)

# =========================================================
# 7. Sidebar
# =========================================================
with st.sidebar:
    st.header("Model Overview")
    st.write(f"**Model alias:** {MODEL_ALIAS}")
    st.write(f"**Classifier:** {TARGET_MODEL_NAME}")
    st.write(f"**Total Features:** {TOTAL_FEATURES} (Showing ALL)")
    st.write(f"**Decision threshold:** {FIXED_THRESHOLD:.6f}")
    st.markdown("---")
    st.caption("Research-use interface only. This tool does not replace clinical judgment.")

# =========================================================
# 8. Input Area
# =========================================================
st.subheader("Patient Feature Input")

with st.form("prediction_form", clear_on_submit=False):
    st.markdown('<div class="small-muted">Enter patient-specific radiomics feature values below. Features are grouped into Elasticity and Venous-phase CT.</div>', unsafe_allow_html=True)
    user_inputs = make_input_widgets(feature_meta, feature_names)

    c1, c2, c3 = st.columns([1.2, 1.2, 3.6])
    with c1:
        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)
    with c2:
        preview = st.form_submit_button("Preview Inputs", use_container_width=True)

input_df = pd.DataFrame([[user_inputs[f] for f in feature_names]], columns=feature_names)

if preview and not submitted:
    with st.expander("Current Input Table", expanded=True):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# =========================================================
# 9. Results Area
# =========================================================
if submitted:
    positive_proba = float(predict_positive_proba(model, input_df)[0])
    negative_proba = 1 - positive_proba
    predicted_class = 1 if positive_proba >= FIXED_THRESHOLD else 0
    predicted_label = POSITIVE_LABEL_NAME if predicted_class == 1 else NEGATIVE_LABEL_NAME

    st.subheader("Prediction Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Response Probability</div><div class="metric-value">{positive_proba * 100:.1f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Category</div><div class="metric-value" style="font-size:1.18rem;">{predicted_label}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Non-response Probability</div><div class="metric-value">{negative_proba * 100:.1f}%</div></div>', unsafe_allow_html=True)

    st.progress(int(round(positive_proba * 100)))

    interpretation_text = make_interpretation_text(positive_proba, FIXED_THRESHOLD)
    if predicted_class == 1:
        st.markdown(f'<div class="result-positive">{interpretation_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-negative">{interpretation_text}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Probability Visualization")
    prob_fig = plot_probability_bar(positive_proba)
    st.pyplot(prob_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Model Explanation (SHAP - All Features)")
    with st.spinner("Computing SHAP explanation for all features..."):
        explanation_full = shap_for_single_case(explainer, input_df, nsamples=SHAP_NSAMPLES)
        force_exp = subset_explanation(explanation_full, top_n=TOTAL_FEATURES, for_force=True)
        full_exp = subset_explanation(explanation_full, top_n=TOTAL_FEATURES, for_force=False)
        shap_df = build_shap_table(explanation_full)

    tab1, tab2, tab3 = st.tabs(["SHAP Force Plot (All)", "Waterfall / Contribution Plot (All)", "All Features Table"])

    with tab1:
        st.caption(f"Interactive force plot for all {TOTAL_FEATURES} features.")
        render_force_plot_html(force_exp, height=290)

    with tab2:
        st.caption(f"Waterfall plot and signed SHAP bar chart for all {TOTAL_FEATURES} features using original feature names.")
        fig1 = plot_waterfall(full_exp, total_features=TOTAL_FEATURES)
        st.pyplot(fig1, use_container_width=True)

        fig2 = plot_top_contrib_bar(full_exp, total_features=TOTAL_FEATURES)
        st.pyplot(fig2, use_container_width=True)

    with tab3:
        st.caption(f"All {TOTAL_FEATURES} features ranked by absolute SHAP magnitude.")
        display_df = shap_df.copy()
        for col in ["Input Value", "SHAP Value", "Absolute SHAP"]:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
        st.dataframe(display_df, use_container_width=True)

# =========================================================
# 10. Footer
# =========================================================
st.markdown("""
<div class="footer-note">
Research-use interface for the multimodal VE-EF model. 
</div>
""", unsafe_allow_html=True)
