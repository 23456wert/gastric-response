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
# 1. 页面与全局配置
# =========================================================
st.set_page_config(
    page_title="UAGC Immunochemotherapy Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent

# 所有文件均放在根目录
MODEL_PATH = APP_DIR / "SVM_rbf.pkl"
X_TRAIN_PATH = APP_DIR / "x_train.csv"
Y_TRAIN_PATH = APP_DIR / "y_train.csv"

# 固定阈值
FIXED_THRESHOLD = 0.4629352474478095

# 模型信息
MODEL_ALIAS = "VE-EF"
TARGET_MODEL_NAME = "SVM_rbf"

# 标题信息
APP_TITLE = "Prediction of Response to Immunotherapy Combined With Chemotherapy in Unresectable Advanced Gastric Cancer"
APP_SUBTITLE = "A publication-style deployment interface for multimodal venous-phase CT and elasticity radiomics"
APP_SHORT = "UAGC Immunochemotherapy Response Predictor"

# 类别命名
POSITIVE_LABEL_NAME = "Responder (CR/PR)"
NEGATIVE_LABEL_NAME = "Non-responder (SD/PD)"

# SHAP 参数
FORCE_TOP_N = 8
FULL_TOP_N = 12
SHAP_NSAMPLES = 160
BACKGROUND_N = 30

# 页脚
FOOTER_TEXT = (
    "Research-use interface for a multimodal radiomics model. "
    "This application is intended for scientific demonstration and does not replace clinical judgment."
)


# =========================================================
# 2. 全局样式（极简顶刊风格）
# =========================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #f4f7fb 100%);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 0.85rem;
        padding-bottom: 1.8rem;
    }

    h1, h2, h3 {
        color: #1f2d3d;
        font-family: "Times New Roman", serif;
        letter-spacing: 0.15px;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fbfe 100%);
        border: 1px solid #e7edf5;
        border-radius: 24px;
        padding: 1.25rem 1.45rem;
        box-shadow: 0 6px 22px rgba(31,45,61,0.06);
        margin-bottom: 0.9rem;
    }

    .hero-badge {
        display: inline-block;
        background: #f2f6fb;
        color: #3e5873;
        border: 1px solid #e3ebf4;
        border-radius: 999px;
        padding: 0.22rem 0.68rem;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
    }

    .note-card {
        background: #f7fafc;
        border: 1px solid #e7edf5;
        border-left: 4px solid #7ea6d8;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        color: #33475b;
        font-size: 0.94rem;
        margin-bottom: 1rem;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e8eef5;
        border-radius: 18px;
        padding: 0.95rem 1rem 0.75rem 1rem;
        box-shadow: 0 4px 12px rgba(31,45,61,0.04);
        margin-bottom: 0.9rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e4ebf3;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 4px 10px rgba(31,45,61,0.04);
        text-align: center;
        min-height: 126px;
    }

    .metric-title {
        font-size: 0.9rem;
        color: #687b8e;
        margin-bottom: 0.18rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #163a63;
        line-height: 1.2;
    }

    .metric-sub {
        font-size: 0.84rem;
        color: #8492a1;
        margin-top: 0.28rem;
    }

    .result-positive {
        background: linear-gradient(135deg, #f1faf3 0%, #ffffff 100%);
        border: 1px solid #c7e8cf;
        border-radius: 14px;
        padding: 0.95rem 1.1rem;
        color: #215f31;
        font-weight: 600;
        margin-top: 0.75rem;
    }

    .result-negative {
        background: linear-gradient(135deg, #fff6f6 0%, #ffffff 100%);
        border: 1px solid #efcaca;
        border-radius: 14px;
        padding: 0.95rem 1.1rem;
        color: #8c3131;
        font-weight: 600;
        margin-top: 0.75rem;
    }

    .caption-muted {
        color: #728191;
        font-size: 0.87rem;
        line-height: 1.5;
    }

    .footer-note {
        font-size: 0.8rem;
        color: #7b8794;
        margin-top: 1rem;
    }

    .mini-tag {
        display: inline-block;
        background: #f5f8fb;
        border: 1px solid #e6edf5;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        margin: 0.08rem 0.18rem 0.08rem 0;
        font-size: 0.76rem;
        color: #526578;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid #e8eef5 !important;
        background: #ffffff !important;
        box-shadow: none !important;
    }

    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px !important;
        padding-left: 14px !important;
        padding-right: 14px !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. 工具函数
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
    if feature_name.startswith("Venousperi_"):
        return "Venous Peritumoral"
    elif feature_name.startswith("Elasticityperi_"):
        return "Elasticity Peritumoral"
    elif feature_name.startswith("Venous_"):
        return "Venous Intratumoral"
    elif feature_name.startswith("Elasticity_"):
        return "Elasticity Intratumoral"
    else:
        return "Other Features"


def safe_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "min": -1.0,
            "max": 1.0,
            "median": 0.0,
            "q1": -0.5,
            "q3": 0.5
        }
    return {
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "median": float(np.median(s)),
        "q1": float(np.percentile(s, 25)),
        "q3": float(np.percentile(s, 75))
    }


def format_widget_label(name: str, max_len: int = 52) -> str:
    s = name
    s = s.replace("Venousperi_", "V-PERI · ")
    s = s.replace("Elasticityperi_", "E-PERI · ")
    s = s.replace("Venous_", "V-INTRA · ")
    s = s.replace("Elasticity_", "E-INTRA · ")
    s = s.replace("ResNet50Feat", "RN50")
    s = s.replace("ResNet50_Feat", "RN50")
    s = s.replace("ResNet50-max", "RN50")
    s = s.replace("firstorder", "1stOrder")
    s = s.replace("wavelet", "Wavelet")
    s = s.replace("log_sigma", "LoG")
    s = s.replace("glszm", "GLSZM")
    s = s.replace("glcm", "GLCM")
    s = s.replace("gldm", "GLDM")
    s = s.replace("ngtdm", "NGTDM")
    s = s.replace("shape", "Shape")
    s = s.replace("original", "Orig")
    s = s.replace("_", " · ")
    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s


def format_force_label(name: str, max_len: int = 34) -> str:
    s = name
    s = s.replace("Venousperi_", "V-P ")
    s = s.replace("Elasticityperi_", "E-P ")
    s = s.replace("Venous_", "V ")
    s = s.replace("Elasticity_", "E ")
    s = s.replace("ResNet50Feat", "RN50")
    s = s.replace("ResNet50_Feat", "RN50")
    s = s.replace("firstorder", "1st")
    s = s.replace("wavelet", "Wav")
    s = s.replace("log_sigma", "LoG")
    s = s.replace("glszm", "GLSZM")
    s = s.replace("glcm", "GLCM")
    s = s.replace("gldm", "GLDM")
    s = s.replace("ngtdm", "NGTDM")
    s = s.replace("original", "Orig")
    s = s.replace("3D", "")
    s = s.replace("_", " · ")
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s


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

    feature_group_df = pd.DataFrame({
        "Feature": feature_names,
        "Group": [infer_feature_group(f) for f in feature_names]
    })

    return model, x_train, y_train, feature_names, feature_meta, background, feature_group_df


def predict_positive_proba(model, X_df: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not support predict_proba().")
    proba = model.predict_proba(X_df)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Invalid predict_proba output format.")
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

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).reshape(-1)[0])
    else:
        base_value = float(base_value)

    return shap.Explanation(
        values=np.array(shap_values).reshape(-1),
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )


def subset_explanation(explanation, top_n=12, for_force=False):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    data = np.array(explanation.data)

    order = np.argsort(np.abs(values))[::-1][:top_n]
    top_values = values[order]
    top_names = names[order]
    top_data = data[order]

    if for_force:
        display_names = [format_force_label(n) for n in top_names]
    else:
        display_names = [format_widget_label(n, max_len=60) for n in top_names]

    return shap.Explanation(
        values=top_values,
        base_values=float(explanation.base_values),
        data=top_data,
        feature_names=display_names
    )


def build_shap_table(explanation, top_n=12) -> pd.DataFrame:
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    data = np.array(explanation.data)

    order = np.argsort(np.abs(values))[::-1][:top_n]

    df = pd.DataFrame({
        "Full Feature Name": names[order],
        "Display Name": [format_widget_label(n, max_len=70) for n in names[order]],
        "Input Value": data[order],
        "SHAP Value": values[order],
        "Absolute SHAP": np.abs(values[order]),
        "Direction": np.where(values[order] >= 0, "Increase response probability", "Decrease response probability")
    }).sort_values("Absolute SHAP", ascending=False)

    return df


def render_force_plot_html(explanation, height=290):
    feature_values = pd.Series(explanation.data, index=explanation.feature_names)

    force_obj = shap.force_plot(
        base_value=float(explanation.base_values),
        shap_values=np.array(explanation.values),
        features=feature_values,
        matplotlib=False
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        shap.save_html(tmp.name, force_obj)
        html_path = tmp.name

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace(
        "<body>",
        """
        <body style="margin:0; background:#ffffff; overflow-x:auto; zoom:0.82;">
        """
    )

    components.html(html, height=height, scrolling=True)


def plot_probability_bar(prob_pos: float):
    prob_neg = 1 - prob_pos
    labels = [NEGATIVE_LABEL_NAME, POSITIVE_LABEL_NAME]
    probs = [prob_neg, prob_pos]
    colors = ["#b8c6d8", "#2f7ed8"]

    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=300)
    bars = ax.barh(labels, probs, color=colors, edgecolor="none")

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", fontsize=11, fontweight="bold")
    ax.set_title("Predicted Class Probabilities", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, probs):
        x = min(val + 0.015, 0.96)
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_waterfall(explanation, max_display=12):
    plt.close("all")
    old_rc = plt.rcParams.copy()

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })

    fig = plt.figure(figsize=(11.5, 7.2), dpi=300)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.37, right=0.97, top=0.95, bottom=0.08)

    plt.rcParams.update(old_rc)
    return fig


def plot_top_contrib_bar(explanation, max_display=12):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    order = np.argsort(np.abs(values))[::-1][:max_display]

    vals = values[order]
    labels = names[order]
    colors = ["#d9534f" if v > 0 else "#3b82f6" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax.barh(range(len(vals)), vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("SHAP value", fontsize=11, fontweight="bold")
    ax.set_title("Top Feature Contributions", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_feature_group_distribution(feature_group_df: pd.DataFrame):
    count_df = (
        feature_group_df["Group"]
        .value_counts()
        .rename_axis("Feature Group")
        .reset_index(name="Count")
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    ax.barh(count_df["Feature Group"][::-1], count_df["Count"][::-1], color="#6c93c2", edgecolor="none")
    ax.set_xlabel("Number of Features", fontsize=11, fontweight="bold")
    ax.set_title("Feature Group Distribution", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(count_df["Count"][::-1]):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig, count_df


def make_interpretation_text(prob_pos: float, threshold: float) -> str:
    if prob_pos >= threshold:
        delta = prob_pos - threshold
        return (
            f"The estimated response probability is {prob_pos:.3f}, which exceeds the fixed decision threshold "
            f"({threshold:.6f}) by {delta:.3f}. This patient is therefore classified as a {POSITIVE_LABEL_NAME}."
        )
    else:
        delta = threshold - prob_pos
        return (
            f"The estimated response probability is {prob_pos:.3f}, which is below the fixed decision threshold "
            f"({threshold:.6f}) by {delta:.3f}. This patient is therefore classified as a {NEGATIVE_LABEL_NAME}."
        )


def make_input_widgets(feature_meta, feature_names, keyword=""):
    group_order = [
        "Venous Intratumoral",
        "Venous Peritumoral",
        "Elasticity Intratumoral",
        "Elasticity Peritumoral",
        "Other Features"
    ]

    grouped = {g: [] for g in group_order}
    for feat in feature_names:
        grouped.setdefault(feature_meta[feat]["group"], []).append(feat)

    keyword = (keyword or "").strip().lower()
    user_values = {}

    for group_name in group_order:
        feats = grouped.get(group_name, [])

        if keyword:
            feats = [f for f in feats if keyword in f.lower() or keyword in format_widget_label(f).lower()]

        if not feats:
            continue

        with st.expander(f"{group_name} ({len(feats)})", expanded=False):
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

                help_text = (
                    f"Full feature name: {feat}\n"
                    f"Training range: {meta['min']:.4f} to {meta['max']:.4f}\n"
                    f"Median: {meta['median']:.4f}\n"
                    f"IQR: {meta['q1']:.4f} - {meta['q3']:.4f}"
                )

                with col:
                    user_values[feat] = st.number_input(
                        label=format_widget_label(feat),
                        min_value=min_v,
                        max_value=max_v,
                        value=default_v,
                        help=help_text,
                        format="%.6f",
                        key=f"input_{feat}"
                    )

    # 未显示的特征自动回填为中位数，保证输入完整
    for feat in feature_names:
        if feat not in user_values:
            user_values[feat] = float(feature_meta[feat]["median"])

    return user_values


# =========================================================
# 4. 加载资源
# =========================================================
check_required_files()
model, x_train, y_train, feature_names, feature_meta, background, feature_group_df = load_assets()
explainer = build_explainer(model, background)

n_total = len(y_train)
n_pos = int(np.sum(y_train == 1))
n_neg = int(np.sum(y_train == 0))
pos_rate = n_pos / n_total if n_total > 0 else 0.0

group_counts = (
    feature_group_df["Group"]
    .value_counts()
    .rename_axis("Feature Group")
    .reset_index(name="Count")
)


# =========================================================
# 5. 顶部头图（更克制版本）
# =========================================================
st.markdown(f"""
<div class="hero-card">
    <div style="margin-bottom:0.32rem;">
        <span class="hero-badge">Clinical AI</span>
        <span class="hero-badge">Radiomics</span>
        <span class="hero-badge">Immunochemotherapy</span>
    </div>
    <h1 style="margin-bottom:0.28rem; font-size:2.0rem;">{APP_TITLE}</h1>
    <div style="font-size:0.98rem; color:#5a6d80; line-height:1.5;">
        {APP_SUBTITLE}
    </div>
    <div style="margin-top:0.55rem; font-size:0.9rem; color:#708192;">
        <b>Model:</b> {MODEL_ALIAS}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Classifier:</b> {TARGET_MODEL_NAME}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Input:</b> Venous-phase CT + Elasticity radiomics
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="note-card">
This application estimates treatment response to <b>immunotherapy combined with chemotherapy</b> in patients with
<b>unresectable advanced gastric cancer</b>. The deployment uses a fixed threshold of
<b>{FIXED_THRESHOLD:.6f}</b>. Positive prediction corresponds to <b>{POSITIVE_LABEL_NAME}</b>, and negative prediction
corresponds to <b>{NEGATIVE_LABEL_NAME}</b>.
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. 侧边栏（简化版）
# =========================================================
with st.sidebar:
    st.header("Overview")
    st.write(f"**App:** {APP_SHORT}")
    st.write(f"**Model:** {MODEL_ALIAS}")
    st.write(f"**Classifier:** {TARGET_MODEL_NAME}")
    st.write(f"**Features:** {len(feature_names)}")
    st.write(f"**Threshold:** {FIXED_THRESHOLD:.6f}")
    st.markdown("---")
    st.caption("Research-use deployment interface.")


# =========================================================
# 7. 页面主标签
# =========================================================
tab_pred, tab_model, tab_guide = st.tabs([
    "Prediction",
    "Model Card",
    "Interpretation Guide"
])


# =========================================================
# 8. Prediction 页
# =========================================================
with tab_pred:
    st.subheader("Patient-specific Prediction")

    st.markdown(
        """
        <div class="section-card">
            <div class="caption-muted">
                Enter patient-specific radiomics feature values to generate a single-case prediction and SHAP-based interpretation.
                To preserve a publication-style minimalist interface, the feature input area is collapsed by default.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1.38, 0.82], gap="large")

    with left:
        with st.form("prediction_form", clear_on_submit=False):
            with st.expander("Patient Feature Panel", expanded=False):
                st.markdown(
                    '<div class="caption-muted">Use the search box below to quickly locate a feature. All unspecified displayed values default to the training median.</div>',
                    unsafe_allow_html=True
                )

                feature_keyword = st.text_input(
                    "Feature keyword search",
                    value="",
                    placeholder="e.g., RN50, GLSZM, Venous, Elasticity, LoG"
                )

                user_inputs = make_input_widgets(
                    feature_meta=feature_meta,
                    feature_names=feature_names,
                    keyword=feature_keyword
                )

            c1, c2, c3 = st.columns([1.05, 1.05, 3.4])
            with c1:
                run_pred = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)
            with c2:
                preview_inputs = st.form_submit_button("Preview Inputs", use_container_width=True)

        input_df = pd.DataFrame([[user_inputs[f] for f in feature_names]], columns=feature_names)

        if preview_inputs and not run_pred:
            with st.expander("Current Input Table", expanded=True):
                st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Study Snapshot")
        st.markdown(f"""
        <div class="mini-tag">UAGC</div>
        <div class="mini-tag">Immunotherapy + Chemotherapy</div>
        <div class="mini-tag">{MODEL_ALIAS}</div>
        <div class="mini-tag">{TARGET_MODEL_NAME}</div>
        <div class="mini-tag">Threshold {FIXED_THRESHOLD:.6f}</div>
        <div class="mini-tag">{len(feature_names)} features</div>
        """, unsafe_allow_html=True)

        st.markdown("### Inference Rule")
        st.write(
            "The model produces a continuous response probability, which is converted to a binary prediction using the fixed deployment threshold."
        )

        st.markdown("### Display Logic")
        st.write(
            f"The force plot is restricted to the top {FORCE_TOP_N} features for visual clarity, while the waterfall plot and feature table preserve the top {FULL_TOP_N} features."
        )

        st.markdown("### Interface Style")
        st.write(
            "This release uses a compact publication-style layout with collapsed inputs and reduced visual clutter for cleaner presentation."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if run_pred:
        positive_proba = float(predict_positive_proba(model, input_df)[0])
        negative_proba = 1 - positive_proba
        predicted_class = 1 if positive_proba >= FIXED_THRESHOLD else 0
        predicted_label = POSITIVE_LABEL_NAME if predicted_class == 1 else NEGATIVE_LABEL_NAME

        st.subheader("Prediction Summary")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Response Probability</div>
                <div class="metric-value">{positive_proba * 100:.1f}%</div>
                <div class="metric-sub">Probability of {POSITIVE_LABEL_NAME}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Category</div>
                <div class="metric-value" style="font-size:1.16rem;">{predicted_label}</div>
                <div class="metric-sub">Threshold = {FIXED_THRESHOLD:.6f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Non-response Probability</div>
                <div class="metric-value">{negative_proba * 100:.1f}%</div>
                <div class="metric-sub">Probability of {NEGATIVE_LABEL_NAME}</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(int(round(positive_proba * 100)))

        interpretation_text = make_interpretation_text(positive_proba, FIXED_THRESHOLD)
        if predicted_class == 1:
            st.markdown(
                f'<div class="result-positive">{interpretation_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-negative">{interpretation_text}</div>',
                unsafe_allow_html=True
            )

        with st.expander("Submitted Feature Values", expanded=False):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Probability Visualization")
        prob_fig = plot_probability_bar(positive_proba)
        st.pyplot(prob_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Model Explanation (SHAP)")
        with st.spinner("Computing SHAP explanation..."):
            explanation_full = shap_for_single_case(explainer, input_df, nsamples=SHAP_NSAMPLES)
            force_exp = subset_explanation(explanation_full, top_n=FORCE_TOP_N, for_force=True)
            full_exp = subset_explanation(explanation_full, top_n=FULL_TOP_N, for_force=False)
            shap_df = build_shap_table(explanation_full, top_n=FULL_TOP_N)

        tab_shap1, tab_shap2, tab_shap3, tab_shap4 = st.tabs([
            f"Force Plot (Top {FORCE_TOP_N})",
            f"Waterfall Plot (Top {FULL_TOP_N})",
            "Contribution Bar Plot",
            "Feature Table"
        ])

        with tab_shap1:
            st.caption(
                f"Interactive force plot of the top {FORCE_TOP_N} features ranked by absolute SHAP value. "
                f"Feature names are aggressively abbreviated to reduce overlap."
            )
            render_force_plot_html(force_exp, height=290)
            st.info(
                "Force plots are visually sensitive to long radiomics feature names. "
                "This deployment therefore reserves the full Top-12 interpretation for the waterfall plot and feature table."
            )

        with tab_shap2:
            st.caption(f"Waterfall plot of the top {FULL_TOP_N} contributors for the current case.")
            fig1 = plot_waterfall(full_exp, max_display=FULL_TOP_N)
            st.pyplot(fig1, use_container_width=True)

        with tab_shap3:
            st.caption(f"Signed SHAP contributions for the top {FULL_TOP_N} features.")
            fig2 = plot_top_contrib_bar(full_exp, max_display=FULL_TOP_N)
            st.pyplot(fig2, use_container_width=True)

        with tab_shap4:
            st.caption(f"Top {FULL_TOP_N} features ranked by absolute SHAP magnitude.")
            show_df = shap_df.copy()
            show_df["Input Value"] = pd.to_numeric(show_df["Input Value"], errors="coerce")
            show_df["SHAP Value"] = pd.to_numeric(show_df["SHAP Value"], errors="coerce")
            show_df["Absolute SHAP"] = pd.to_numeric(show_df["Absolute SHAP"], errors="coerce")
            st.dataframe(show_df, use_container_width=True)

        st.subheader("Case-level Interpretation Summary")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        top3 = shap_df.head(3).copy()
        top3_lines = []
        for _, row in top3.iterrows():
            direction = "increased" if row["SHAP Value"] >= 0 else "decreased"
            top3_lines.append(
                f"- **{row['Display Name']}** {direction} the predicted response probability "
                f"(SHAP = {row['SHAP Value']:.4f})."
            )

        st.markdown(
            f"""
**Predicted outcome:** {predicted_label}  
**Response probability:** {positive_proba:.3f}  
**Decision threshold:** {FIXED_THRESHOLD:.6f}

**Most influential features in this case**  
{chr(10).join(top3_lines)}
            """
        )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 9. Model Card 页
# =========================================================
with tab_model:
    st.subheader("Model Card")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Cohort", f"{n_total}")
    c2.metric("Responders", f"{n_pos}")
    c3.metric("Non-responders", f"{n_neg}")
    c4.metric("Positive Rate", f"{pos_rate:.1%}")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Deployment Summary")
    st.write(
        "This application deploys a multimodal radiomics classifier for response prediction in unresectable advanced gastric cancer. "
        "The system is designed for single-case inference using manually entered feature values derived from venous-phase CT and elasticity imaging."
    )

    st.markdown("### Core Deployment Parameters")
    st.write(f"- Model alias: **{MODEL_ALIAS}**")
    st.write(f"- Classifier: **{TARGET_MODEL_NAME}**")
    st.write(f"- Fixed decision threshold: **{FIXED_THRESHOLD:.6f}**")
    st.write(f"- Positive class: **{POSITIVE_LABEL_NAME}**")
    st.write(f"- Negative class: **{NEGATIVE_LABEL_NAME}**")
    st.write(f"- Total input features: **{len(feature_names)}**")
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.0, 1.15], gap="large")

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Feature Group Table")
        st.dataframe(group_counts, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Feature Group Distribution")
        fig_group, _ = plot_feature_group_distribution(feature_group_df)
        st.pyplot(fig_group, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Expected Repository Files")
    st.code(
        "app.py\n"
        "SVM_rbf.pkl\n"
        "x_train.csv\n"
        "y_train.csv",
        language="text"
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 10. Guide 页
# =========================================================
with tab_guide:
    st.subheader("Interpretation Guide")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### How to Read the Prediction")
    st.write(
        "The model outputs a continuous probability of treatment response. "
        f"When the probability is greater than or equal to {FIXED_THRESHOLD:.6f}, "
        f"the patient is classified as {POSITIVE_LABEL_NAME}; otherwise, the patient is classified as {NEGATIVE_LABEL_NAME}."
    )

    st.markdown("### How to Read SHAP")
    st.write(
        "SHAP quantifies how individual features move the prediction away from the model baseline. "
        "Positive SHAP values push the prediction toward response, whereas negative SHAP values push it toward non-response."
    )

    st.markdown("### Why the Force Plot Shows Fewer Features")
    st.write(
        "Radiomics feature names are often very long and highly structured. "
        f"For visual clarity, the force plot is limited to the top {FORCE_TOP_N} contributors, "
        f"while the waterfall plot and table preserve the top {FULL_TOP_N} contributors."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Feature Group Abbreviations")
    st.write("- **V-INTRA**: Venous intratumoral features")
    st.write("- **V-PERI**: Venous peritumoral features")
    st.write("- **E-INTRA**: Elasticity intratumoral features")
    st.write("- **E-PERI**: Elasticity peritumoral features")
    st.write("- **RN50**: ResNet50-derived deep feature")
    st.write("- **LoG / GLSZM / GLCM / GLDM / NGTDM**: Common radiomics feature families")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Notes for Deployment")
    st.write(
        "To ensure stable cloud deployment, make sure the filename case matches exactly. "
        "On Streamlit Cloud and Linux servers, `SVM_rbf.pkl` and `SVM_RBF.pkl` are treated as different files."
    )
    st.write(
        "Because Kernel SHAP is computationally heavier than direct tree explainers, this deployment uses a reduced background set "
        "and a moderate number of sampling steps to balance interpretability and responsiveness."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 11. 页脚
# =========================================================
st.markdown(
    f'<div class="footer-note">{FOOTER_TEXT}</div>',
    unsafe_allow_html=True
)
