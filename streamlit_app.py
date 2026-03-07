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
# 1. 页面设置
# =========================================================
st.set_page_config(
    page_title="UAGC Immunochemotherapy Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. 根目录文件（不要文件夹版本）
# =========================================================
APP_DIR = Path(__file__).resolve().parent

# 注意：这里请与你实际文件名保持完全一致（Linux/Streamlit Cloud 区分大小写）
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
APP_SUBTITLE = "Multimodal venous-phase CT and elasticity radiomics model for individual treatment response estimation"

POSITIVE_LABEL_NAME = "Responder (CR/PR)"
NEGATIVE_LABEL_NAME = "Non-responder (SD/PD)"

# SHAP 显示策略
FORCE_TOP_N = 8     # force plot 默认只显示 8 个，避免名称重叠
FULL_TOP_N = 12     # waterfall / bar / table 显示 12 个
SHAP_NSAMPLES = 160
BACKGROUND_N = 30

# =========================================================
# 3. 页面样式
# =========================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #f4f7fb 100%);
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #1f2d3d;
        font-family: "Times New Roman", serif;
        letter-spacing: 0.2px;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #e6edf6;
        border-radius: 22px;
        padding: 1.35rem 1.6rem;
        box-shadow: 0 8px 24px rgba(31,45,61,0.08);
        margin-bottom: 1rem;
    }

    .note-card {
        background: #f4f8fc;
        border-left: 5px solid #4a90e2;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        color: #30465d;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dfe8f4;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        box-shadow: 0 4px 14px rgba(31,45,61,0.06);
        text-align: center;
        min-height: 132px;
    }

    .metric-title {
        font-size: 0.92rem;
        color: #607387;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #163a63;
        line-height: 1.2;
    }

    .metric-sub {
        font-size: 0.86rem;
        color: #7d8c9c;
        margin-top: 0.3rem;
    }

    .result-positive {
        background: linear-gradient(135deg, #eef9f1 0%, #ffffff 100%);
        border: 1px solid #b9e5c4;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        color: #1f5d2e;
        font-weight: 600;
        margin-top: 0.75rem;
    }

    .result-negative {
        background: linear-gradient(135deg, #fff4f4 0%, #ffffff 100%);
        border: 1px solid #f0c3c3;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        color: #8a2d2d;
        font-weight: 600;
        margin-top: 0.75rem;
    }

    .section-card {
        background: white;
        border: 1px solid #e7edf5;
        border-radius: 18px;
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: 0 4px 14px rgba(31,45,61,0.05);
        margin-bottom: 1rem;
    }

    .footer-note {
        font-size: 0.82rem;
        color: #708090;
        margin-top: 1rem;
    }

    .small-muted {
        color: #6d7d8d;
        font-size: 0.88rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid #e6edf5 !important;
        background: #ffffff !important;
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e7edf5;
        border-radius: 18px;
        padding: 1rem 1rem 0.2rem 1rem;
        box-shadow: 0 4px 14px rgba(31,45,61,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. 工具函数
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
        return "Venous Peritumoral Features"
    elif feature_name.startswith("Elasticityperi_"):
        return "Elasticity Peritumoral Features"
    elif feature_name.startswith("Venous_"):
        return "Venous Intratumoral Features"
    elif feature_name.startswith("Elasticity_"):
        return "Elasticity Intratumoral Features"
    else:
        return "Other Features"

def format_widget_label(name: str, max_len: int = 48) -> str:
    """
    输入控件标签：尽量保留辨识度，但避免过长。
    """
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
    """
    force plot 专用：必须更短，否则很容易重叠。
    """
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

def safe_stats(series: pd.Series):
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
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Invalid predict_proba output.")
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

    explanation = shap.Explanation(
        values=np.array(shap_values).reshape(-1),
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )
    return explanation

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
        display_names = [format_widget_label(n, max_len=44) for n in top_names]

    return shap.Explanation(
        values=top_values,
        base_values=float(explanation.base_values),
        data=top_data,
        feature_names=display_names
    )

def build_shap_table(explanation, top_n=12):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    data = np.array(explanation.data)

    order = np.argsort(np.abs(values))[::-1][:top_n]

    df = pd.DataFrame({
        "Full Feature Name": names[order],
        "Display Name": [format_widget_label(n, max_len=60) for n in names[order]],
        "Input Value": data[order],
        "SHAP Value": values[order],
        "Absolute SHAP": np.abs(values[order]),
        "Direction": np.where(values[order] >= 0, "Increase response probability", "Decrease response probability")
    }).sort_values("Absolute SHAP", ascending=False)

    return df

def render_force_plot_html(explanation, height=280):
    """
    force plot 容易被长特征名撑爆，所以：
    1）只显示 top 8
    2）使用强缩写标签
    3）在 iframe 中缩放显示
    """
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

    # 尽量压缩显示比例，减轻重叠
    html = html.replace(
        "<body>",
        """
        <body style="margin:0; background:#ffffff; overflow-x:auto; zoom:0.82;">
        """
    )

    components.html(html, height=height, scrolling=True)

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

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    colors = ["#d9534f" if v > 0 else "#3b82f6" for v in vals]

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
        ax.text(x, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}",
                va="center", ha="left", fontsize=10, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def make_input_widgets(feature_meta, feature_names):
    group_order = [
        "Venous Intratumoral Features",
        "Venous Peritumoral Features",
        "Elasticity Intratumoral Features",
        "Elasticity Peritumoral Features",
        "Other Features"
    ]

    grouped = {g: [] for g in group_order}
    for feat in feature_names:
        g = feature_meta[feat]["group"]
        grouped.setdefault(g, []).append(feat)

    user_values = {}

    for group_name in group_order:
        feats = grouped.get(group_name, [])
        if not feats:
            continue

        with st.expander(group_name, expanded=False):
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
                        label=format_widget_label(feat, max_len=52),
                        min_value=min_v,
                        max_value=max_v,
                        value=default_v,
                        help=help_text,
                        format="%.6f",
                        key=f"input_{feat}"
                    )

    return user_values

def make_interpretation_text(prob_pos, threshold):
    if prob_pos >= threshold:
        delta = prob_pos - threshold
        return (
            f"The estimated probability of response is {prob_pos:.3f}, which is above the predefined threshold "
            f"({threshold:.6f}) by {delta:.3f}. The model therefore classifies this patient as a "
            f"{POSITIVE_LABEL_NAME}."
        )
    else:
        delta = threshold - prob_pos
        return (
            f"The estimated probability of response is {prob_pos:.3f}, which is below the predefined threshold "
            f"({threshold:.6f}) by {delta:.3f}. The model therefore classifies this patient as a "
            f"{NEGATIVE_LABEL_NAME}."
        )

# =========================================================
# 5. 加载资源
# =========================================================
check_required_files()
model, x_train, y_train, feature_names, feature_meta, background = load_assets()
explainer = build_explainer(model, background)

# =========================================================
# 6. 顶部信息
# =========================================================
st.markdown(f"""
<div class="hero-card">
    <h1 style="margin-bottom:0.35rem;">{APP_TITLE}</h1>
    <div style="font-size:1.02rem; color:#4f647a; line-height:1.5;">
        {APP_SUBTITLE}<br>
        <b>Deployed model:</b> {MODEL_ALIAS}
        &nbsp;|&nbsp;
        <b>Classifier:</b> {TARGET_MODEL_NAME}
        &nbsp;|&nbsp;
        <b>Input modality:</b> Venous-phase CT + Elasticity radiomics
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="note-card">
This application predicts treatment response to <b>immunotherapy combined with chemotherapy</b> in patients with
<b>unresectable advanced gastric cancer</b>. The positive class corresponds to <b>{POSITIVE_LABEL_NAME}</b> and the
negative class corresponds to <b>{NEGATIVE_LABEL_NAME}</b>. The deployment uses a fixed decision threshold of
<b>{FIXED_THRESHOLD:.6f}</b>.
</div>
""", unsafe_allow_html=True)

# =========================================================
# 7. 侧边栏
# =========================================================
with st.sidebar:
    st.header("Model Overview")
    st.write(f"**Model alias:** {MODEL_ALIAS}")
    st.write(f"**Classifier:** {TARGET_MODEL_NAME}")
    st.write(f"**Model file:** `{MODEL_PATH.name}`")
    st.write(f"**Feature count:** {len(feature_names)}")
    st.write(f"**Decision threshold:** {FIXED_THRESHOLD:.6f}")
    st.write(f"**Force plot display:** Top {FORCE_TOP_N}")
    st.write(f"**Waterfall/Table display:** Top {FULL_TOP_N}")
    st.markdown("---")
    st.caption("Research-use interface only. This tool does not replace clinical judgment.")

# =========================================================
# 8. 输入区
# =========================================================
st.subheader("Patient Feature Input")

with st.form("prediction_form", clear_on_submit=False):
    st.markdown('<div class="small-muted">Enter patient-specific radiomics feature values below and submit the form to generate the prediction and SHAP explanation.</div>', unsafe_allow_html=True)

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
# 9. 预测结果
# =========================================================
if submitted:
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
            <div class="metric-value" style="font-size:1.18rem;">{predicted_label}</div>
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

    tab1, tab2, tab3 = st.tabs([
        f"SHAP Force Plot (Top {FORCE_TOP_N})",
        f"Waterfall / Contribution Plot (Top {FULL_TOP_N})",
        f"Top {FULL_TOP_N} Feature Table"
    ])

    with tab1:
        st.caption(
            f"Interactive force plot for the top {FORCE_TOP_N} features with the largest absolute SHAP values. "
            f"Feature names are aggressively abbreviated to avoid overlap."
        )
        render_force_plot_html(force_exp, height=290)

        st.info(
            f"For radiomics models with long feature names, force plots become crowded very easily. "
            f"Therefore, the app shows only the top {FORCE_TOP_N} features here, while the top {FULL_TOP_N} features remain available in the waterfall plot and table."
        )

    with tab2:
        st.caption(
            f"Waterfall plot and signed SHAP bar chart for the top {FULL_TOP_N} contributing features."
        )
        fig1 = plot_waterfall(full_exp, max_display=FULL_TOP_N)
        st.pyplot(fig1, use_container_width=True)

        fig2 = plot_top_contrib_bar(full_exp, max_display=FULL_TOP_N)
        st.pyplot(fig2, use_container_width=True)

    with tab3:
        st.caption(f"Top {FULL_TOP_N} features ranked by absolute SHAP magnitude.")
        display_df = shap_df.copy()
        display_df["Input Value"] = pd.to_numeric(display_df["Input Value"], errors="coerce")
        display_df["SHAP Value"] = pd.to_numeric(display_df["SHAP Value"], errors="coerce")
        display_df["Absolute SHAP"] = pd.to_numeric(display_df["Absolute SHAP"], errors="coerce")
        st.dataframe(display_df, use_container_width=True)

# =========================================================
# 10. 页脚
# =========================================================
st.markdown("""
<div class="footer-note">
Research-use interface for the multimodal VE-EF model.  
The decision threshold is fixed at deployment, and SHAP visualization is optimized for single-case interpretation of radiomics features with long names.
</div>
""", unsafe_allow_html=True)
