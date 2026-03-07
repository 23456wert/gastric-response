import os
from pathlib import Path
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# 1. 页面设置
# =========================================================
st.set_page_config(
    page_title="Gastric Cancer Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. 所有文件都在当前目录（无文件夹版本）
# =========================================================
APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "SVM_rbf.pkl"
X_TRAIN_PATH = APP_DIR / "x_train.csv"
Y_TRAIN_PATH = APP_DIR / "y_train.csv"
METRIC_PATH = APP_DIR / "model_performance_metrics.csv"

MODEL_ALIAS = "VE-EF"
TARGET_MODEL_NAME = "SVM_rbf"

POSITIVE_LABEL_NAME = "Responder (CR/PR)"
NEGATIVE_LABEL_NAME = "Non-responder (SD/PD)"

# =========================================================
# 3. 样式
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        color: #1f2d3d;
        font-family: "Times New Roman", serif;
    }
    .top-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 18px rgba(31,45,61,0.08);
        border: 1px solid #e8edf5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
        border: 1px solid #dfe8f4;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 3px 10px rgba(31,45,61,0.06);
        text-align: center;
    }
    .metric-title {
        font-size: 0.95rem;
        color: #5b6b7f;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #163a63;
    }
    .metric-sub {
        font-size: 0.88rem;
        color: #7b8a9a;
    }
    .section-note {
        background: #f3f7fb;
        border-left: 5px solid #4a90e2;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        color: #2f4257;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .footer-note {
        font-size: 0.82rem;
        color: #708090;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. 工具函数
# =========================================================
def check_required_files():
    required = [MODEL_PATH, X_TRAIN_PATH, Y_TRAIN_PATH]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        st.error("Missing required files in repository root:")
        for p in missing:
            st.write(f"- {p}")
        st.stop()

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["ID", "Id", "id", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

def normalize_dataset_name(x: str) -> str:
    s = str(x).strip().lower()
    if "fit" in s:
        return "Train_Fit"
    if "test" in s:
        return "Test"
    if "train" in s or "oof" in s:
        return "Train"
    return str(x)

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

def pretty_feature_label(name: str) -> str:
    return name.replace("_", " · ")

def load_threshold():
    if not METRIC_PATH.exists():
        return 0.5
    try:
        df = pd.read_csv(METRIC_PATH)
        if not {"Dataset", "Model", "Threshold"}.issubset(df.columns):
            return 0.5
        df["Dataset_std"] = df["Dataset"].apply(normalize_dataset_name)
        sub = df[
            (df["Model"].astype(str).str.strip() == TARGET_MODEL_NAME) &
            (df["Dataset_std"] == "Test")
        ]
        if sub.empty:
            sub = df[
                (df["Model"].astype(str).str.strip() == TARGET_MODEL_NAME) &
                (df["Dataset_std"] == "Train")
            ]
        if sub.empty:
            return 0.5
        th = pd.to_numeric(sub.iloc[0]["Threshold"], errors="coerce")
        return 0.5 if pd.isna(th) else float(th)
    except Exception:
        return 0.5

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
        s = pd.to_numeric(x_train[col], errors="coerce")
        feature_meta[col] = {
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "median": float(np.nanmedian(s)),
            "q1": float(np.nanpercentile(s, 25)),
            "q3": float(np.nanpercentile(s, 75)),
            "group": infer_feature_group(col),
        }

    background = x_train.sample(min(50, len(x_train)), random_state=42)
    threshold = load_threshold()
    return model, x_train, y_train, feature_names, feature_meta, background, threshold

def predict_positive_proba(model, X_df: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X_df)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Model predict_proba output format is invalid.")
    return proba[:, 1]

@st.cache_resource(show_spinner=False)
def build_explainer(_model, background_df):
    def f(data):
        data_df = pd.DataFrame(data, columns=background_df.columns)
        return predict_positive_proba(_model, data_df)
    return shap.KernelExplainer(f, background_df.values)

def shap_for_single_case(explainer, input_df, nsamples=200):
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

def plot_waterfall(explanation, max_display=15):
    fig = plt.figure(figsize=(10, 7), dpi=300)
    try:
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
    except Exception:
        plt.close(fig)
        fig = plot_top_contrib_bar(explanation, max_display=max_display)
    return fig

def plot_top_contrib_bar(explanation, max_display=15):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    order = np.argsort(np.abs(values))[::-1][:max_display]
    vals = values[order]
    labels = names[order]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    colors = ["#d9534f" if v > 0 else "#3b82f6" for v in vals]
    ax.barh(range(len(vals)), vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("SHAP value", fontsize=11)
    ax.set_title("Top Feature Contributions", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    return fig

def make_input_widgets(feature_meta, feature_names):
    grouped = {}
    for f in feature_names:
        g = feature_meta[f]["group"]
        grouped.setdefault(g, []).append(f)

    user_values = {}

    for group_name, feats in grouped.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(3)
            for idx, feat in enumerate(feats):
                meta = feature_meta[feat]
                col = cols[idx % 3]

                min_v = meta["min"]
                max_v = meta["max"]
                default_v = meta["median"]

                if min_v == max_v:
                    min_v = min_v - 1e-6
                    max_v = max_v + 1e-6

                help_text = (
                    f"Train range: {meta['min']:.4f} to {meta['max']:.4f}\n"
                    f"Median: {meta['median']:.4f}\n"
                    f"IQR: {meta['q1']:.4f} - {meta['q3']:.4f}"
                )

                with col:
                    user_values[feat] = st.number_input(
                        label=pretty_feature_label(feat),
                        min_value=float(min_v),
                        max_value=float(max_v),
                        value=float(default_v),
                        help=help_text,
                        format="%.6f"
                    )

    return user_values

# =========================================================
# 5. 加载资源
# =========================================================
check_required_files()
model, x_train, y_train, feature_names, feature_meta, background, threshold = load_assets()
explainer = build_explainer(model, background)

# =========================================================
# 6. 页面头部
# =========================================================
st.markdown(f"""
<div class="top-card">
    <h1 style="margin-bottom:0.3rem;">Multimodal Prediction of Chemotherapy Response in Unresectable Advanced Gastric Cancer</h1>
    <div style="font-size:1.0rem; color:#4b5d73;">
        Best deployed model: <b>{MODEL_ALIAS}</b> &nbsp;|&nbsp;
        Architecture: <b>{TARGET_MODEL_NAME}</b> &nbsp;|&nbsp;
        Input: <b>Venous CT + Elasticity Map Features</b>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-note">
This application estimates the probability of treatment response for unresectable advanced gastric cancer using the deployed multimodal model.
Positive prediction corresponds to <b>Responder (CR/PR)</b>; negative prediction corresponds to <b>Non-responder (SD/PD)</b>.
</div>
""", unsafe_allow_html=True)

# =========================================================
# 7. 侧边栏
# =========================================================
with st.sidebar:
    st.header("Model Overview")
    st.write(f"**Model alias:** {MODEL_ALIAS}")
    st.write(f"**Model file:** `{MODEL_PATH.name}`")
    st.write(f"**Feature count:** {len(feature_names)}")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    st.write(f"**Positive class:** {POSITIVE_LABEL_NAME}")
    st.write(f"**Negative class:** {NEGATIVE_LABEL_NAME}")
    st.markdown("---")
    st.caption("For research demonstration only. Not a substitute for clinical judgment.")

# =========================================================
# 8. 输入区
# =========================================================
st.subheader("Patient Feature Input")
user_inputs = make_input_widgets(feature_meta, feature_names)
input_df = pd.DataFrame([[user_inputs[f] for f in feature_names]], columns=feature_names)

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    run_pred = st.button("Run Prediction", type="primary", use_container_width=True)
with col_btn2:
    show_input = st.button("Show Input Table", use_container_width=True)

if show_input:
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# =========================================================
# 9. 预测结果
# =========================================================
if run_pred:
    positive_proba = float(predict_positive_proba(model, input_df)[0])
    negative_proba = 1 - positive_proba
    predicted_class = 1 if positive_proba >= threshold else 0
    predicted_label = POSITIVE_LABEL_NAME if predicted_class == 1 else NEGATIVE_LABEL_NAME

    st.subheader("Prediction Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Response Probability</div>
            <div class="metric-value">{positive_proba*100:.1f}%</div>
            <div class="metric-sub">Positive class probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Category</div>
            <div class="metric-value" style="font-size:1.2rem;">{predicted_label}</div>
            <div class="metric-sub">Threshold = {threshold:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Non-response Probability</div>
            <div class="metric-value">{negative_proba*100:.1f}%</div>
            <div class="metric-sub">Negative class probability</div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(round(positive_proba * 100)))

    st.markdown("""
    <div class="section-note">
    Interpretation: the displayed probability corresponds to the likelihood of treatment response (CR/PR) estimated by the deployed multimodal model.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Model Explanation (SHAP)")
    with st.spinner("Computing SHAP explanation..."):
        explanation = shap_for_single_case(explainer, input_df, nsamples=200)

    fig1 = plot_waterfall(explanation, max_display=15)
    st.pyplot(fig1, use_container_width=True)

    fig2 = plot_top_contrib_bar(explanation, max_display=15)
    st.pyplot(fig2, use_container_width=True)

    shap_df = pd.DataFrame({
        "Feature": explanation.feature_names,
        "Input Value": explanation.data,
        "SHAP Value": explanation.values,
        "Absolute SHAP": np.abs(explanation.values)
    }).sort_values("Absolute SHAP", ascending=False)

    st.markdown("**Top 15 contributing features**")
    st.dataframe(shap_df.head(15), use_container_width=True)

# =========================================================
# 10. 页脚
# =========================================================
st.markdown("""
<div class="footer-note">
Research-use interface for the VE-EFrad multimodal model.
All deployment files are loaded from the repository root for GitHub upload compatibility.
</div>
""", unsafe_allow_html=True)