from pathlib import Path
import textwrap

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Polygon

st.set_page_config(
    page_title="UAGC Immunochemotherapy Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "SVM_rbf.pkl"
X_TRAIN_PATH = APP_DIR / "x_train.csv"
Y_TRAIN_PATH = APP_DIR / "y_train.csv"
SCALER_PATH = APP_DIR / "zscore_scaler.pkl"

FIXED_THRESHOLD = 0.4629352474478095

MODEL_ALIAS = "SVM_rbf"
TARGET_MODEL_NAME = "SVM_rbf"

APP_TITLE = "Prediction of Response to Immunotherapy Combined With Chemotherapy in Unresectable Advanced Gastric Cancer"
APP_SUBTITLE = "Multimodal venous-phase CT and elasticity radiomics model for individual treatment response estimation"

POSITIVE_LABEL_NAME = "Responder"
NEGATIVE_LABEL_NAME = "Non-responder"

SHAP_NSAMPLES = 160
BACKGROUND_N = 30
FORCE_MAX_DISPLAY = 10
FORCE_LABEL_WIDTH = 24
FORCE_MAX_ROWS_PER_COL = 4

st.markdown("""
<style>
    .main { background: linear-gradient(180deg, #f7f9fc 0%, #f4f7fb 100%); }
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }
    h1, h2, h3 { color: #1f2d3d; font-family: "Times New Roman", serif; letter-spacing: 0.2px; }
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

def check_required_files():
    required = [MODEL_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, SCALER_PATH]
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
    if "Venous" in feature_name:
        return "Venous-phase CT Features"
    return "Other Features"

def format_widget_label(name: str, max_len: int = 150) -> str:
    if len(name) > max_len:
        return name[:max_len - 1] + "…"
    return name

def transform_input_with_scaler(input_df_raw: pd.DataFrame, scaler, feature_names) -> pd.DataFrame:
    X = input_df_raw.copy()
    X = X[feature_names]
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=feature_names, index=X.index)

@st.cache_resource(show_spinner=True)
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    x_train = pd.read_csv(X_TRAIN_PATH)
    x_train = clean_columns(x_train)

    y_train = pd.read_csv(Y_TRAIN_PATH)
    if "label" not in y_train.columns:
        raise ValueError("y_train.csv missing column 'label'")
    y_train = y_train["label"].astype(int).values

    feature_names = list(x_train.columns)

    feature_meta = {}
    for col in feature_names:
        feature_meta[col] = {
            "group": infer_feature_group(col)
        }

    background = x_train.sample(min(BACKGROUND_N, len(x_train)), random_state=42)
    return model, scaler, x_train, y_train, feature_names, feature_meta, background

def predict_positive_proba(model, X_model_df: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X_model_df)
    return proba[:, 1]

@st.cache_resource(show_spinner=False)
def build_explainer(_model, background_df):
    def f(data):
        data_df = pd.DataFrame(data, columns=background_df.columns)
        return predict_positive_proba(_model, data_df)
    return shap.KernelExplainer(f, background_df.values)

def shap_for_single_case(explainer, input_df_model, nsamples=160):
    shap_values = explainer.shap_values(input_df_model.values, nsamples=nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = float(np.array(explainer.expected_value).reshape(-1)[0])
    else:
        base_value = float(explainer.expected_value)

    return shap.Explanation(
        values=np.array(shap_values).reshape(-1),
        base_values=base_value,
        data=input_df_model.iloc[0].values,
        feature_names=input_df_model.columns.tolist()
    )

def subset_explanation(explanation, top_n=None):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    data = np.array(explanation.data)

    if top_n is None:
        top_n = len(values)

    order = np.argsort(np.abs(values))[::-1][:top_n]

    return shap.Explanation(
        values=values[order],
        base_values=float(explanation.base_values),
        data=data[order],
        feature_names=names[order].tolist()
    )

def build_shap_table(explanation, input_df_raw=None, input_df_model=None):
    values = np.array(explanation.values)
    names = np.array(explanation.feature_names)
    order = np.argsort(np.abs(values))[::-1]

    data = {
        "Feature Name": names[order],
        "SHAP Value": values[order],
        "Absolute SHAP": np.abs(values[order]),
        "Direction": np.where(values[order] >= 0, "Increase response probability", "Decrease response probability")
    }

    if input_df_raw is not None:
        raw_series = input_df_raw.iloc[0]
        data["Raw Input"] = raw_series.loc[names[order]].values

    if input_df_model is not None:
        model_series = input_df_model.iloc[0]
        data["Standardized Input"] = model_series.loc[names[order]].values

    return pd.DataFrame(data).sort_values("Absolute SHAP", ascending=False)

def break_feature_name(name: str, width: int = 24) -> str:
    name = str(name)
    for sep in ["_", ".", "-"]:
        name = name.replace(sep, sep + "\u200b")
    return textwrap.fill(
        name,
        width=width,
        break_long_words=False,
        break_on_hyphens=False
    )

def draw_segment_patch(ax, x0, x1, y_center, height, color, arrow_size, direction="right"):
    if x1 < x0:
        x0, x1 = x1, x0

    width = x1 - x0
    arrow = min(arrow_size, width * 0.45) if width > 0 else arrow_size
    y0 = y_center - height / 2.0
    y1 = y_center + height / 2.0

    if direction == "right":
        pts = [
            (x0, y0),
            (x1 - arrow, y0),
            (x1, y_center),
            (x1 - arrow, y1),
            (x0, y1),
            (x0 + arrow, y_center)
        ]
    else:
        pts = [
            (x0 + arrow, y0),
            (x1, y0),
            (x1 - arrow, y_center),
            (x1, y1),
            (x0 + arrow, y1),
            (x0, y_center)
        ]

    poly = Polygon(
        pts,
        closed=True,
        facecolor=color,
        edgecolor="white",
        linewidth=1.0,
        joinstyle="round"
    )
    ax.add_patch(poly)

def layout_side_labels(
    segments,
    side,
    min_x,
    max_x,
    span,
    max_rows_per_col=4,
    label_width=24,
    y_start=0.70
):
    if not segments:
        return [], 0, y_start

    def x_anchor(seg):
        return (min(seg["x0"], seg["x1"]) + max(seg["x0"], seg["x1"])) / 2.0

    if side == "right":
        ordered = sorted(segments, key=x_anchor)
    else:
        ordered = sorted(segments, key=x_anchor, reverse=True)

    def make_label_text(seg):
        return f"[{seg['display_no']}] " + break_feature_name(seg["name"], width=label_width)

    def estimate_row_height(label_text):
        n_lines = label_text.count("\n") + 1
        return 0.10 + (n_lines - 1) * 0.060

    cols = []
    current_col = []

    for seg in ordered:
        item = dict(seg)
        item["label_text"] = make_label_text(seg)
        item["row_height"] = estimate_row_height(item["label_text"])

        if len(current_col) >= max_rows_per_col:
            cols.append(current_col)
            current_col = []

        current_col.append(item)

    if current_col:
        cols.append(current_col)

    label_offset = max(0.10 * span, 0.14)
    col_step = max(0.30 * span, 0.28)

    if side == "right":
        x_base = max_x + label_offset
    else:
        x_base = min_x - label_offset

    laid_out = []
    max_y_used = y_start

    for col_idx, col_segments in enumerate(cols):
        if side == "right":
            x_text = x_base + col_idx * col_step
            ha = "left"
        else:
            x_text = x_base - col_idx * col_step
            ha = "right"

        y_cursor = y_start + (0.05 if col_idx % 2 == 1 else 0.0)

        for seg in col_segments:
            laid_out.append({
                **seg,
                "x_text": x_text,
                "y_text": y_cursor,
                "ha": ha
            })
            max_y_used = max(max_y_used, y_cursor)
            y_cursor += seg["row_height"]

    return laid_out, len(cols), max_y_used

def plot_guided_force_like(explanation, prediction_value=None, base_value=None):
    values = np.array(explanation.values, dtype=float)
    names = np.array(explanation.feature_names, dtype=object)

    if base_value is None:
        base_value = float(explanation.base_values)
    else:
        base_value = float(base_value)

    if prediction_value is None:
        prediction_value = base_value + float(np.sum(values))
    else:
        prediction_value = float(prediction_value)

    item_info = []
    for i, (name, value) in enumerate(zip(names, values), start=1):
        item_info.append({
            "name": str(name),
            "value": float(value),
            "display_no": i
        })

    pos_items = [x for x in item_info if x["value"] >= 0]
    neg_items = [x for x in item_info if x["value"] < 0]

    pos_items = sorted(pos_items, key=lambda x: abs(x["value"]), reverse=True)
    neg_items = sorted(neg_items, key=lambda x: abs(x["value"]), reverse=True)

    neg_segments = []
    current_left = base_value
    for item in neg_items:
        v = item["value"]
        x0 = current_left + v
        x1 = current_left
        neg_segments.append({
            "name": item["name"],
            "value": v,
            "x0": x0,
            "x1": x1,
            "display_no": item["display_no"]
        })
        current_left = x0

    pos_segments = []
    current_right = base_value
    for item in pos_items:
        v = item["value"]
        x0 = current_right
        x1 = current_right + v
        pos_segments.append({
            "name": item["name"],
            "value": v,
            "x0": x0,
            "x1": x1,
            "display_no": item["display_no"]
        })
        current_right = x1

    xs = [base_value, prediction_value]
    for s in neg_segments + pos_segments:
        xs.extend([s["x0"], s["x1"]])

    min_x = min(xs)
    max_x = max(xs)
    span = max(max_x - min_x, 1e-6)

    left_labels, left_cols, left_max_y = layout_side_labels(
        neg_segments,
        side="left",
        min_x=min_x,
        max_x=max_x,
        span=span,
        max_rows_per_col=FORCE_MAX_ROWS_PER_COL,
        label_width=FORCE_LABEL_WIDTH,
        y_start=0.70
    )

    right_labels, right_cols, right_max_y = layout_side_labels(
        pos_segments,
        side="right",
        min_x=min_x,
        max_x=max_x,
        span=span,
        max_rows_per_col=FORCE_MAX_ROWS_PER_COL,
        label_width=FORCE_LABEL_WIDTH,
        y_start=0.70
    )

    col_step = max(0.30 * span, 0.28)
    side_pad = max(0.14 * span, 0.20)

    left_margin = side_pad + max(0, left_cols - 1) * col_step + 0.34 * span
    right_margin = side_pad + max(0, right_cols - 1) * col_step + 0.34 * span

    y_top = max(left_max_y, right_max_y) + 0.24
    n_label_items = max(len(left_labels), len(right_labels), 1)
    fig_height = min(max(6.0, 5.0 + 0.17 * n_label_items), 9.5)

    plt.close("all")
    old_rc = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Arial", "DejaVu Serif"],
        "font.size": 9,
        "axes.unicode_minus": False
    })

    fig, ax = plt.subplots(figsize=(18, fig_height), dpi=300)

    bar_y = 0.0
    bar_h = 0.34

    neg_color = "#F2C94C"
    pos_color = "#B73779"
    line_color = "#9AA3AF"
    text_color = "#2E3440"

    arrow_size = max(0.018 * span, 0.015)
    num_text_threshold = max(0.040 * span, 0.055)

    for seg in neg_segments:
        draw_segment_patch(
            ax=ax,
            x0=seg["x0"],
            x1=seg["x1"],
            y_center=bar_y,
            height=bar_h,
            color=neg_color,
            arrow_size=arrow_size,
            direction="left"
        )

    for seg in pos_segments:
        draw_segment_patch(
            ax=ax,
            x0=seg["x0"],
            x1=seg["x1"],
            y_center=bar_y,
            height=bar_h,
            color=pos_color,
            arrow_size=arrow_size,
            direction="right"
        )

    for seg in neg_segments + pos_segments:
        width = abs(seg["x1"] - seg["x0"])
        xc = (seg["x0"] + seg["x1"]) / 2.0

        if width >= num_text_threshold:
            ax.text(
                xc,
                bar_y,
                str(seg["display_no"]),
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if seg["value"] > 0 else "#3A3A3A",
                fontweight="bold"
            )
        else:
            ax.text(
                xc,
                bar_y + bar_h / 2 + 0.05,
                str(seg["display_no"]),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#4B5563",
                fontweight="bold"
            )

    def draw_label_item(item):
        x_left = min(item["x0"], item["x1"])
        x_right = max(item["x0"], item["x1"])
        x_anchor = (x_left + x_right) / 2.0
        y_anchor = bar_y + bar_h / 2.0
        y_knee = item["y_text"] - 0.035

        ax.plot(
            [x_anchor, x_anchor, item["x_text"]],
            [y_anchor, y_knee, y_knee],
            color=line_color,
            linewidth=0.8,
            solid_capstyle="round"
        )

        ax.text(
            item["x_text"],
            item["y_text"],
            item["label_text"],
            ha=item["ha"],
            va="bottom",
            fontsize=7.4,
            color=text_color,
            linespacing=1.04,
            clip_on=False
        )

    for item in left_labels:
        draw_label_item(item)

    for item in right_labels:
        draw_label_item(item)

    ax.axvline(base_value, color="#9AA0AA", linestyle=(0, (3, 3)), linewidth=1.0, zorder=0)
    ax.text(
        base_value,
        -0.46,
        "base value",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#6B7280"
    )

    ax.axvline(prediction_value, color="#7A7A7A", linestyle=(0, (3, 3)), linewidth=1.0, zorder=0)
    ax.text(
        prediction_value,
        0.16,
        f"f(x) = {prediction_value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#444444"
    )

    ax.set_xlim(min_x - left_margin, max_x + right_margin)
    ax.set_ylim(-0.62, y_top)
    ax.set_yticks([])
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("#AAB2BF")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.20)

    plt.rcParams.update(old_rc)
    return fig

def plot_waterfall(explanation, total_features):
    plt.close("all")
    old_rc = plt.rcParams.copy()
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Arial", "DejaVu Serif"]
    })

    fig_height = max(7.2, total_features * 0.35)
    plt.figure(figsize=(16, fig_height), dpi=300)
    shap.plots.waterfall(explanation, max_display=total_features, show=False)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.55, right=0.97, top=0.98, bottom=0.05)

    plt.rcParams.update(old_rc)
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
                col = cols[idx % 3]
                with col:
                    user_values[feat] = st.number_input(
                        label=format_widget_label(feat),
                        value=0.0,
                        format="%.6f",
                        key=f"input_{feat}"
                    )

    return user_values

def make_interpretation_text(prob_pos, threshold):
    if prob_pos >= threshold:
        return (
            f"The estimated probability of response is {prob_pos:.3f}, "
            f"which is above the threshold ({threshold:.6f}). "
            f"The model classifies this patient as a {POSITIVE_LABEL_NAME}."
        )
    return (
        f"The estimated probability of response is {prob_pos:.3f}, "
        f"which is below the threshold ({threshold:.6f}). "
        f"The model classifies this patient as a {NEGATIVE_LABEL_NAME}."
    )

check_required_files()
model, scaler, x_train, y_train, feature_names, feature_meta, background = load_assets()
explainer = build_explainer(model, background)
TOTAL_FEATURES = len(feature_names)

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

st.markdown(
    f'<div class="note-card">The deployment uses a fixed decision threshold of <b>{FIXED_THRESHOLD:.6f}</b>. Raw input values will be transformed by the saved Z-score scaler before prediction and SHAP analysis.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Model Overview")
    st.write(f"**Model:** {MODEL_ALIAS}")
    st.write(f"**Classifier:** {TARGET_MODEL_NAME}")
    st.write(f"**Total Features:** {TOTAL_FEATURES}")
    st.write(f"**Decision threshold:** {FIXED_THRESHOLD:.6f}")
    st.write(f"**Force plot features:** Top {min(FORCE_MAX_DISPLAY, TOTAL_FEATURES)}")
    st.write(f"**Scaler:** {SCALER_PATH.name}")
    st.markdown("---")
    st.caption("Research-use interface only. This tool does not replace clinical judgment.")

st.subheader("Patient Feature Input")

with st.form("prediction_form", clear_on_submit=False):
    st.markdown(
        '<div class="small-muted">Enter raw patient-specific radiomics feature values below. The app will automatically apply the training-time Z-score transformation before prediction.</div>',
        unsafe_allow_html=True
    )
    user_inputs = make_input_widgets(feature_meta, feature_names)

    c1, c2, c3 = st.columns([1.2, 1.2, 3.6])
    with c1:
        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)
    with c2:
        preview = st.form_submit_button("Preview Inputs", use_container_width=True)

input_df_raw = pd.DataFrame([[user_inputs[f] for f in feature_names]], columns=feature_names)
input_df_model = transform_input_with_scaler(input_df_raw, scaler, feature_names)

if preview and not submitted:
    preview_df = pd.DataFrame({
        "Raw Input": input_df_raw.iloc[0],
        "Standardized Input": input_df_model.iloc[0]
    })
    with st.expander("Current Input Table", expanded=True):
        st.dataframe(preview_df, use_container_width=True)

if submitted:
    positive_proba = float(predict_positive_proba(model, input_df_model)[0])
    negative_proba = 1 - positive_proba
    predicted_class = 1 if positive_proba >= FIXED_THRESHOLD else 0
    predicted_label = POSITIVE_LABEL_NAME if predicted_class == 1 else NEGATIVE_LABEL_NAME

    st.subheader("Prediction Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Response Probability</div><div class="metric-value">{positive_proba * 100:.1f}%</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Predicted Category</div><div class="metric-value" style="font-size:1.18rem;">{predicted_label}</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Non-response Probability</div><div class="metric-value">{negative_proba * 100:.1f}%</div></div>',
            unsafe_allow_html=True
        )

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

    st.subheader("Model Explanation (SHAP)")

    explanation_ready = False
    explanation_full = None
    force_exp = None
    full_exp = None
    shap_df = None

    try:
        with st.spinner("Computing SHAP explanation..."):
            explanation_full = shap_for_single_case(explainer, input_df_model, nsamples=SHAP_NSAMPLES)
            force_exp = subset_explanation(
                explanation_full,
                top_n=min(FORCE_MAX_DISPLAY, TOTAL_FEATURES)
            )
            full_exp = subset_explanation(
                explanation_full,
                top_n=TOTAL_FEATURES
            )
            shap_df = build_shap_table(
                explanation_full,
                input_df_raw=input_df_raw,
                input_df_model=input_df_model
            )
        explanation_ready = True
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

    if explanation_ready:
        tab1, tab2, tab3 = st.tabs(["SHAP Guided Force Plot", "Waterfall Plot", "Features Table"])

        with tab1:
            st.caption(
                f"Custom force-style SHAP plot with leader lines and indexed labels "
                f"(top {min(FORCE_MAX_DISPLAY, TOTAL_FEATURES)} features by absolute SHAP value)."
            )
            try:
                fig_force = plot_guided_force_like(
                    explanation=force_exp,
                    prediction_value=positive_proba,
                    base_value=float(explanation_full.base_values)
                )
                st.pyplot(fig_force, use_container_width=True)
            except Exception as e:
                st.error(f"SHAP guided force plot rendering failed: {e}")

        with tab2:
            st.caption("Waterfall plot using original feature names.")
            try:
                fig1 = plot_waterfall(full_exp, total_features=TOTAL_FEATURES)
                st.pyplot(fig1, use_container_width=True)
            except Exception as e:
                st.error(f"SHAP waterfall plot rendering failed: {e}")

        with tab3:
            st.caption("Features ranked by absolute SHAP magnitude.")
            try:
                display_df = shap_df.copy()
                for col in ["Raw Input", "Standardized Input", "SHAP Value", "Absolute SHAP"]:
                    if col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
                st.dataframe(display_df, use_container_width=True)
            except Exception as e:
                st.error(f"SHAP feature table rendering failed: {e}")

st.markdown("""
<div class="footer-note">
Research-use interface for the multimodal model.
</div>
""", unsafe_allow_html=True)
