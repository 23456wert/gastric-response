from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import textwrap

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Polygon


st.set_page_config(
    page_title="UAGC Immunochemotherapy Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    model_path: Path = APP_DIR / "SVM_rbf.pkl"
    x_train_path: Path = APP_DIR / "x_train.csv"
    y_train_path: Path = APP_DIR / "y_train.csv"
    scaler_path: Path = APP_DIR / "zscore_scaler.pkl"
    fixed_threshold: float = 0.4629352474478095
    model_alias: str = "SVM_rbf"
    target_model_name: str = "SVM_rbf"
    app_title: str = (
        "Prediction of Response to Immunotherapy Combined With Chemotherapy "
        "in Unresectable Advanced Gastric Cancer"
    )
    app_subtitle: str = (
        "Multimodal venous-phase CT and elasticity radiomics model for "
        "individual treatment response estimation"
    )
    positive_label: str = "Responder"
    negative_label: str = "Non-responder"
    shap_nsamples: int = 160
    background_n: int = 30
    force_max_display: int = 10
    force_label_width: int = 24
    force_max_rows_per_col: int = 4


CFG = AppConfig()


APP_STYLE = """
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
"""

st.markdown(APP_STYLE, unsafe_allow_html=True)


@dataclass
class Assets:
    model: object
    scaler: object
    x_train: pd.DataFrame
    y_train: np.ndarray
    feature_names: list[str]
    feature_meta: dict[str, dict[str, str]]
    background: pd.DataFrame


@dataclass
class PredictionResult:
    positive_proba: float
    negative_proba: float
    predicted_class: int
    predicted_label: str


@dataclass
class ExplanationBundle:
    full: shap.Explanation
    force: shap.Explanation
    waterfall: shap.Explanation
    table: pd.DataFrame


def stop_if_missing(paths: list[Path]) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if not missing:
        return
    st.error("Missing required files in repository root:")
    for name in missing:
        st.write(f"- {name}")
    st.stop()


@contextmanager
def mpl_rc(params: dict):
    old_rc = plt.rcParams.copy()
    plt.rcParams.update(params)
    try:
        yield
    finally:
        plt.rcParams.update(old_rc)


def normalize_feature_names(columns) -> pd.Index:
    return (
        pd.Index(columns)
        .astype(str)
        .str.replace(r"[-(),\s]+", "_", regex=True)
        .str.strip("_")
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    removable = [c for c in ("ID", "Id", "id", "Unnamed: 0") if c in df.columns]
    if removable:
        df = df.drop(columns=removable, errors="ignore")
    df.columns = normalize_feature_names(df.columns)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def infer_feature_group(feature_name: str) -> str:
    if "Elasticity" in feature_name:
        return "Elasticity Features"
    if "Venous" in feature_name:
        return "Venous-phase CT Features"
    return "Other Features"


def format_widget_label(name: str, max_len: int = 150) -> str:
    return name if len(name) <= max_len else f"{name[:max_len - 1]}…"


def break_feature_name(name: str, width: int) -> str:
    name = str(name)
    for sep in ("_", ".", "-"):
        name = name.replace(sep, sep + "\u200b")
    return textwrap.fill(name, width=width, break_long_words=False, break_on_hyphens=False)


def align_estimator_feature_names(estimator, feature_names: list[str]) -> None:
    if hasattr(estimator, "n_features_in_"):
        if int(estimator.n_features_in_) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: x_train.csv has {len(feature_names)} features, "
                f"but saved estimator expects {int(estimator.n_features_in_)} features."
            )
    if hasattr(estimator, "feature_names_in_"):
        estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)


def render_metric_card(title: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        (
            '<div class="metric-card">'
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f"{sub_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_result_banner(text: str, positive: bool) -> None:
    css_class = "result-positive" if positive else "result-negative"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def render_sidebar(total_features: int) -> None:
    with st.sidebar:
        st.header("Model Overview")
        st.write(f"**Model:** {CFG.model_alias}")
        st.write(f"**Classifier:** {CFG.target_model_name}")
        st.write(f"**Total Features:** {total_features}")
        st.write(f"**Decision threshold:** {CFG.fixed_threshold:.6f}")
        st.write(f"**Force plot features:** Top {min(CFG.force_max_display, total_features)}")
        st.write(f"**Scaler:** {CFG.scaler_path.name}")
        st.markdown("---")
        st.caption("Research-use interface only. This tool does not replace clinical judgment.")


@st.cache_resource(show_spinner=True)
def load_assets() -> Assets:
    model = joblib.load(CFG.model_path)
    scaler = joblib.load(CFG.scaler_path)

    x_train = clean_columns(pd.read_csv(CFG.x_train_path))
    feature_names = list(x_train.columns)

    align_estimator_feature_names(scaler, feature_names)
    align_estimator_feature_names(model, feature_names)

    y_train_df = pd.read_csv(CFG.y_train_path)
    if "label" not in y_train_df.columns:
        raise ValueError("y_train.csv missing column 'label'")
    y_train = y_train_df["label"].astype(int).to_numpy()

    feature_meta = {name: {"group": infer_feature_group(name)} for name in feature_names}
    background = x_train[feature_names].sample(min(CFG.background_n, len(x_train)), random_state=42)

    return Assets(
        model=model,
        scaler=scaler,
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        feature_meta=feature_meta,
        background=background,
    )


@st.cache_resource(show_spinner=False)
def build_explainer(_model, background_df: pd.DataFrame):
    def predict_fn(data):
        data_df = pd.DataFrame(data, columns=background_df.columns)
        return predict_positive_proba(_model, data_df)

    return shap.KernelExplainer(predict_fn, background_df.values)


def transform_input_with_scaler(input_df_raw: pd.DataFrame, scaler, feature_names: list[str]) -> pd.DataFrame:
    x = input_df_raw[feature_names].copy()
    x_scaled = scaler.transform(x)
    return pd.DataFrame(x_scaled, columns=feature_names, index=x.index)


def predict_positive_proba(model, x_model_df: pd.DataFrame) -> np.ndarray:
    x = x_model_df.copy()
    return model.predict_proba(x)[:, 1]


def build_input_data(user_inputs: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame([[user_inputs[name] for name in feature_names]], columns=feature_names)


def run_prediction(model, input_df_model: pd.DataFrame) -> PredictionResult:
    positive_proba = float(predict_positive_proba(model, input_df_model)[0])
    predicted_class = int(positive_proba >= CFG.fixed_threshold)
    predicted_label = CFG.positive_label if predicted_class == 1 else CFG.negative_label
    return PredictionResult(
        positive_proba=positive_proba,
        negative_proba=1 - positive_proba,
        predicted_class=predicted_class,
        predicted_label=predicted_label,
    )


def shap_for_single_case(explainer, input_df_model: pd.DataFrame, nsamples: int) -> shap.Explanation:
    shap_values = explainer.shap_values(input_df_model.values, nsamples=nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = float(np.asarray(expected_value).reshape(-1)[0])
    else:
        base_value = float(expected_value)

    return shap.Explanation(
        values=np.asarray(shap_values).reshape(-1),
        base_values=base_value,
        data=input_df_model.iloc[0].to_numpy(),
        feature_names=input_df_model.columns.tolist(),
    )


def subset_explanation(explanation: shap.Explanation, top_n: int | None = None) -> shap.Explanation:
    values = np.asarray(explanation.values)
    names = np.asarray(explanation.feature_names)
    data = np.asarray(explanation.data)

    if top_n is None:
        top_n = len(values)

    order = np.argsort(np.abs(values))[::-1][:top_n]
    return shap.Explanation(
        values=values[order],
        base_values=float(explanation.base_values),
        data=data[order],
        feature_names=names[order].tolist(),
    )


def build_shap_table(
    explanation: shap.Explanation,
    input_df_raw: pd.DataFrame | None = None,
    input_df_model: pd.DataFrame | None = None,
) -> pd.DataFrame:
    values = np.asarray(explanation.values)
    names = np.asarray(explanation.feature_names)
    order = np.argsort(np.abs(values))[::-1]

    data = {
        "Feature Name": names[order],
        "SHAP Value": values[order],
        "Absolute SHAP": np.abs(values[order]),
        "Direction": np.where(values[order] >= 0, "Increase response probability", "Decrease response probability"),
    }

    if input_df_raw is not None:
        raw_series = input_df_raw.iloc[0]
        data["Raw Input"] = raw_series.loc[names[order]].to_numpy()

    if input_df_model is not None:
        model_series = input_df_model.iloc[0]
        data["Standardized Input"] = model_series.loc[names[order]].to_numpy()

    return pd.DataFrame(data).sort_values("Absolute SHAP", ascending=False)


def prepare_explanations(explainer, input_df_raw: pd.DataFrame, input_df_model: pd.DataFrame, total_features: int) -> ExplanationBundle:
    full = shap_for_single_case(explainer, input_df_model, nsamples=CFG.shap_nsamples)
    force = subset_explanation(full, top_n=min(CFG.force_max_display, total_features))
    waterfall = subset_explanation(full, top_n=total_features)
    table = build_shap_table(full, input_df_raw=input_df_raw, input_df_model=input_df_model)
    return ExplanationBundle(full=full, force=force, waterfall=waterfall, table=table)


def draw_segment_patch(ax, x0, x1, y_center, height, color, arrow_size, direction: str) -> None:
    if x1 < x0:
        x0, x1 = x1, x0

    width = x1 - x0
    arrow = min(arrow_size, width * 0.45) if width > 0 else arrow_size
    y0, y1 = y_center - height / 2.0, y_center + height / 2.0

    if direction == "right":
        points = [(x0, y0), (x1 - arrow, y0), (x1, y_center), (x1 - arrow, y1), (x0, y1), (x0 + arrow, y_center)]
    else:
        points = [(x0 + arrow, y0), (x1, y0), (x1 - arrow, y_center), (x1, y1), (x0 + arrow, y1), (x0, y_center)]

    ax.add_patch(
        Polygon(
            points,
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            joinstyle="round",
        )
    )


def layout_side_labels(
    segments: list[dict],
    side: str,
    min_x: float,
    max_x: float,
    span: float,
    max_rows_per_col: int,
    label_width: int,
    y_start: float = 0.70,
):
    if not segments:
        return [], 0, y_start

    def x_anchor(seg):
        return (min(seg["x0"], seg["x1"]) + max(seg["x0"], seg["x1"])) / 2.0

    ordered = sorted(segments, key=x_anchor, reverse=(side == "left"))

    def label_text(seg: dict) -> str:
        return f"[{seg['display_no']}] " + break_feature_name(seg["name"], width=label_width)

    def row_height(text: str) -> float:
        line_count = text.count("\n") + 1
        return 0.10 + (line_count - 1) * 0.060

    columns: list[list[dict]] = []
    current_col: list[dict] = []

    for seg in ordered:
        item = dict(seg)
        item["label_text"] = label_text(seg)
        item["row_height"] = row_height(item["label_text"])

        if len(current_col) >= max_rows_per_col:
            columns.append(current_col)
            current_col = []
        current_col.append(item)

    if current_col:
        columns.append(current_col)

    label_offset = max(0.10 * span, 0.14)
    col_step = max(0.30 * span, 0.28)
    x_base = max_x + label_offset if side == "right" else min_x - label_offset

    laid_out = []
    max_y_used = y_start

    for col_idx, col in enumerate(columns):
        if side == "right":
            x_text = x_base + col_idx * col_step
            ha = "left"
        else:
            x_text = x_base - col_idx * col_step
            ha = "right"

        y_cursor = y_start + (0.05 if col_idx % 2 else 0.0)
        for seg in col:
            laid_out.append({**seg, "x_text": x_text, "y_text": y_cursor, "ha": ha})
            max_y_used = max(max_y_used, y_cursor)
            y_cursor += seg["row_height"]

    return laid_out, len(columns), max_y_used


def plot_guided_force_like(explanation: shap.Explanation, prediction_value: float | None = None, base_value: float | None = None):
    values = np.asarray(explanation.values, dtype=float)
    names = np.asarray(explanation.feature_names, dtype=object)

    base_value = float(explanation.base_values if base_value is None else base_value)
    prediction_value = float(base_value + np.sum(values) if prediction_value is None else prediction_value)

    items = [{"name": str(name), "value": float(value), "display_no": i} for i, (name, value) in enumerate(zip(names, values), start=1)]
    pos_items = sorted((x for x in items if x["value"] >= 0), key=lambda x: abs(x["value"]), reverse=True)
    neg_items = sorted((x for x in items if x["value"] < 0), key=lambda x: abs(x["value"]), reverse=True)

    def build_segments(source_items: list[dict], start: float, positive: bool) -> list[dict]:
        segments = []
        current = start
        for item in source_items:
            value = item["value"]
            x0, x1 = (current, current + value) if positive else (current + value, current)
            segments.append({"name": item["name"], "value": value, "x0": x0, "x1": x1, "display_no": item["display_no"]})
            current = x1 if positive else x0
        return segments

    neg_segments = build_segments(neg_items, base_value, positive=False)
    pos_segments = build_segments(pos_items, base_value, positive=True)

    xs = [base_value, prediction_value] + [v for seg in (neg_segments + pos_segments) for v in (seg["x0"], seg["x1"])]
    min_x, max_x = min(xs), max(xs)
    span = max(max_x - min_x, 1e-6)

    left_labels, left_cols, left_max_y = layout_side_labels(
        neg_segments,
        side="left",
        min_x=min_x,
        max_x=max_x,
        span=span,
        max_rows_per_col=CFG.force_max_rows_per_col,
        label_width=CFG.force_label_width,
    )
    right_labels, right_cols, right_max_y = layout_side_labels(
        pos_segments,
        side="right",
        min_x=min_x,
        max_x=max_x,
        span=span,
        max_rows_per_col=CFG.force_max_rows_per_col,
        label_width=CFG.force_label_width,
    )

    col_step = max(0.30 * span, 0.28)
    side_pad = max(0.14 * span, 0.20)
    left_margin = side_pad + max(0, left_cols - 1) * col_step + 0.34 * span
    right_margin = side_pad + max(0, right_cols - 1) * col_step + 0.34 * span

    y_top = max(left_max_y, right_max_y) + 0.24
    n_label_items = max(len(left_labels), len(right_labels), 1)
    fig_height = min(max(6.0, 5.0 + 0.17 * n_label_items), 9.5)

    plt.close("all")
    with mpl_rc({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Arial", "DejaVu Serif"],
        "font.size": 9,
        "axes.unicode_minus": False,
    }):
        fig, ax = plt.subplots(figsize=(18, fig_height), dpi=300)

        bar_y, bar_h = 0.0, 0.34
        neg_color, pos_color = "#F2C94C", "#B73779"
        line_color, text_color = "#9AA3AF", "#2E3440"
        arrow_size = max(0.018 * span, 0.015)
        num_text_threshold = max(0.040 * span, 0.055)

        for seg in neg_segments:
            draw_segment_patch(ax, seg["x0"], seg["x1"], bar_y, bar_h, neg_color, arrow_size, direction="left")
        for seg in pos_segments:
            draw_segment_patch(ax, seg["x0"], seg["x1"], bar_y, bar_h, pos_color, arrow_size, direction="right")

        for seg in neg_segments + pos_segments:
            width = abs(seg["x1"] - seg["x0"])
            x_center = (seg["x0"] + seg["x1"]) / 2.0
            if width >= num_text_threshold:
                ax.text(
                    x_center,
                    bar_y,
                    str(seg["display_no"]),
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="white" if seg["value"] > 0 else "#3A3A3A",
                    fontweight="bold",
                )
            else:
                ax.text(
                    x_center,
                    bar_y + bar_h / 2 + 0.05,
                    str(seg["display_no"]),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#4B5563",
                    fontweight="bold",
                )

        def draw_label(item: dict) -> None:
            x_anchor = (min(item["x0"], item["x1"]) + max(item["x0"], item["x1"])) / 2.0
            y_anchor = bar_y + bar_h / 2.0
            y_knee = item["y_text"] - 0.035
            ax.plot([x_anchor, x_anchor, item["x_text"]], [y_anchor, y_knee, y_knee], color=line_color, linewidth=0.8, solid_capstyle="round")
            ax.text(
                item["x_text"],
                item["y_text"],
                item["label_text"],
                ha=item["ha"],
                va="bottom",
                fontsize=7.4,
                color=text_color,
                linespacing=1.04,
                clip_on=False,
            )

        for item in left_labels + right_labels:
            draw_label(item)

        ax.axvline(base_value, color="#9AA0AA", linestyle=(0, (3, 3)), linewidth=1.0, zorder=0)
        ax.text(base_value, -0.46, "base value", ha="center", va="top", fontsize=8.5, color="#6B7280")

        ax.axvline(prediction_value, color="#7A7A7A", linestyle=(0, (3, 3)), linewidth=1.0, zorder=0)
        ax.text(prediction_value, 0.16, f"f(x) = {prediction_value:.3f}", ha="center", va="bottom", fontsize=8.8, color="#444444")

        ax.set_xlim(min_x - left_margin, max_x + right_margin)
        ax.set_ylim(-0.62, y_top)
        ax.set_yticks([])
        ax.set_xlabel("SHAP value", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(False)

        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color("#AAB2BF")
        ax.spines["bottom"].set_linewidth(1.0)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.20)
        return fig


def plot_waterfall(explanation: shap.Explanation, total_features: int):
    plt.close("all")
    fig_height = max(7.2, total_features * 0.35)
    with mpl_rc({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Arial", "DejaVu Serif"],
    }):
        plt.figure(figsize=(16, fig_height), dpi=300)
        shap.plots.waterfall(explanation, max_display=total_features, show=False)
        fig = plt.gcf()
        fig.subplots_adjust(left=0.55, right=0.97, top=0.98, bottom=0.05)
        return fig


def plot_probability_bar(prob_pos: float):
    prob_neg = 1 - prob_pos
    labels = [CFG.negative_label, CFG.positive_label]
    probs = [prob_neg, prob_pos]
    colors = ["#b9c6d6", "#2f7ed8"]

    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=300)
    bars = ax.barh(labels, probs, color=colors, edgecolor="none")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", fontsize=11, fontweight="bold")
    ax.set_title("Predicted Class Probabilities", fontsize=13, fontweight="bold")

    for bar, value in zip(bars, probs):
        ax.text(
            min(value + 0.015, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def make_input_widgets(feature_meta: dict[str, dict[str, str]], feature_names: list[str]) -> dict[str, float]:
    group_order = ["Elasticity Features", "Venous-phase CT Features", "Other Features"]
    grouped = {group: [] for group in group_order}

    for feat in feature_names:
        group = feature_meta[feat]["group"]
        if group in grouped:
            grouped[group].append(feat)

    values: dict[str, float] = {}
    for group_name in group_order:
        feats = grouped.get(group_name, [])
        if not feats:
            continue

        with st.expander(f"{group_name} ({len(feats)} features)", expanded=False):
            columns = st.columns(3)
            for idx, feat in enumerate(feats):
                with columns[idx % 3]:
                    values[feat] = st.number_input(
                        label=format_widget_label(feat),
                        value=0.0,
                        format="%.6f",
                        key=f"input_{feat}",
                    )
    return values


def make_interpretation_text(result: PredictionResult) -> str:
    comparator = "above" if result.positive_proba >= CFG.fixed_threshold else "below"
    return (
        f"The estimated probability of response is {result.positive_proba:.3f}, "
        f"which is {comparator} the threshold ({CFG.fixed_threshold:.6f}). "
        f"The model classifies this patient as a {result.predicted_label}."
    )


def render_header(total_features: int) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin-bottom:0.35rem;">{CFG.app_title}</h1>
            <div style="font-size:1.02rem; color:#4f647a; line-height:1.5;">
                {CFG.app_subtitle}<br>
                <b>Deployed model:</b> {CFG.model_alias} &nbsp;|&nbsp;
                <b>Classifier:</b> {CFG.target_model_name} &nbsp;|&nbsp;
                <b>Total Features:</b> {total_features}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="note-card">'
            f'The deployment uses a fixed decision threshold of <b>{CFG.fixed_threshold:.6f}</b>. '
            "Raw input values will be transformed by the saved Z-score scaler before prediction and SHAP analysis."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_preview(raw_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    preview_df = pd.DataFrame({
        "Raw Input": raw_df.iloc[0],
        "Standardized Input": model_df.iloc[0],
    })
    with st.expander("Current Input Table", expanded=True):
        st.dataframe(preview_df, use_container_width=True)


def render_prediction_summary(result: PredictionResult) -> None:
    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Response Probability", f"{result.positive_proba * 100:.1f}%")
    with col2:
        render_metric_card("Predicted Category", result.predicted_label)
    with col3:
        render_metric_card("Non-response Probability", f"{result.negative_proba * 100:.1f}%")

    st.progress(int(round(result.positive_proba * 100)))
    render_result_banner(make_interpretation_text(result), positive=(result.predicted_class == 1))


def render_probability_section(result: PredictionResult) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Probability Visualization")
    st.pyplot(plot_probability_bar(result.positive_proba), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_shap_section(bundle: ExplanationBundle, result: PredictionResult, total_features: int) -> None:
    st.subheader("Model Explanation (SHAP)")
    tab1, tab2, tab3 = st.tabs(["SHAP Guided Force Plot", "Waterfall Plot", "Features Table"])

    with tab1:
        st.caption(
            f"Custom force-style SHAP plot with leader lines and indexed labels "
            f"(top {min(CFG.force_max_display, total_features)} features by absolute SHAP value)."
        )
        try:
            fig_force = plot_guided_force_like(
                explanation=bundle.force,
                prediction_value=result.positive_proba,
                base_value=float(bundle.full.base_values),
            )
            st.pyplot(fig_force, use_container_width=True)
        except Exception as exc:
            st.error(f"SHAP guided force plot rendering failed: {exc}")

    with tab2:
        st.caption("Waterfall plot using original feature names.")
        try:
            st.pyplot(plot_waterfall(bundle.waterfall, total_features=total_features), use_container_width=True)
        except Exception as exc:
            st.error(f"SHAP waterfall plot rendering failed: {exc}")

    with tab3:
        st.caption("Features ranked by absolute SHAP magnitude.")
        try:
            display_df = bundle.table.copy()
            for col in ("Raw Input", "Standardized Input", "SHAP Value", "Absolute SHAP"):
                if col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
            st.dataframe(display_df, use_container_width=True)
        except Exception as exc:
            st.error(f"SHAP feature table rendering failed: {exc}")


def main() -> None:
    stop_if_missing([CFG.model_path, CFG.x_train_path, CFG.y_train_path, CFG.scaler_path])

    assets = load_assets()
    explainer = build_explainer(assets.model, assets.background)
    total_features = len(assets.feature_names)

    render_header(total_features)
    render_sidebar(total_features)

    st.subheader("Patient Feature Input")
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown(
            (
                '<div class="small-muted">'
                "Enter raw patient-specific radiomics feature values below. "
                "The app will automatically apply the training-time Z-score transformation before prediction."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        user_inputs = make_input_widgets(assets.feature_meta, assets.feature_names)

        col1, col2, _ = st.columns([1.2, 1.2, 3.6])
        with col1:
            submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)
        with col2:
            preview = st.form_submit_button("Preview Inputs", use_container_width=True)

    input_df_raw = build_input_data(user_inputs, assets.feature_names)
    input_df_model = transform_input_with_scaler(input_df_raw, assets.scaler, assets.feature_names)

    if preview and not submitted:
        render_preview(input_df_raw, input_df_model)

    if not submitted:
        st.markdown('<div class="footer-note">Research-use interface for the multimodal model.</div>', unsafe_allow_html=True)
        return

    result = run_prediction(assets.model, input_df_model)
    render_prediction_summary(result)
    render_probability_section(result)

    try:
        with st.spinner("Computing SHAP explanation..."):
            bundle = prepare_explanations(explainer, input_df_raw, input_df_model, total_features)
        render_shap_section(bundle, result, total_features)
    except Exception as exc:
        st.error(f"SHAP explanation failed: {exc}")

    st.markdown('<div class="footer-note">Research-use interface for the multimodal model.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
