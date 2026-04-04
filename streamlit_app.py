import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

FEATURE_CSV = r"E:\pixelmedai\function_pm\run\ICC0.75\radDLRMmax5mm.csv"
LABEL_CSV = r"E:\pixelmedai\function_pm\run\137\label.csv"

SCALER_PATH = r"E:\pixelmedai\function_pm\run\streamlit_assets\zscore_scaler.pkl"
OUT_DIR = r"E:\pixelmedai\function_pm\run\streamlit_assets"
os.makedirs(OUT_DIR, exist_ok=True)

X_TRAIN_OUT = os.path.join(OUT_DIR, "x_train.csv")

def clean_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"V\.nii\.gz$", "", regex=True)
        .str.replace(r"\.nii\.gz$", "", regex=True)
    )

def normalize_sep_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[\s]+", "", name)
    name = re.sub(r"[-_]+", "", name)
    return name

def align_dataframe_columns_by_target_names(df, target_feature_names):
    current_cols = list(df.columns)
    norm_to_current = {}
    for c in current_cols:
        norm_to_current.setdefault(normalize_sep_name(c), []).append(c)

    selected_cols = []
    missing = []
    ambiguous = []

    for name in target_feature_names:
        if name in df.columns:
            selected_cols.append(name)
            continue

        norm_name = normalize_sep_name(name)
        candidates = norm_to_current.get(norm_name, [])

        if len(candidates) == 1:
            selected_cols.append(candidates[0])
        elif len(candidates) == 0:
            missing.append(name)
        else:
            ambiguous.append((name, candidates))

    if missing:
        raise ValueError("缺少 scaler 所需特征：\n" + "\n".join(map(str, missing)))

    if ambiguous:
        msg = ["{} --> {}".format(k, v) for k, v in ambiguous]
        raise ValueError("特征多重匹配歧义：\n" + "\n".join(msg))

    out = df[selected_cols].copy()
    out.columns = list(target_feature_names)
    return out

df_feature = pd.read_csv(FEATURE_CSV)
df_label = pd.read_csv(LABEL_CSV)

df_feature["ID"] = clean_id(df_feature["ID"])

if "IDV.nii.gz" in df_label.columns and "ID" not in df_label.columns:
    df_label = df_label.rename(columns={"IDV.nii.gz": "ID"})
df_label["ID"] = clean_id(df_label["ID"])

merged_data = df_feature.merge(df_label, on="ID", how="inner")

num_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[num_cols] = merged_data[num_cols].apply(lambda s: s.fillna(s.mean()))

merged_data.columns = (
    merged_data.columns
    .str.replace(r"[-(),\s]+", "_", regex=True)
    .str.strip("_")
)

feature_cols = merged_data.columns[1:-2]
X_feat = merged_data.loc[:, feature_cols]

selector = VarianceThreshold(threshold=0)
selector.fit(X_feat)

kept_features_mask = selector.get_support()
kept_features = X_feat.columns[kept_features_mask]
dropped_features = [col for col in X_feat.columns if col not in kept_features]
merged_data.drop(columns=dropped_features, inplace=True)

if "group" not in merged_data.columns:
    raise ValueError("merged_data 中未找到 group 列，无法导出训练集 x_train.csv。")

train_mask = merged_data["group"].astype(str).str.lower().eq("train")
if train_mask.sum() == 0:
    raise ValueError("没有找到 group == 'train' 的样本。")

final_feature_cols = merged_data.columns[1:-2]
x_train_raw = merged_data.loc[train_mask, final_feature_cols].copy()

scaler = joblib.load(SCALER_PATH)

if not hasattr(scaler, "feature_names_in_"):
    raise ValueError("当前 scaler 不含 feature_names_in_，无法严格对齐。")

scaler_feature_names = list(scaler.feature_names_in_)
x_train_raw = align_dataframe_columns_by_target_names(x_train_raw, scaler_feature_names)

x_train_std = pd.DataFrame(
    scaler.transform(x_train_raw),
    columns=scaler_feature_names,
    index=x_train_raw.index
)

x_train_std.to_csv(X_TRAIN_OUT, index=False, encoding="utf-8-sig")

print("已保存：")
print(X_TRAIN_OUT)
print("x_train.csv shape:", x_train_std.shape)
