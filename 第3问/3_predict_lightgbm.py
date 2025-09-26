# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的


import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from config import OUT_DIR, ARTifacts, USE_CORAL, RANDOM_SEED

IN_FEAT  = OUT_DIR / "target_features.parquet"
OUT_SEG  = OUT_DIR / "target_segment_preds.csv"
OUT_FILE = OUT_DIR / "target_file_preds.csv"
OUT_PNG  = OUT_DIR / "pca_before_after.png"

def load_artifacts():
    with open(ARTifacts/"feat_cols.json", "r", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(ARTifacts/"classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    scaler = joblib.load(ARTifacts/"scaler.pkl")
    model  = joblib.load(ARTifacts/"model.pkl")
    return feat_cols, classes, scaler, model

def coral_align(Xs, Xt):
    Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-3
    Ct = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1]) * 1e-3
    Es, Us = np.linalg.eigh(Cs); Es[Es < 1e-6] = 1e-6
    Et, Ut = np.linalg.eigh(Ct); Et[Et < 1e-6] = 1e-6
    Cs_inv_sqrt = Us @ np.diag(Es**-0.5) @ Us.T
    Ct_sqrt     = Ut @ np.diag(Et**0.5)  @ Ut.T
    return Xt @ Cs_inv_sqrt @ Ct_sqrt

def pca_plot(X_before, X_after, save_path):
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    A = pca.fit_transform(X_before)
    B = pca.fit_transform(X_after)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.scatter(A[:,0], A[:,1], s=5, alpha=0.5); plt.title("Target PCA (before)")
    plt.subplot(1,2,2); plt.scatter(B[:,0], B[:,1], s=5, alpha=0.5); plt.title("Target PCA (after CORAL)")
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()

def aggregate_file_level(df_seg):
    hard = df_seg.groupby("file")["pred"].agg(lambda s: s.value_counts().idxmax()).to_frame("hard_label")
    proba_cols = [c for c in df_seg.columns if c.startswith("proba_")]
    soft = df_seg.groupby("file")[proba_cols].mean()
    soft["soft_label"] = soft.idxmax(axis=1).str.replace("proba_", "", regex=False)
    return hard.join(soft)

def main():
    feat_cols, classes, scaler_src, model = load_artifacts()
    df = pd.read_parquet(IN_FEAT)

    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = df[c] if c in df.columns else 0.0
    X = X[feat_cols].astype(float).values

    X_std = scaler_src.transform(X)

    if USE_CORAL:
        X_aligned = coral_align(X_std, X_std)
        pca_plot(X_std, X_aligned, OUT_PNG)
        X_use = X_aligned
    else:
        X_use = X_std

    proba = model.predict_proba(X_use)
    y_pred = [classes[i] for i in proba.argmax(axis=1)]

    out = df[["seg_file", "file", "seg_idx"]].copy()
    out["pred"] = y_pred
    for i, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, i]
    out.to_csv(OUT_SEG, index=False, encoding="utf-8-sig")

    agg = aggregate_file_level(out)
    agg.to_csv(OUT_FILE, encoding="utf-8-sig")

    print(f"✅ 迁移诊断完成：段级→{OUT_SEG}，文件级→{OUT_FILE}")
    if USE_CORAL:
        print(f"🖼 生成 PCA 可视化：{OUT_PNG}")

if __name__ == "__main__":
    main()
