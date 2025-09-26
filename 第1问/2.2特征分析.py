# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

"""
特征降维与可视化：类别分布、特征相关性、PCA、t-SNE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

FEAT_PARQUET = Path("第1问/2features/features.parquet")
OUT_DIR = Path("第1问/2特征分析"); OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(FEAT_PARQUET)

non_feat_cols = ["seg_file", "label", "load", "rpm", "size_code", "clock_pos"]
feat_cols = [c for c in df.columns if c not in non_feat_cols]
X = df[feat_cols].values
y = df["label"].astype(str).values

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 类别分布
plt.figure(figsize=(6, 4))
sns.countplot(x="label", hue="label", data=df,
              order=df["label"].value_counts().index,
              palette="Set2", legend=False)
plt.title("Class Distribution")
plt.savefig(OUT_DIR/"class_distribution.png", dpi=150)
plt.close()

# 特征相关性
plt.figure(figsize=(12, 10))
corr = pd.DataFrame(X, columns=feat_cols).corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR/"feature_correlation.png", dpi=150)
plt.close()

# PCA 方差解释率
pca = PCA()
X_pca = pca.fit_transform(X_std)
expl_var = pca.explained_variance_ratio_

plt.figure(figsize=(6, 4))
plt.plot(np.cumsum(expl_var) * 100, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.savefig(OUT_DIR/"pca_variance.png", dpi=150)
plt.close()

# PCA 前两维
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, alpha=0.6, palette="Set1", s=20)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (first 2 components)")
plt.legend()
plt.savefig(OUT_DIR/"pca_scatter.png", dpi=150)
plt.close()

# t-SNE
tsne = TSNE(n_components=2, random_state=42, init="pca",
            learning_rate="auto", perplexity=30)
X_tsne = tsne.fit_transform(X_std)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, alpha=0.6, palette="Set1", s=20)
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.title("t-SNE (2D)")
plt.legend()
plt.savefig(OUT_DIR/"tsne_scatter.png", dpi=150)
plt.close()

# PCA vs t-SNE
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, alpha=0.6,
                palette="Set1", s=15, ax=axs[0])
axs[0].set_title("PCA 2D")
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, alpha=0.6,
                palette="Set1", s=15, ax=axs[1])
axs[1].set_title("t-SNE 2D")
plt.tight_layout()
plt.savefig(OUT_DIR/"pca_tsne_compare.png", dpi=150)
plt.close()

print(f"✅ 可视化完成，图片已保存到 {OUT_DIR}")
