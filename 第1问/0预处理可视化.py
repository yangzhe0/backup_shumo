# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DETAIL_CSV = Path("data/0detail.csv")
CSV_DIR    = Path("data/0csv无处理")
out_dir    = Path("data/0analyse")
out_dir.mkdir(parents=True, exist_ok=True)

df_index = pd.read_csv(DETAIL_CSV)


# 图1: 样本数量分布
plt.figure(figsize=(6, 4))
df_index["label"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Distribution of Sample Counts")
plt.xlabel("Category"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(out_dir / "图1_样本数量分布.png", dpi=150)
plt.close()

# 图2: 样本占比
plt.figure(figsize=(6, 6))
df_index["label"].value_counts().plot(
    kind="pie", autopct="%.1f%%",
    colors=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
)
plt.ylabel("")
plt.title("Sample Percentage")
plt.tight_layout()
plt.savefig(out_dir / "图2_样本占比.png", dpi=150)
plt.close()


# 图3: IR 波形示例
row = df_index[df_index["label"] == "IR"].iloc[0]
df = pd.read_csv(row["csv_path"])
plt.figure(figsize=(10, 3))
plt.plot(df.iloc[:2000, 0].values, lw=0.7)
plt.title(f"IR Waveform Example ({row['file']})")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(out_dir / "图3_IR波形.png", dpi=150)
plt.close()

# 图4: OR 波形示例
row = df_index[df_index["label"] == "OR"].iloc[0]
df = pd.read_csv(row["csv_path"])
plt.figure(figsize=(10, 3))
plt.plot(df.iloc[:2000, 0].values, lw=0.7, color="orange")
plt.title(f"OR Waveform Example ({row['file']})")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(out_dir / "图4_OR波形.png", dpi=150)
plt.close()

# 图5: B 波形示例
row = df_index[df_index["label"] == "B"].iloc[0]
df = pd.read_csv(row["csv_path"])
plt.figure(figsize=(10, 3))
plt.plot(df.iloc[:2000, 0].values, lw=0.7, color="green")
plt.title(f"B Waveform Example ({row['file']})")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(out_dir / "图5_B波形.png", dpi=150)
plt.close()


# 图6: IR 不同载荷下的对比
plt.figure(figsize=(10, 3))
sub = df_index[df_index["label"] == "IR"].dropna(subset=["load"])
for load in sorted(sub["load"].unique()):
    row = sub[sub["load"] == load].iloc[0]
    sig = pd.read_csv(row["csv_path"]).iloc[:2000, 0].values
    plt.plot(sig, lw=0.7, label=f"Load {load}")
plt.title("IR - Different Loads")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "图6_IR不同载荷.png", dpi=150)
plt.close()

# 图7: OR 不同载荷下的对比
plt.figure(figsize=(10, 3))
sub = df_index[df_index["label"] == "OR"].dropna(subset=["load"])
for load in sorted(sub["load"].unique()):
    row = sub[sub["load"] == load].iloc[0]
    sig = pd.read_csv(row["csv_path"]).iloc[:2000, 0].values
    plt.plot(sig, lw=0.7, label=f"Load {load}")
plt.title("OR - Different Loads")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "图7_OR不同载荷.png", dpi=150)
plt.close()

# 图8: B 不同载荷下的对比
plt.figure(figsize=(10, 3))
sub = df_index[df_index["label"] == "B"].dropna(subset=["load"])
for load in sorted(sub["load"].unique()):
    row = sub[sub["load"] == load].iloc[0]
    sig = pd.read_csv(row["csv_path"]).iloc[:2000, 0].values
    plt.plot(sig, lw=0.7, label=f"Load {load}")
plt.title("B - Different Loads")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "图8_B不同载荷.png", dpi=150)
plt.close()


# 图9: 正常 vs 故障对比
plt.figure(figsize=(10, 3))
for lab, color in zip(["N", "IR", "OR", "B"], ["black", "red", "orange", "green"]):
    sub = df_index[df_index["label"] == lab]
    if not sub.empty:
        row = sub.iloc[0]
        sig = pd.read_csv(row["csv_path"]).iloc[:2000, 0].values
        plt.plot(sig, lw=0.7, label=lab, color=color)
plt.title("Normal vs Fault")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "图9_正常vs故障.png", dpi=150)
plt.close()

# 图10: 相同载荷下不同故障对比 (以 load=0 为例)
plt.figure(figsize=(10, 3))
sub = df_index.dropna(subset=["load"])
same_load = sub[sub["load"] == 0]
for lab, color in zip(["IR", "OR", "B"], ["red", "orange", "green"]):
    sub_lab = same_load[same_load["label"] == lab]
    if not sub_lab.empty:
        row = sub_lab.iloc[0]
        sig = pd.read_csv(row["csv_path"]).iloc[:2000, 0].values
        plt.plot(sig, lw=0.7, label=lab, color=color)
plt.title("Different Faults Under Same Load (Load=0)")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "图10_同载荷不同故障.png", dpi=150)
plt.close()

print("✅ 所有 10 张图已生成:", out_dir)
