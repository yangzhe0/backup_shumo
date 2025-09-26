# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

# 1重采样_CNN_fix.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import resample_poly
from tqdm import tqdm

# 输入/输出路径配置
SRC_DIR     = Path("第1问/0csv无处理")
DST_DIR     = Path("第1问/1.1csv_32k")
NPY_DIR     = Path("第1问/1.1npy_32k")
DETAIL_CSV  = Path("第1问/0detail.csv")
OUT_DETAIL  = Path("第1问/1.1detail_32k.csv")

DST_DIR.mkdir(parents=True, exist_ok=True)
NPY_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 32000
EXPECTED_CHANNELS = ["DE", "FE", "BA"]   # 需要保证的通道列表

df_index = pd.read_csv(DETAIL_CSV)
rows = []

for _, row in tqdm(df_index.iterrows(), total=len(df_index)):
    csv_path = Path(row["csv_path"])
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)
    sr = row["sr_hz"]
    if pd.isna(sr):
        continue
    sr = int(sr)

    sigs_resampled = []
    available_channels = []

    # 针对每个目标通道，重采样或补零
    for ch in EXPECTED_CHANNELS:
        cols = [c for c in df.columns if ch in c.upper()]
        if cols:
            sig = df[cols[0]].values
            target_len = int(round(len(sig) * TARGET_SR / sr))

            sig_resampled = resample_poly(sig, TARGET_SR, sr)

            # 保证长度一致：不足补零，多余截断
            if len(sig_resampled) < target_len:
                sig_resampled = np.pad(sig_resampled, (0, target_len - len(sig_resampled)))
            elif len(sig_resampled) > target_len:
                sig_resampled = sig_resampled[:target_len]

            sigs_resampled.append(sig_resampled)
            available_channels.append(ch)
        else:
            # 如果通道缺失，补零信号
            if "target_len" not in locals():
                target_len = int(round(len(df) * TARGET_SR / sr))
            sigs_resampled.append(np.zeros(target_len))

    sigs_resampled = np.stack(sigs_resampled, axis=0)  # [C, T]

    # 保存 CSV（如需兼容旧流程，可取消注释）
    rel_path = csv_path.relative_to(SRC_DIR)
    out_path = DST_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pd.DataFrame(sigs_resampled.T, columns=EXPECTED_CHANNELS).to_csv(out_path, index=False)

    # 保存 NPY（CNN 可直接加载）
    npy_path = NPY_DIR / rel_path.with_suffix(".npy")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, sigs_resampled.astype(np.float32))

    # 更新索引信息
    new_meta = row.to_dict()
    new_meta["sr_hz"] = TARGET_SR
    new_meta["csv_path"] = str(out_path)
    new_meta["npy_path"] = str(npy_path)
    new_meta["channels"] = ",".join(available_channels)
    rows.append(new_meta)

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

print(f"✅ 重采样完成，共 {len(df_out)} 个文件")
print("新索引:", OUT_DETAIL)
print("npy 存放目录:", NPY_DIR)
