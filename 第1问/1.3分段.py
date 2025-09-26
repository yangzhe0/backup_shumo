# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import pywt

DETAIL_CSV = Path("第1问/1.2detail_32k_DE.csv")
SEGMENTS_PARQUET = Path("第1问/1.3segments.parquet")
SEG_WAVE_DIR = Path("第1问/1.3segments_wave")

fs = 32000
L, hop = 4096, 2048
lowcut, highcut = 100, min(5000, int(0.45 * fs))

SEG_WAVE_DIR.mkdir(parents=True, exist_ok=True)


def bandpass_filter(x, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, x)


def wavelet_denoise(x, wavelet="db4", level=4):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeffs = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


def normalize(x):
    if np.max(x) == np.min(x):
        return np.zeros_like(x)
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1


df_index = pd.read_csv(DETAIL_CSV)
rows = []

for _, row in tqdm(df_index.iterrows(), total=len(df_index)):
    csv_path = Path(row["csv_path"])
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)
    if df.empty:
        continue

    sig = df.iloc[:, 0].values.astype(float)
    num_segs = (len(sig) - L) // hop + 1

    for seg_idx in range(num_segs):
        start, end = seg_idx * hop, seg_idx * hop + L
        x = sig[start:end]
        if len(x) < L:
            continue

        x = bandpass_filter(x, fs, lowcut, highcut)
        x = wavelet_denoise(x)
        x = normalize(x)

        rel = csv_path.relative_to(DETAIL_CSV.parent.parent)
        safe_rel = "_".join(rel.parts).replace(".csv", "")
        seg_file = SEG_WAVE_DIR / f"{safe_rel}_seg{seg_idx}.npy"
        np.save(seg_file, x.astype(np.float32))

        rows.append({
            "src_csv": str(csv_path),
            "seg_file": str(seg_file),
            "seg_idx": seg_idx,
            "start": start,
            "end": end,
            "label": row.get("label"),
            "load": row.get("load"),
            "rpm": 600,
            "size_code": row.get("size_code"),
            "clock_pos": row.get("clock_pos"),
        })

df_segments = pd.DataFrame(rows)
df_segments.to_parquet(SEGMENTS_PARQUET, index=False)

print(f"✅ 已生成 {len(df_segments)} 个段")
print(f"索引表: {SEGMENTS_PARQUET}")
print(f"波形段目录: {SEG_WAVE_DIR}")
