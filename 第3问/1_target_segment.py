# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import pywt

from config import DETAIL_CSV, RAW_DIR, OUT_DIR, FS, L, HOP, LOWC, HIGHC

try:
    from config import CHANNEL_INDEX
except Exception:
    CHANNEL_INDEX = 0
try:
    from config import NORM_PM1
except Exception:
    NORM_PM1 = 0

OUT_PARQUET = OUT_DIR / "target_segments.parquet"
OUT_WAVE    = OUT_DIR / "target_segments_wave"; OUT_WAVE.mkdir(parents=True, exist_ok=True)


def _safe_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    if RAW_DIR and (RAW_DIR / p).exists():
        return RAW_DIR / p
    return p


def bandpass(x, fs, low, high, order=4):
    nyq = fs * 0.5
    hi = min(high, nyq * 0.999)
    lo = max(1e-3, low)
    if hi <= lo:
        return x.astype(float)
    b, a = butter(order, [lo/nyq, hi/nyq], btype="bandpass")
    return filtfilt(b, a, x).astype(float)


def wdenoise(x, wavelet="db4", level=4):
    x = np.asarray(x, dtype=float)
    coeffs = pywt.wavedec(x, wavelet, mode="per")
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745) if len(coeffs[-1]) else 0.0
    uth = sigma * np.sqrt(2 * np.log(len(x))) if sigma > 0 else 0.0
    coeffs = [coeffs[0]] + [pywt.threshold(c, uth, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")[:len(x)]


def norm_pm1(x):
    m = np.max(np.abs(x))
    return x if (not np.isfinite(m) or m == 0) else (x / m)


def read_signal_csv(path: Path):
    df = pd.read_csv(path, sep=None, engine="python")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in list(df.columns):
        if str(c).lower() in ("time", "times", "timestamp"):
            if c in numeric_cols:
                numeric_cols.remove(c)
    if not numeric_cols:
        raise RuntimeError(f"{path.name}: 无数值通道列")
    idx = CHANNEL_INDEX if CHANNEL_INDEX is not None else 0
    idx = int(idx) if (0 <= int(idx) < len(numeric_cols)) else 0
    return df[numeric_cols[idx]].to_numpy(dtype=float, copy=False)


def segment_signal(x, L, hop):
    if len(x) < L:
        return []
    n = (len(x) - L) // hop + 1
    return [(k * hop, k * hop + L) for k in range(n)]


def main():
    detail = pd.read_csv(DETAIL_CSV, sep=None, engine="python")
    if "csv_path" not in detail.columns:
        raise RuntimeError("detail.csv 必须包含列 csv_path")
    rows = []
    for _, r in tqdm(detail.iterrows(), total=len(detail), desc="分段"):
        p = _safe_path(r["csv_path"])
        if not Path(p).exists():
            print(f"⚠️ 文件不存在: {p}")
            continue
        x = read_signal_csv(p)
        x = bandpass(x, FS, LOWC, HIGHC)
        x = wdenoise(x)
        for st, ed in segment_signal(x, L, HOP):
            seg = x[st:ed].astype(np.float32)
            if NORM_PM1:
                seg = norm_pm1(seg)
            seg = seg.reshape(1, -1)
            sf = OUT_WAVE / f"{Path(p).stem}_seg{st//HOP}.npy"
            np.save(sf, seg)
            row = {"seg_file": str(sf), "file": Path(p).stem, "seg_idx": st // HOP}
            if "rpm" in r.index:
                row["rpm"] = r["rpm"]
            rows.append(row)
    pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
    print(f"✅ 分段完成：{len(rows)} 段 → {OUT_PARQUET}")


if __name__ == "__main__":
    main()
