import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
# =============================
# 参数（可按需调整）
# =============================
FS = 32000
WIN = 2048
STEP = 1024
EPS = 1e-12

# SKF6205（驱动端 DE）的轴承几何参数（题目附件给出）
# n=9; d=0.3126 inch; D=1.537 inch; 角度假设≈0°
N_BALLS = 9
BALL_DIAM_IN = 0.3126
PITCH_DIAM_IN = 1.537
CONTACT_ANGLE_DEG = 0.0

# 频带能量提取时，中心±BW_HZ 的矩形带宽
BW_HZ = 15.0
# 提取谐波个数
N_HARM = 6


# =============================
# 工具函数：路径/文件名解析兜底
# =============================
def parse_meta_from_path(path: Path, allow_rpm_from_name: bool = False) -> Dict[str, Optional[str]]:
    d = {"label": None, "size_code": None, "load": None, "clock_pos": None, "rpm": None, "split": None}
    parts = [p for p in path.parts]
    name = path.stem

    for lab in ("B", "IR", "OR", "N"):
        if lab in parts:
            d["label"] = lab
            break

    for s in ("12kHz_DE_data", "48kHz_DE_data", "48kHz_Normal_data"):
        if s in parts:
            d["split"] = s
            break

    m_size = re.search(r"[\\/](\d{4})[\\/]", str(path))
    if m_size:
        d["size_code"] = m_size.group(1)

    m_load = re.search(r"_(\d)(?:[_\.\)]|$)", name)
    if m_load:
        d["load"] = m_load.group(1)

    m_clock = re.search(r"@(\d{1,2})", name)
    if m_clock:
        d["clock_pos"] = m_clock.group(1)

    # ❌ 默认不再从文件名取 rpm，避免把原始 1797/1750 带回来
    if allow_rpm_from_name:
        m_rpm = re.search(r"(\d+)\s*rpm", name, flags=re.I)
        if m_rpm:
            d["rpm"] = m_rpm.group(1)

    return d

# =============================
# 轴承特征频率（Hz）
# =============================
def bearing_freqs_hz(rpm: float,
                     n_balls: int = N_BALLS,
                     ball_diam_in: float = BALL_DIAM_IN,
                     pitch_diam_in: float = PITCH_DIAM_IN,
                     contact_angle_deg: float = CONTACT_ANGLE_DEG) -> Tuple[float, float, float]:
    """返回 (BPFO, BPFI, BSF) in Hz"""
    fr = rpm / 60.0
    phi = np.deg2rad(contact_angle_deg)
    d = ball_diam_in
    D = pitch_diam_in
    bpfo = 0.5 * n_balls * fr * (1 - (d / D) * np.cos(phi))
    bpfi = 0.5 * n_balls * fr * (1 + (d / D) * np.cos(phi))
    bsf = (D / (2.0 * d)) * fr * (1 - ((d / D) * np.cos(phi)) ** 2)
    return bpfo, bpfi, bsf


# =============================
# 频域特征 & 频带能量
# =============================
def spectral_entropy(pxx: np.ndarray) -> float:
    p = pxx / (np.sum(pxx) + EPS)
    return -np.sum(p * np.log(p + EPS))

def band_energy(fft_freqs: np.ndarray, fft_amp2: np.ndarray, center_hz: float, bw_hz: float) -> float:
    mask = (fft_freqs >= center_hz - bw_hz) & (fft_freqs <= center_hz + bw_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(fft_amp2[mask]))

def harmonic_band_energy(fft_freqs: np.ndarray, fft_amp2: np.ndarray, f0: float, n_harm: int, bw_hz: float) -> float:
    total = 0.0
    for k in range(1, n_harm + 1):
        total += band_energy(fft_freqs, fft_amp2, k * f0, bw_hz)
    return total
    # feats["BPFO_energy"] = harmonic_band_energy(F, A2, bpfo, N_HARM, BW_HZ) 

# =============================
# 单窗口特征
# =============================
def extract_window_features(sig: np.ndarray, rpm: Optional[float], fs: int = FS) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    # ---- 时域 ----
    x = sig.astype(float)
    feats["mean"] = np.mean(x)
    feats["std"] = np.std(x)
    feats["rms"] = float(np.sqrt(np.mean(x ** 2)))
    feats["kurtosis"] = float(kurtosis(x, fisher=True, bias=False))
    feats["skewness"] = float(skew(x, bias=False))
    feats["crest_factor"] = float(np.max(np.abs(x)) / (feats["rms"] + EPS))
    # 零交叉率
    feats["zcr"] = float(np.mean(np.diff(np.signbit(x)) != 0))
    # 峰峰值 & 平均绝对值（文档里的 p2p, MAV）
    feats["p2p"] = float(np.ptp(x))
    feats["mav"] = float(np.mean(np.abs(x)))


    # 波形/脉冲/裕度因子
    absx = np.abs(x) + EPS
    feats["shape_factor"] = float(feats["rms"] / (np.mean(absx)))
    feats["impulse_factor"] = float(np.max(absx) / (np.mean(absx)))
    feats["margin_factor"] = float(np.max(absx) / ((np.mean(np.sqrt(absx))) ** 2 + EPS))


    # ---- 频域 ----
    # 单边幅谱
    X = rfft(x * np.hanning(len(x)))
    A = np.abs(X)
    A2 = A ** 2
    F = rfftfreq(len(x), d=1.0 / fs)

    # 基础谱特征
    feats["spec_energy"] = float(np.sum(A2))
    if np.any(A > 0):
        idx_max = int(np.argmax(A))
        feats["main_freq"] = float(F[idx_max])
    else:
        feats["main_freq"] = 0.0
    feats["spec_centroid"] = float(np.sum(F * A) / (np.sum(A) + EPS))
    # 95% roll-off
    cumsum = np.cumsum(A2)
    cutoff_idx = int(np.searchsorted(cumsum, 0.95 * cumsum[-1]))
    feats["spec_rolloff95"] = float(F[min(cutoff_idx, len(F) - 1)])
    # 谱熵
    feats["spec_entropy"] = float(spectral_entropy(A2 + EPS))
    # RMS Frequency 与 频率标准差（Root Variance Frequency）
    psum = float(np.sum(A2) + EPS)
    fc = float(np.sum(F * A2) / psum)   # 用功率加权的质心（与上面的幅度质心不同）
    feats["rms_freq"] = float(np.sqrt(np.sum((F**2) * A2) / psum))
    feats["freq_std"] = float(np.sqrt(np.sum(((F - fc)**2) * A2) / psum))

    # ---- 特征频率能量（依赖 rpm）----
    if rpm is not None and rpm > 0:
        bpfo, bpfi, bsf = bearing_freqs_hz(rpm)
        total_e = feats["spec_energy"] + EPS

        feats["BPFO_energy"] = harmonic_band_energy(F, A2, bpfo, N_HARM, BW_HZ)
        feats["BPFI_energy"] = harmonic_band_energy(F, A2, bpfi, N_HARM, BW_HZ)
        feats["BSF_energy"]  = harmonic_band_energy(F, A2, bsf,  N_HARM, BW_HZ)

        feats["BPFO_ratio"] = feats["BPFO_energy"] / total_e
        feats["BPFI_ratio"] = feats["BPFI_energy"] / total_e
        feats["BSF_ratio"]  = feats["BSF_energy"]  / total_e
    else:
        for k in ("BPFO_energy", "BPFI_energy", "BSF_energy", "BPFO_ratio", "BPFI_ratio", "BSF_ratio"):
            feats[k] = np.nan

    return feats


# =============================
# 窗口切分
# =============================
def sliding_windows(sig: np.ndarray, win: int = WIN, step: int = STEP):
    n = len(sig)
    for s in range(0, n - win + 1, step):
        yield sig[s:s + win]


# =============================
# 处理单文件
# =============================
def process_file(file: Path, detail_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = pd.read_csv(file, compression="gzip")

    # ---- 初始化元数据变量 ----
    label, rpm, load, size_code = None, None, None, None
    # 从原始代码看，pos 似乎总是 DE
    pos = "DE"

    # ---- 优先级 1: 从 detail.csv 文件查找 ----
    if detail_df is not None:
        base = file.stem.replace(".csv", "")  # 兼容 .csv.gz
        # hit = detail_df[detail_df["file"].str.contains(base, na=False)]
        hit = detail_df[detail_df["file"].str.contains(base, na=False, regex=False)]
        if not hit.empty:
            hit_row = hit.iloc[0]
            label = hit_row.get("label")
            # 确保 rpm 列存在且值不为空
            if "rpm" in hit_row and pd.notna(hit_row["rpm"]):
                rpm = float(hit_row["rpm"])
            load = hit_row.get("load")
            size_code = hit_row.get("size_code")

    # ---- 优先级 2: 从信号文件内部列查找（作为兜底）----
    if label is None and "label" in df.columns:
        label = df["label"].iloc[0]
    if rpm is None and "rpm" in df.columns and pd.notna(df["rpm"].iloc[0]):
        rpm = float(df["rpm"].iloc[0])
    if load is None and "load" in df.columns:
        load = df["load"].iloc[0]
    if size_code is None and "size_code" in df.columns:
        size_code = df["size_code"].iloc[0]
    if "pos" in df.columns:
        pos = df["pos"].iloc[0]

    # ---- 优先级 3: 从路径/文件名解析（最后兜底）----
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [label, rpm, load, size_code]):
        meta = parse_meta_from_path(file)
        label = label or meta.get("label")
        load = load or meta.get("load")
        size_code = size_code or meta.get("size_code")
        if rpm is None and meta.get("rpm"):
            rpm = float(meta["rpm"])

    # 读取信号序列
    if "sig" not in df.columns:
        raise ValueError(f"{file} 缺少列 'sig'，无法提取特征。")
    sig = df["sig"].values.astype(float)

    rows: List[Dict] = []
    for w in sliding_windows(sig, WIN, STEP):
        feats = extract_window_features(w, rpm)
        feats.update({
            "label": label,
            "rpm": rpm,
            "load": load,
            "size_code": size_code,
            "pos": pos,
            "file": file.name
        })
        rows.append(feats)

    return pd.DataFrame(rows)

# =============================
# 批量处理 + 简要分析
# =============================
def run(data_dir: str, out_csv: str, detail_csv: Optional[str] = None):
    data_dir = Path(data_dir)
    detail_df = None
    if detail_csv and Path(detail_csv).exists():
        detail_df = pd.read_csv(detail_csv)

    all_df = []
    for f in data_dir.rglob("*.csv.gz"):
        print(f"[+] {f}")
        try:
            df_feats = process_file(f, detail_df)
            all_df.append(df_feats)
        except Exception as e:
            print(f"[!] 跳过 {f}: {e}")

    if not all_df:
        raise RuntimeError("没有生成任何特征，请检查数据目录或文件格式。")

    feats = pd.concat(all_df, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 特征表已保存：{out_csv}")

    # ---- 简要分析输出 ----
    # 1) 各类别特征均值
    cols_show = ["rms", "kurtosis", "spec_entropy", "BPFO_ratio", "BPFI_ratio", "BSF_ratio"]
    gmean = feats.groupby("label")[cols_show].mean().reset_index()
    mean_csv = str(Path(out_csv).with_name("feature_means_by_label.csv"))
    gmean.to_csv(mean_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 各类别特征均值保存：{mean_csv}")

 

# =============================
# CLI
# =============================
if __name__ == "__main__":
    # ==== 路径写死在这里 ====
    data_dir   = r"data/csv_DE_N_32"
    out_csv    = r"result/features.csv"
    detail_csv = r"result/detail_32.csv"

    run(data_dir, out_csv, detail_csv)
