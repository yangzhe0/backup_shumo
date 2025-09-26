
"""
改进版 4特征提取.py
- 长窗口（默认 8192）+ 50% 重叠
- 自适应带宽 BW = max(2*Δf, 0.02*f0)
- 谐波聚能（默认 8）
- 包络谱（Hilbert）能量/比值
- 侧带能量（±m*fr）、调制深度与对称性
- 倒谱在 1/f0 处的峰值

使用方式与原脚本一致：读取 data/csv_DE_N_32/*.csv.gz，写出 result/features.csv
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re

import numpy as np
import pandas as pd

from numpy.fft import rfft, rfftfreq, irfft
from scipy.signal import hilbert, get_window
from scipy.stats import kurtosis, skew

# =============================
# 全局参数（可按需调整）
# =============================
FS = 32000
WIN = 8192          # 建议 4096~8192；Δf=FS/WIN，WIN=8192时约3.9Hz
STEP = WIN // 2     # 50% 重叠
EPS = 1e-12

# 轴承几何参数（DE=SKF6205）
N_BALLS = 9
# 毫米！！
BALL_DIAM_IN = 7.94
PITCH_DIAM_IN = 39.04
CONTACT_ANGLE_DEG = 0.0

N_HARM = 8          # 谐波个数（k=1..N_HARM）
N_SIDEBANDS = 2     # 侧带阶数（±m*fr, m=1..N_SIDEBANDS）

# =============================
# 工具：路径/文件名解析（兜底）
# =============================
def parse_meta_from_path(path: Path, allow_rpm_from_name: bool = False) -> Dict[str, Optional[str]]:
    d = {"label": None, "size_code": None, "load": None, "clock_pos": None, "rpm": None, "split": None}
    parts = list(path.parts)
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
                     contact_angle_deg: float = CONTACT_ANGLE_DEG) -> Tuple[float, float, float, float]:
    """返回 (fr, BPFO, BPFI, BSF) in Hz"""
    fr = rpm / 60.0
    phi = np.deg2rad(contact_angle_deg)
    d = ball_diam_in
    D = pitch_diam_in
    bpfo = 0.5 * n_balls * fr * (1 - (d / D) * np.cos(phi))
    bpfi = 0.5 * n_balls * fr * (1 + (d / D) * np.cos(phi))
    bsf  = (D / (2.0 * d)) * fr * (1 - ((d / D) * np.cos(phi)) ** 2)
    return fr, bpfo, bpfi, bsf

# =============================
# 频域工具
# =============================
def _windowed_fft(x: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = get_window("hann", len(x), fftbins=True)
    X = rfft(x * w)
    A = np.abs(X)
    A2 = A**2
    F = rfftfreq(len(x), d=1.0/fs)
    return F, A, A2

def spectral_entropy(pxx: np.ndarray) -> float:
    p = pxx / (np.sum(pxx) + EPS)
    return float(-np.sum(p * np.log(p + EPS)))

def delta_f(fs: int, win: int) -> float:
    return fs / float(win)

def adaptive_bw(f0: float, fs: int, win: int, rel: float = 0.02, min_bins: int = 2) -> float:
    """BW = max(min_bins * Δf, rel * f0)"""
    df = delta_f(fs, win)
    return max(min_bins * df, rel * max(f0, df))

def band_energy(F: np.ndarray, A2: np.ndarray, center: float, bw: float) -> float:
    if center <= 0.0 or bw <= 0.0:
        return 0.0
    m = (F >= center - bw) & (F <= center + bw)
    if not np.any(m):
        return 0.0
    return float(np.sum(A2[m]))

def harmonic_band_energy(F: np.ndarray, A2: np.ndarray, f0: float, n_harm: int, bw: float) -> float:
    total = 0.0
    for k in range(1, n_harm + 1):
        total += band_energy(F, A2, k * f0, bw)
    return float(total)

def sideband_energy(F: np.ndarray, A2: np.ndarray,
                    f0: float, fr: float, bw: float,
                    n_side: int = N_SIDEBANDS, n_harm: int = 1) -> float:
    """在 k*f0 ± m*fr 的栅格上累积能量"""
    total = 0.0
    for k in range(1, n_harm + 1):
        carrier = k * f0
        for m in range(1, n_side + 1):
            total += band_energy(F, A2, carrier + m * fr, bw)
            total += band_energy(F, A2, carrier - m * fr, bw)
    return float(total)

def sideband_asymmetry(F: np.ndarray, A2: np.ndarray,
                       f0: float, fr: float, bw: float, k: int = 1, m: int = 1) -> float:
    """单阶对称性：|E(f0+fr)-E(f0-fr)|/(sum+eps)"""
    c = k * f0
    e_plus = band_energy(F, A2, c + m * fr, bw)
    e_minus = band_energy(F, A2, c - m * fr, bw)
    s = e_plus + e_minus + EPS
    return float(abs(e_plus - e_minus) / s)

# =============================
# 包络谱 & 倒谱
# =============================
def envelope_spectrum(x: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = np.abs(hilbert(x))
    return _windowed_fft(env, fs)

def cepstrum_peak(x: np.ndarray, fs: int, target_period: float, tol: float = 0.15) -> float:
    """
    实倒谱（对对数幅谱做 IFFT）并在目标倒频处取峰值。
    tol 为相对容差（±tol*target_period 的窗内取最大）。
    """
    F, A, _ = _windowed_fft(x, fs)
    log_mag = np.log(A + EPS)
    # 零相位补满为对称谱以获得实倒谱（近似处理）
    sym = np.concatenate([log_mag, log_mag[-2:0:-1]])  # 去掉直流与Nyquist重复项
    ceps = np.abs(irfft(sym))
    quef = np.arange(len(ceps)) / float(fs)  # 秒
    if target_period <= 0:
        return 0.0
    window = (quef >= target_period * (1 - tol)) & (quef <= target_period * (1 + tol))
    if not np.any(window):
        return 0.0
    return float(np.max(ceps[window]))

# =============================
# 单窗口特征
# =============================
def extract_window_features(sig: np.ndarray, rpm: Optional[float], fs: int = FS, win_len: int = WIN) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    # ---- 时域 ----
    x = sig.astype(float)
    feats["mean"] = float(np.mean(x))
    feats["std"] = float(np.std(x))
    feats["rms"] = float(np.sqrt(np.mean(x ** 2) + EPS))
    feats["kurtosis"] = float(kurtosis(x, fisher=True, bias=False))
    feats["skewness"] = float(skew(x, bias=False))
    feats["crest_factor"] = float((np.max(np.abs(x)) + EPS) / (feats["rms"] + EPS))
    feats["zcr"] = float(np.mean(np.diff(np.signbit(x)) != 0))
    feats["p2p"] = float(np.ptp(x))
    feats["mav"] = float(np.mean(np.abs(x)))

    absx = np.abs(x) + EPS
    feats["shape_factor"]  = float(feats["rms"] / (np.mean(absx)))
    feats["impulse_factor"] = float(np.max(absx) / (np.mean(absx)))
    feats["margin_factor"]  = float(np.max(absx) / ((np.mean(np.sqrt(absx))) ** 2 + EPS))

    # ---- 频域（原始谱）----
    F, A, A2 = _windowed_fft(x, fs)
    feats["spec_energy"] = float(np.sum(A2))
    if np.any(A > 0):
        feats["main_freq"] = float(F[int(np.argmax(A))])
    else:
        feats["main_freq"] = 0.0
    feats["spec_centroid"] = float(np.sum(F * A) / (np.sum(A) + EPS))
    csum = np.cumsum(A2)
    idx95 = int(np.searchsorted(csum, 0.95 * csum[-1]))
    feats["spec_rolloff95"] = float(F[min(idx95, len(F) - 1)])
    feats["spec_entropy"] = float(spectral_entropy(A2 + EPS))
    ps = float(np.sum(A2) + EPS)
    fc = float(np.sum(F * A2) / ps)
    feats["rms_freq"] = float(np.sqrt(np.sum((F ** 2) * A2) / ps))
    feats["freq_std"] = float(np.sqrt(np.sum(((F - fc) ** 2) * A2) / ps))

    # ---- 包络谱 ----
    Fe, Ae, Ae2 = envelope_spectrum(x, fs)
    feats["env_spec_energy"] = float(np.sum(Ae2))
    feats["env_spec_entropy"] = float(spectral_entropy(Ae2 + EPS))

    # ---- 机理相关：特征频率/侧带/倒谱 ----
    if rpm is not None and rpm > 0:
        fr, bpfo, bpfi, bsf = bearing_freqs_hz(rpm)

        # 自适应带宽（按频点分别算）
        bw_bpfo = adaptive_bw(bpfo, fs, win_len)
        bw_bpfi = adaptive_bw(bpfi, fs, win_len)
        bw_bsf  = adaptive_bw(bsf,  fs, win_len)

        total_e = feats["spec_energy"] + EPS
        total_ee = feats["env_spec_energy"] + EPS

        # —— 原始谱：谐波聚能 + 比值
        bpfo_e = harmonic_band_energy(F, A2, bpfo, N_HARM, bw_bpfo)
        bpfi_e = harmonic_band_energy(F, A2, bpfi, N_HARM, bw_bpfi)
        bsf_e  = harmonic_band_energy(F, A2, bsf,  N_HARM, bw_bsf)
        feats["BPFO_energy"] = float(bpfo_e); feats["BPFO_ratio"] = float(bpfo_e / total_e)
        feats["BPFI_energy"] = float(bpfi_e); feats["BPFI_ratio"] = float(bpfi_e / total_e)
        feats["BSF_energy"]  = float(bsf_e ); feats["BSF_ratio"]  = float(bsf_e  / total_e)

        # —— 包络谱：谐波聚能 + 比值（对冲击更敏感）
        bpfo_ee = harmonic_band_energy(Fe, Ae2, bpfo, N_HARM, bw_bpfo)
        bpfi_ee = harmonic_band_energy(Fe, Ae2, bpfi, N_HARM, bw_bpfi)
        bsf_ee  = harmonic_band_energy(Fe, Ae2, bsf,  N_HARM, bw_bsf)
        feats["BPFO_env_ratio"] = float(bpfo_ee / total_ee)
        feats["BPFI_env_ratio"] = float(bpfi_ee / total_ee)
        feats["BSF_env_ratio"]  = float(bsf_ee  / total_ee)

        # —— 侧带能量（包络谱）：±m*fr
        sb_bpfi = sideband_energy(Fe, Ae2, bpfi, fr, bw_bpfi, n_side=N_SIDEBANDS, n_harm=1)
        sb_bsf  = sideband_energy(Fe, Ae2, bsf,  fr, bw_bsf,  n_side=N_SIDEBANDS, n_harm=1)
        feats["BPFI_SB_AMdepth"] = float(sb_bpfi / (band_energy(Fe, Ae2, bpfi, bw_bpfi) + EPS))
        feats["BSF_SB_AMdepth"]  = float(sb_bsf  / (band_energy(Fe, Ae2, bsf,  bw_bsf ) + EPS))

        # —— 侧带对称性（m=1, k=1）
        feats["BPFI_SB_sym"] = sideband_asymmetry(Fe, Ae2, bpfi, fr, bw_bpfi, k=1, m=1)
        feats["BSF_SB_sym"]  = sideband_asymmetry(Fe, Ae2, bsf,  fr, bw_bsf,  k=1, m=1)

        # —— 倒谱峰（等间隔冲击的周期峰）
        feats["cep_BPFO"] = cepstrum_peak(x, fs, 1.0 / max(bpfo, EPS))
        feats["cep_BPFI"] = cepstrum_peak(x, fs, 1.0 / max(bpfi, EPS))
        feats["cep_BSF"]  = cepstrum_peak(x, fs, 1.0 / max(bsf,  EPS))
    else:
        # rpm 不可用 -> 与 rpm 相关的特征置 NaN
        for k in ("BPFO_energy","BPFI_energy","BSF_energy",
                  "BPFO_ratio","BPFI_ratio","BSF_ratio",
                  "BPFO_env_ratio","BPFI_env_ratio","BSF_env_ratio",
                  "BPFI_SB_AMdepth","BSF_SB_AMdepth","BPFI_SB_sym","BSF_SB_sym",
                  "cep_BPFO","cep_BPFI","cep_BSF"):
            feats[k] = np.nan

    return feats

# =============================
# 滑窗
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

    label = rpm = load = size_code = None
    pos = "DE"

    if detail_df is not None:
        base = file.stem.replace(".csv", "")
        hit = detail_df[detail_df["file"].str.contains(base, na=False, regex=False)]
        if not hit.empty:
            hit_row = hit.iloc[0]
            label = hit_row.get("label")
            if "rpm" in hit_row and pd.notna(hit_row["rpm"]):
                rpm = float(hit_row["rpm"])
            load = hit_row.get("load"); size_code = hit_row.get("size_code")

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

    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [label, rpm, load, size_code]):
        meta = parse_meta_from_path(file)
        label = label or meta.get("label")
        load = load or meta.get("load")
        size_code = size_code or meta.get("size_code")
        if rpm is None and meta.get("rpm"):
            rpm = float(meta["rpm"])

    if "sig" not in df.columns:
        raise ValueError(f"{file} 缺少列 'sig'，无法提取特征。")
    sig = df["sig"].values.astype(float)

    rows: List[Dict] = []
    for w in sliding_windows(sig, WIN, STEP):
        feats = extract_window_features(w, rpm, FS, WIN)
        feats.update({
            "label": label, "rpm": rpm, "load": load, "size_code": size_code,
            "pos": pos, "file": file.name
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

    # 各类别关键特征均值（便于快速 sanity check）
    cols_show = [
        "rms","kurtosis","spec_entropy",
        "BPFO_ratio","BPFI_ratio","BSF_ratio",
        "BPFO_env_ratio","BPFI_env_ratio","BSF_env_ratio",
        "BPFI_SB_AMdepth","BSF_SB_AMdepth",
        "cep_BPFO","cep_BPFI","cep_BSF"
    ]
    keep = [c for c in cols_show if c in feats.columns]
    if keep:
        gmean = feats.groupby("label")[keep].mean().reset_index()
        mean_csv = str(Path(out_csv).with_name("feature_means_by_label.csv"))
        gmean.to_csv(mean_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 各类别特征均值保存：{mean_csv}")

# =============================
# CLI
# =============================
if __name__ == "__main__":
    data_dir   = r"data/csv_DE_N_32"   # 你的导出目录
    out_csv    = r"result/features.csv"
    detail_csv = r"result/detail_32.csv"
    run(data_dir, out_csv, detail_csv)
