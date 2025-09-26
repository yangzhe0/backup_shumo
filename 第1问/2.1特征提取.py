# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

"""
从分段波形提取特征：
时域、频域、包络谱特征、小波包能量与熵
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch, hilbert
from scipy.stats import kurtosis, skew, entropy
import pywt
from tqdm import tqdm

SEGMENTS_PARQUET = Path("第1问/1.3segments.parquet")
SEG_WAVE_DIR     = Path("第1问/1.3segments_wave")
OUT_DIR          = Path("第1问/2features"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET      = OUT_DIR / "features.parquet"

fs = 32000
BEARING = dict(n=9, d=0.3126, D=1.537)   # SKF6205 参数


def safe_entropy(x, bins=64):
    h, _ = np.histogram(x, bins=bins, density=True)
    return float(entropy(h + 1e-12))


def time_features(x):
    x = np.asarray(x, dtype=float).ravel()   # ★ flatten
    x_abs = np.abs(x)
    rms = np.sqrt(np.mean(x**2) + 1e-20)
    mean_abs = np.mean(x_abs) + 1e-20
    sqrt_abs_mean = np.mean(np.sqrt(x_abs)) + 1e-20
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)),
        rms=float(rms),
        skew=float(skew(x, bias=False)),
        kurtosis=float(kurtosis(x, fisher=True, bias=False)),
        p2p=float(np.max(x) - np.min(x)),
        crest_factor=float(np.max(x_abs) / rms),
        shape_factor=float(rms / mean_abs),
        impulse_factor=float(np.max(x_abs) / mean_abs),
        clearance_factor=float(np.max(x_abs) / (sqrt_abs_mean**2)),
        sig_entropy=float(safe_entropy(x, bins=64)),
    )

def spectral_features(x, sr):
    x = np.asarray(x, dtype=float).ravel()   # ★ flatten
    # ★ 给 nperseg 一个下限，避免 1 段的崩溃
    nper = min(len(x), 4096)
    nper = max(nper, 256)
    # 可选：提高分辨率更稳
    nfft = max(16384, 1<<int(np.ceil(np.log2(nper))))
    f, Pxx = welch(x, fs=sr, nperseg=nper, noverlap=nper//2, nfft=nfft, detrend="constant")
    Pxx = Pxx + 1e-20
    Psum = np.trapz(Pxx, f)
    p_norm = Pxx / Psum
    peak_idx = int(np.argmax(Pxx))
    centroid = float(np.sum(f * p_norm))
    spread   = float(np.sqrt(np.sum(((f - centroid)**2) * p_norm)))
    spec_entropy = float(-np.sum(p_norm * np.log(p_norm)))
    cumsum = np.cumsum(p_norm)
    idx95 = np.searchsorted(cumsum, 0.95)
    idx95 = min(idx95, len(f) - 1)
    rolloff95 = float(f[idx95])
    rms_freq = float(np.sqrt(np.sum((f**2) * Pxx) / Psum))
    return dict(
        freq_peak=float(f[peak_idx]),
        spec_centroid=centroid,
        freq_std=spread,
        spec_entropy=spec_entropy,
        spec_rolloff95=rolloff95,  # ★ 注意拼写：spec_rolloff95（两个 l）
        rms_freq=rms_freq,
    )



def char_freqs(rpm, n, d, D):
    fr = rpm / 60.0
    bpfo = fr * n/2.0 * (1 - d/D)
    bpfi = fr * n/2.0 * (1 + d/D)
    bsf  = fr * (D/d) * (1 - (d/D)**2)
    ftf  = 0.5 * fr * (1 - d/D)
    return fr, bpfo, bpfi, bsf, ftf


def envelope_features(x, sr, rpm, tol=0.05, bearing=BEARING):
    if rpm is None or not np.isfinite(rpm) or rpm <= 1:
        return {}
    x = np.asarray(x, dtype=float).ravel()   # ★ flatten
    env = np.abs(hilbert(x))
    nper = min(len(env), 4096); nper = max(nper, 256)
    f, P = welch(env, fs=sr, nperseg=nper, noverlap=nper//2, detrend="constant")
    P = P + 1e-20
    out = {}
    _, bpfo, bpfi, bsf, ftf = char_freqs(float(rpm), bearing["n"], bearing["d"], bearing["D"])
    for name, f0 in {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}.items():
        lo, hi = (1 - tol) * f0, (1 + tol) * f0
        m = (f >= lo) & (f <= hi)
        E = float(np.trapz(P[m], f[m])) if np.any(m) else 0.0
        out[f"{name}_E"] = E
    return out

def wavelet_packet_features(x, wavelet="db4", level=3):
    x = np.asarray(x, dtype=float).ravel()   # ★ flatten
    wp = pywt.WaveletPacket(x, wavelet, maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, "freq")]
    energies = [np.sum(np.square(wp[node].data)) for node in nodes]
    total = np.sum(energies) + 1e-20
    probs = np.array(energies) / total
    H = -np.sum(probs * np.log(probs + 1e-12))
    feats = {f"wp_E{i}": float(p) for i, p in enumerate(probs)}
    feats["wp_entropy"] = float(H)
    return feats


def main():
    df = pd.read_parquet(SEGMENTS_PARQUET)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        seg_file = Path(r["seg_file"])
        if not seg_file.exists():
            continue
        x = np.load(seg_file).astype(float).ravel()
        feat = dict(
            seg_file=str(seg_file),
            label=r.get("label"),
            load=r.get("load"),
            rpm=r.get("rpm"),
            size_code=r.get("size_code"),
            clock_pos=r.get("clock_pos"),
        )
        feat.update(time_features(x))
        feat.update(spectral_features(x, fs))
        feat.update(envelope_features(x, fs, r.get("rpm")))
        feat.update(wavelet_packet_features(x))
        rows.append(feat)
    df_feat = pd.DataFrame(rows)
    df_feat.to_parquet(OUT_PARQUET, index=False)
    print(f"✅ 特征提取完成：{len(df_feat)} 段，已写入 {OUT_PARQUET}")


if __name__ == "__main__":
    main()
