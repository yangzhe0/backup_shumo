# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch, hilbert
from scipy.stats import kurtosis, skew, entropy
import pywt
from tqdm import tqdm
from config import OUT_DIR, FS

IN_PARQUET = OUT_DIR / "target_segments.parquet"
OUT_PARQUET = OUT_DIR / "target_features.parquet"

BEARING = dict(n=9, d=0.3126, D=1.537)
RPM_DEFAULT = 600.0

def safe_entropy(x, bins=64):
    h, _ = np.histogram(x, bins=bins, density=True)
    return float(entropy(h + 1e-12))

def time_feats(x):
    x = x.astype(float).ravel()
    xa = np.abs(x)
    rms = np.sqrt(np.mean(x**2) + 1e-20)
    mean_abs = np.mean(xa) + 1e-20
    sqrt_abs_mean = np.mean(np.sqrt(xa)) + 1e-20
    return dict(
        mean=float(np.mean(x)), std=float(np.std(x, ddof=1)), rms=float(rms),
        skew=float(skew(x, bias=False)), kurtosis=float(kurtosis(x, fisher=True, bias=False)),
        p2p=float(np.max(x) - np.min(x)), crest_factor=float(np.max(xa) / rms),
        shape_factor=float(rms / mean_abs), impulse_factor=float(np.max(xa) / mean_abs),
        clearance_factor=float(np.max(xa) / (sqrt_abs_mean**2)),
        sig_entropy=float(safe_entropy(x, bins=64))
    )

def spec_feats(x, fs):
    f, P = welch(x.ravel(), fs=fs, nperseg=min(len(x.ravel()), 2048))
    P += 1e-20
    Psum = np.trapz(P, f)
    pn = P / Psum
    peak = float(f[int(np.argmax(P))])
    centroid = float(np.sum(f * pn))
    spread = float(np.sqrt(np.sum(((f - centroid)**2) * pn)))
    sent = float(-np.sum(pn * np.log(pn)))
    idx95 = min(np.searchsorted(np.cumsum(pn), 0.95), len(f)-1)
    roll95 = float(f[idx95])
    rmsf = float(np.sqrt(np.sum((f**2) * P) / Psum))
    return dict(freq_peak=peak, spec_centroid=centroid, freq_std=spread,
                spec_entropy=sent, spec_rolloff95=roll95, rms_freq=rmsf)

def char_freqs(rpm, n, d, D):
    fr = rpm/60.0
    bpfo = fr * n/2.0 * (1 - d/D)
    bpfi = fr * n/2.0 * (1 + d/D)
    bsf = fr * (D/d) * (1 - (d/D)**2)
    ftf = 0.5 * fr * (1 - d/D)
    return fr, bpfo, bpfi, bsf, ftf

def envelope_feats(x, fs, rpm, tol=0.10):
    env = np.abs(hilbert(x.ravel()))
    f, P = welch(env, fs=fs, nperseg=min(len(env), 4096))
    P += np.finfo(float).eps
    out = {}
    _, bpfo, bpfi, bsf, ftf = char_freqs(rpm, **BEARING)
    for name, f0 in {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}.items():
        lo, hi = (1-tol)*f0, (1+tol)*f0
        m = (f >= lo) & (f <= hi)
        out[f"{name}_E"] = float(np.trapz(P[m], f[m])) if np.any(m) else 0.0
    return out

def wp_feats(x, wavelet="db4", level=3):
    wp = pywt.WaveletPacket(x.ravel(), wavelet, maxlevel=level)
    nodes = [n.path for n in wp.get_level(level, "freq")]
    Es = [np.sum(np.square(wp[n].data)) for n in nodes]
    total = np.sum(Es) + 1e-20
    p = np.array(Es)/total
    H = float(-np.sum(p * np.log(p + 1e-12)))
    out = {f"wp_E{i}": float(p[i]) for i in range(len(p))}
    out["wp_entropy"] = H
    return out

def main():
    df = pd.read_parquet(IN_PARQUET)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        x = np.load(r["seg_file"]).astype(float)
        rpm = RPM_DEFAULT
        feat = dict(seg_file=r["seg_file"], file=r["file"], seg_idx=r["seg_idx"])
        feat.update(time_feats(x))
        feat.update(spec_feats(x, FS))
        feat.update(envelope_feats(x, FS, rpm))
        feat.update(wp_feats(x))
        fr, bpfo, bpfi, bsf, ftf = char_freqs(RPM_DEFAULT, **BEARING)
        feat.update(dict(fr=fr, bpfo=bpfo, bpfi=bpfi, bsf=bsf, ftf=ftf))
        rows.append(feat)
    pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
    print(f"✅ 特征提取完成：{len(rows)} 段 → {OUT_PARQUET}")

if __name__ == "__main__":
    main()
