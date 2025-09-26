# 2重采样.py  —— 保持原接口/输出不变，仅内部加入角域归一
import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path
from fractions import Fraction
from scipy.signal import resample_poly
from tqdm import tqdm

FS_OUT = 32000      # 目标采样率（保持不变）
FR_REF = 10.0       # 参考转频，统一到 10 Hz（10 转/秒）
MAX_DEN = 480       # 有理数逼近分母上限，供 resample_poly 使用

def _infer_fr_from_mat(data: dict) -> float | None:
    """从 .mat 中抓 rpm/RPM/n 等，返回 fr=rpm/60；抓不到则 None"""
    for k in list(data.keys()):
        if k.startswith("__"): 
            continue
        if "RPM" in k or "rpm" in k or k.lower()=="n":
            try:
                val = float(np.array(data[k]).squeeze().item())
                if val > 0:
                    return val / 60.0
            except Exception:
                pass
    return None

def _get_fr(row, mat_data) -> float | None:
    """优先用索引表的 rpm；没有就读 .mat；都没有返回 None"""
    if "rpm" in row and not pd.isna(row["rpm"]):
        try:
            rpm = float(row["rpm"])
            if rpm > 0:
                return rpm / 60.0
        except Exception:
            pass
    return _infer_fr_from_mat(mat_data)

def _resample_only_fs(sig: np.ndarray, fs_in: int) -> np.ndarray:
    """仅采样率重采样（兼容老逻辑）"""
    if fs_in == FS_OUT:
        return sig
    if fs_in == 12000:   up, down = 8, 3    # 12k → 32k
    elif fs_in == 48000: up, down = 2, 3    # 48k → 32k
    else:
        raise ValueError(f"不支持的采样率: {fs_in}")
    return resample_poly(sig, up, down)

def _resample_angle_32k(sig: np.ndarray, fs_in: int, fr: float) -> np.ndarray:
    """
    一步完成：
      1) 时间轴缩放到参考转频 FR_REF（角域归一）
      2) 采样率对齐到 32k
    总比例 = (FS_OUT/fs_in) * (FR_REF/fr)
    """
    total = (FS_OUT / fs_in) * (FR_REF / fr)
    frac = Fraction(total).limit_denominator(MAX_DEN)
    return resample_poly(sig, frac.numerator, frac.denominator)

def batch_resample(csv_path: str, out_root: str = "data/32kHz", use_angle: bool = True):
    """
    保持原签名/默认输出；仅多了一个开关 use_angle：
      - True：角域归一 + 32k（推荐）
      - False：仅采样率重采样（旧行为）
    输出结构、变量名、列向量形状与原来一致，后续程序可直接用。
    """
    df = pd.read_csv(csv_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 兼容 CSV 中可能出现的 Windows 路径分隔符（反斜杠）或首尾空白
        raw_path = str(row.get("path", ""))
        raw_path = raw_path.strip().replace('\\', '/')
        fpath = Path(raw_path)
        sr = int(row["sr_hz"]) if not pd.isna(row["sr_hz"]) else None
        if sr not in (12000, 48000, 32000):
            continue

        # 检查文件是否存在，避免因为路径分隔符问题导致 loadmat 抛 OSError
        if not fpath.exists():
            # 如果是相对路径（以 data/... 开头），也尝试相对于仓库根目录解析一次
            alt = Path.cwd() / fpath
            if alt.exists():
                fpath = alt
            else:
                print(f"⚠️ 文件不存在，跳过: {raw_path}")
                continue

        try:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"⚠️ 无法读取 .mat 文件 (skipping) {fpath}: {e}")
            continue

        # 读取转频（角域需要）
        fr = _get_fr(row, mat) if use_angle else None

        signals = {}
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            arr = np.squeeze(v)
            # 仅对“一维时间序列”做重采样；标量/结构保持原样
            if arr.ndim == 1 and arr.size > 10 and np.isfinite(arr).all():
                if use_angle and (fr is not None) and (fr > 0):
                    y = _resample_angle_32k(arr.astype(float), sr, fr)
                else:
                    y = _resample_only_fs(arr.astype(float), sr)
                signals[k] = y.reshape(-1, 1)   # 仍旧“列向量”存储
            else:
                signals[k] = v

        # 输出到原来的目录结构（不改路径，不改文件名）
        rel_path = fpath.relative_to(Path("data/源域数据集"))
        out_path = (Path(out_root) / rel_path).with_suffix(".mat")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(out_path, signals)

    print(f"✅ 已完成重采样（use_angle={use_angle}），输出：{out_root}")

if __name__ == "__main__":
    # 仅采样率重采样
    # 不对角域归一，保持原逻辑
    batch_resample("result/12_48k_detail.csv", out_root="data/32kHz", use_angle=False)
