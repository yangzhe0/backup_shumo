# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

import re
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm

SRC_ROOT   = Path("0数据集/源域数据集")
CSV_DIR    = Path("第1问/0csv无处理")
DETAIL_CSV = Path("第1问/0detail.csv")

CSV_DIR.mkdir(parents=True, exist_ok=True)
DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)


def extract_signals(path: Path):
    """
    从 .mat 文件提取所有有效的一维信号。
    对 v7.3 格式（HDF5）单独处理。
    """
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        signals = {
            k: np.array(v).squeeze()
            for k, v in data.items()
            if not k.startswith("__")
            and isinstance(v, np.ndarray)
            and v.ndim == 1
            and v.size > 10
        }
        return signals, data
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            signals = {}
            for k in f.keys():
                v = np.array(f[k]).squeeze()
                if v.ndim == 1 and v.size > 10:
                    signals[k] = v
            return signals, f
    return {}, None


def rpm_from_data(data):
    """
    尝试从数据内容或键名中解析转速 rpm。
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if "RPM" in k or "rpm" in k or k.lower() == "n":
                arr = np.array(v).squeeze()
                if np.isscalar(arr):
                    return int(arr)
                if arr.size == 1:
                    return int(arr.item())

    if isinstance(data, h5py.File):
        for k in data.keys():
            if "RPM" in k or "rpm" in k or k.lower() == "n":
                arr = np.array(data[k]).squeeze()
                if np.isscalar(arr):
                    return int(arr)
                if arr.size == 1:
                    return int(arr.item())
    return None


def parse_meta(path: Path, data=None):
    """
    从文件名/路径推断采样率、工况标签、负载、转速等元信息。
    """
    name = path.stem
    split = next((p for p in path.parts if "Hz" in p), None)

    sr = 12000 if "12kHz" in str(path) else (48000 if "48kHz" in str(path) else None)

    label = None
    if "IR" in str(path):
        label = "IR"
    elif "OR" in str(path):
        label = "OR"
    elif "B" in str(path):
        label = "B"
    elif "Normal" in str(path) or name.startswith("N"):
        label = "N"

    m_size = re.search(r"(?:B|IR|OR)(\d{3})", name, re.I)
    size_code = m_size.group(1) if m_size else None

    m_load = re.search(r"_(\d)(?:[_\.\)]|$)", name)
    load = int(m_load.group(1)) if m_load else None

    m_rpm = re.search(r"(\d+)\s*rpm", name, re.I)
    rpm = int(m_rpm.group(1)) if m_rpm else None
    if rpm is None and data is not None:
        rpm = rpm_from_data(data)

    m_clock = re.search(r"@(\d{1,2})", name)
    clock_pos = int(m_clock.group(1)) if m_clock else None

    return {
        "path": str(path),
        "file": path.name,
        "split": split,
        "sr_hz": sr,
        "label": label,
        "size_code": size_code,
        "load": load,
        "rpm": rpm,
        "clock_pos": clock_pos,
    }


# 主流程：批量转换 MAT → CSV，并记录索引表
rows = []
for f in tqdm(list(SRC_ROOT.rglob("*.mat"))):
    signals, data = extract_signals(f)
    if not signals:
        continue

    meta = parse_meta(f, data)

    out_path = CSV_DIR / f.relative_to(SRC_ROOT)
    out_path = out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(signals)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    meta["csv_path"] = str(out_path)
    rows.append(meta)

df_index = pd.DataFrame(rows)
df_index.to_csv(DETAIL_CSV, index=False, encoding="utf-8-sig")
print(f"✅ 转换完成，共 {len(df_index)} 个文件")
