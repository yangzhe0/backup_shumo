import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
import scipy.io as sio

# 方便重采样之后复用
HZ=32
# 不在前期做角域归一
rpm1=False

if (HZ!=32):
    DATA_ROOT = Path("data/源域数据集")
else:
    DATA_ROOT = Path("data/32k_none")
SPLIT_NAMES = ["12kHz_DE_data", "12kHz_FE_data", "48kHz_DE_data", "48kHz_Normal_data"]
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

def find_splits(root: Path) -> Dict[str, Path]:
    found = {}
    if root.exists():
        for s in SPLIT_NAMES:
            p = root / s
            if p.exists() and p.is_dir():
                found[s] = p
    return found

def scan_all_mat(split_dirs: Dict[str, Path]) -> List[Path]:
    mats = []
    for _, d in split_dirs.items():
        mats += list(d.rglob("*.mat"))
        mats += list(d.rglob("*.MAT"))
    return sorted(set(mats))

def infer_pos_from_split(split: Optional[str]) -> Optional[str]:
    if not split:
        return None
    return "DE" if "DE" in split else ("FE" if "FE" in split else None)

def infer_sr_from_split(split: Optional[str]) -> Optional[int]:
    if (HZ!=32):
        if not split:
            return None
        if split.startswith("12kHz"):
            return 12000
        if split.startswith("48kHz"):
            return 48000
        return None
    else:
        return 32000

def infer_split_from_path(p: Path) -> Optional[str]:
    for s in SPLIT_NAMES:
        if s in p.parts:
            return s
    return None

def infer_label_from_path_or_name(p: Path, split: Optional[str]) -> Optional[str]:
    name = p.stem
    if (split and "Normal" in split) or name.startswith("N_") or "Normal" in name:
        return "N"
    if "IR" in p.parts:
        return "IR"
    if "OR" in p.parts:
        return "OR"
    if "B" in p.parts:
        return "B"
    if "OR" in name:
        return "OR"
    if re.search(r"\bIR\d{3}\b", name, re.I):
        return "IR"
    if re.search(r"\bB\d{3}\b", name, re.I):
        return "B"
    m = re.search(r"\b(IR|OR|B)\d{3}\b", name, re.I)
    if m:
        return m.group(1).upper()
    return None

def rpm_from_name(name: str) -> Optional[int]:
    m = re.search(r"\((\d+)\s*rpm\)", name, re.I)
    return int(m.group(1)) if m else None

def rpm_from_file(fpath: Path) -> Optional[int]:
    try:
        data = sio.loadmat(fpath)
        valid_keys = [k for k in data.keys() if not k.startswith("__")]
        for k in valid_keys:
            arr = np.array(data[k])
            if arr.size == 1:
                try:
                    return int(arr.item())
                except Exception:
                    continue
    except NotImplementedError:
        import h5py
        with h5py.File(fpath, "r") as f:
            def walk(group):
                for k, v in group.items():
                    if isinstance(v, h5py.Dataset):
                        arr = np.array(v)
                        if arr.size == 1:
                            yield arr
                    elif isinstance(v, h5py.Group):
                        yield from walk(v)
            for arr in walk(f):
                try:
                    return int(arr.item())
                except Exception:
                    continue
    return None

def infer_size_and_load_from_name(name: str) -> Tuple[Optional[str], Optional[int]]:
    m_size = re.search(r"(?:B|IR|OR)(\d{3})", name, re.I)
    size_code = m_size.group(1) if m_size else None
    m_load = re.search(r"_(\d)(?:[_\.]|$)", name)
    load = int(m_load.group(1)) if m_load else None
    return size_code, load

def clock_from_name(name: str):
    name = name.replace("＠", "@")
    m = re.search(r"@(\d{1,2})(?=(?:[_\.\-\s(]|$))", name)
    if not m:
        return None
    return int(m.group(1))

def build_manifest(mat_files: List[Path]) -> pd.DataFrame:
    rows = []
    none_counts = {"split": 0, "pos": 0, "sr_hz": 0, "label": 0, "size_code": 0, "load": 0, "rpm": 0, "clock_pos": 0}
    for f in mat_files:
        split = infer_split_from_path(f)
        pos   = infer_pos_from_split(split)
        sr    = infer_sr_from_split(split)
        label = infer_label_from_path_or_name(f, split)
        size_code, load = infer_size_and_load_from_name(f.name)
        if (rpm1):
            rpm = 600
        else:
            rpm = rpm_from_name(f.name) or rpm_from_file(f)
        clock_pos = clock_from_name(f.name)
        rec = {
            "path": str(f),
            "file": f.name,
            "split": split if split else None,
            "pos": pos,
            "sr_hz": sr,
            "label": label if label else None,
            "size_code": size_code if size_code else None,
            "load": load if load is not None else None,
            "rpm": rpm if rpm is not None else None,
            "clock_pos": clock_pos
        }
        for k, v in rec.items():
            if v is None and k in none_counts:
                none_counts[k] += 1
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["split", "label", "size_code", "load", "clock_pos"], na_position="last").reset_index(drop=True)
    total = len(df)
    print("⚠️ None 值统计：")
    for k, c in none_counts.items():
        print(f"  - {k:10s}: {c:4d} / {total:4d}")
    return df


splits = find_splits(DATA_ROOT)
mat_files = scan_all_mat(splits)
df = build_manifest(mat_files)
print(f"共扫描到 {len(df)} 个文件。")
out_csv = fr"result/detail_{HZ}.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")