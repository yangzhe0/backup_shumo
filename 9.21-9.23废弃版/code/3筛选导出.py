import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
# 构造输出路径



def extract_de_signal(mat_data):
    """从 .mat 文件中提取 DE_time 信号"""
    for k in mat_data.keys():
        if "DE_time" in k:
            return np.squeeze(mat_data[k])
    return None


def export_single_file(row, out_root, src_root="data/32kHz"):
    """处理单个 .mat 文件并导出对应的 CSV"""
    fpath = Path(row["path"])
    mat_data = sio.loadmat(fpath)
    signal = extract_de_signal(mat_data)
    if signal is None:
        print(f"⚠️ 未找到 DE_time: {fpath}")
        return None

    rows = []
    for i, val in enumerate(signal):
        rec = {
            # "path": row["path"],
            # "file": row["file"],
            # "line_index": i,
            # "split": row["split"],
            "pos": row["pos"],
            "sr_hz": row["sr_hz"],
            "label": row["label"],
            "size_code": row["size_code"],
            "load": row["load"],
            "rpm": row["rpm"],
            "rpm/60": row["rpm"] / 60 if not pd.isna(row["rpm"]) else None,
            "clock_pos": row["clock_pos"],
            "sig": val
        }
        rows.append(rec)

    df_out = pd.DataFrame(rows)

    # 构造输出路径 (以 src_root 为基准)
    rel_path = fpath.relative_to(src_root).with_suffix(".csv")
    out_path = Path(out_root) / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(out_path, index=False)
    return out_path

def build_all_dataset(detail_csv="result/detail_32.csv", out_root="data/csv_DE_N_32", all_csv="result/all_DE_32_long.csv"):
    df = pd.read_csv(detail_csv)

    df_de = df[(df["pos"] == "DE") | (df["label"] == "N")].reset_index(drop=True)

    all_paths = []
    for _, row in tqdm(df_de.iterrows(), total=len(df_de)):
        out_path = export_single_file(row, out_root)
        if out_path:
            all_paths.append(out_path)

def extract_de_signal(mat_data):
    # 更快地找键
    key = next((k for k in mat_data.keys() if "DE_time" in k), None)
    return None if key is None else np.squeeze(mat_data[key])

def export_single_file(row, out_root, src_root="data/32k_none"):
    fpath = Path(row["path"])
    # 用 squeeze_me=True 加速并简化结构
    mat_data = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    signal = extract_de_signal(mat_data)
    if signal is None:
        print(f"⚠️ 未找到 DE_time: {fpath}")
        return None

    # ✅ 向量化构建，不再用 Python for 循环逐点append
    df_out = pd.DataFrame({"sig": np.asarray(signal).reshape(-1)})
    df_out["pos"] = row["pos"]
    df_out["sr_hz"] = row["sr_hz"]
    df_out["label"] = row["label"]
    df_out["size_code"] = row["size_code"]
    df_out["load"] = row["load"]
    df_out["rpm"] = row["rpm"]
    df_out["rpm/60"] = (row["rpm"] / 60) if not pd.isna(row["rpm"]) else None
    df_out["clock_pos"] = row["clock_pos"]

    rel_path = fpath.relative_to(src_root).with_suffix(".csv.gz")  # ✅ 压缩
    out_path = Path(out_root) / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ 压缩写出，IO更快、占用更小
    df_out.to_csv(out_path, index=False, compression="gzip")
    return out_path

def build_all_dataset(detail_csv="result/detail_32.csv",
                      out_root="data/csv_DE_N_32"):
    df = pd.read_csv(detail_csv)
    df_sel = df[(df["pos"] == "DE") | (df["label"] == "N")].reset_index(drop=True)  # ✅

    for _, row in tqdm(df_sel.iterrows(), total=len(df_sel)):
        export_single_file(row, out_root)
    # 不合并，节省空间

if __name__ == "__main__":
    build_all_dataset(r"result/32k_none_detail.csv", r"data/csv_DE_N_32")
