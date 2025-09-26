import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import resample, stft

def load_and_clean_index(index_path: str, column_search_map: dict, target_domain_prefix: str):
    df_index = pd.read_csv(index_path, encoding='utf-8-sig')
    df_index.columns = [str(c).strip().lower() for c in df_index.columns]

    found_cols = {}
    for standard_name, possible_names in column_search_map.items():
        for col in df_index.columns:
            if any(p_name in col for p_name in possible_names):
                found_cols[standard_name] = col
                break

    required_keys = ['csv_path', 'fs', 'rpm', 'label']
    if not all(key in found_cols for key in required_keys):
        missing = [key for key in required_keys if key not in found_cols]
        raise ValueError(f"错误：索引文件中缺少必要的列: {missing}")

    df_filtered = df_index[list(found_cols.values())].copy()
    rename_dict = {v: k for k, v in found_cols.items()}
    df_filtered.rename(columns=rename_dict, inplace=True)
    df_filtered['fs'] = pd.to_numeric(df_filtered['fs'], errors='coerce')
    df_filtered['rpm'] = pd.to_numeric(df_filtered['rpm'], errors='coerce')
    df_filtered['domain'] = np.where(df_filtered['csv_path'].str.startswith(target_domain_prefix), 'target', 'source')

    return df_filtered

def select_channel_from_file(file_path: str) -> np.ndarray | None:
    try:
        p = Path(file_path)
        if not p.exists():
            return None
        df_signal = pd.read_csv(p)
        df_signal.columns = [str(c).strip().upper() for c in df_signal.columns]

        if 'DE' in df_signal.columns:
            return df_signal['DE'].values
        elif 'FE' in df_signal.columns:
            return df_signal['FE'].values
        elif 'BA' in df_signal.columns:
            return df_signal['BA'].values
        else:
            return None
    except Exception:
        return None

def resample_signal(signal_array: np.ndarray, original_fs: int, target_fs: int = 32000) -> np.ndarray:
    if original_fs == target_fs:
        return signal_array
    duration = len(signal_array) / original_fs
    num_target_samples = int(duration * target_fs)
    return resample(signal_array, num_target_samples).astype(np.float32)

def angle_sync(signal_array: np.ndarray, rpm: float, fs: int = 32000, ppr: int = 720) -> np.ndarray | None:
    if rpm <= 0 or signal_array.size == 0:
        return None
    fr = rpm / 60.0
    t_original = np.arange(len(signal_array)) / fs
    total_revs = t_original[-1] * fr
    if total_revs < 1:
        return None
    num_full_revs = int(np.floor(total_revs))
    total_angle = 2 * np.pi * num_full_revs
    num_new_points = num_full_revs * ppr
    angle_new = np.linspace(0, total_angle, num=num_new_points, endpoint=False)
    angle_original = 2 * np.pi * fr * t_original
    return np.interp(angle_new, angle_original, signal_array).astype(np.float32)

def generate_stft_image(angle_domain_signal: np.ndarray, ppr: int = 720) -> np.ndarray:
    orders, _, Zxx = stft(angle_domain_signal, fs=ppr, nperseg=256, noverlap=192)
    return np.log1p(np.abs(Zxx)).astype(np.float32)
