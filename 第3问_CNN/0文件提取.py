# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm

# ROOT = Path("第3问_CNN/inputs")            # 原始数据目录
# DETAIL_CSV = Path("第3问_CNN/0detail.csv") # 原始索引表
# OUT_DIR = Path("第3问_CNN/cleaned")        # 新目录（存处理后的csv）
# INDEX_OUT = Path("第3问_CNN/index.csv")    # 新索引表

# OUT_DIR.mkdir(parents=True, exist_ok=True)

# df_detail = pd.read_csv(DETAIL_CSV)
# rows = []

# # 定义 A~P 列
# ap_cols = list("ABCDEFGHIJKLMNOP")

# for csv_path in tqdm(list(ROOT.rglob("*.csv"))):
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         print(f"❌ 读取失败 {csv_path}: {e}")
#         continue

#     # ---- 提取通道 ----
#     de_cols = [c for c in df.columns if "DE" in c.upper()]
#     fe_cols = [c for c in df.columns if "FE" in c.upper()]
#     ba_cols = [c for c in df.columns if "BA" in c.upper()]

#     new_df = pd.DataFrame()
#     new_df["DE"] = df[de_cols].iloc[:, 0] if de_cols else ""
#     new_df["FE"] = df[fe_cols].iloc[:, 0] if fe_cols else ""
#     new_df["BA"] = df[ba_cols].iloc[:, 0] if ba_cols else ""

#     # ---- 合并 A~P 到 NO ----
#     vals = []
#     for c in ap_cols:
#         if c in df.columns:
#             # 保留非空值并将每个非空值转换为字符串
#             series = df[c].dropna().astype(str)
#             # 过滤掉空格内容并拼接为"列名:值"格式
#             vals.extend([f"{v.strip()}" for v in series if v.strip() != ""])

#     # 将拼接的结果添加到新列 NO
#     new_df["NO"] = ";".join(vals) if vals else ""

#     # ---- 保存到新目录 ----
#     rel = csv_path.relative_to(ROOT)
#     new_path = OUT_DIR / rel
#     new_path.parent.mkdir(parents=True, exist_ok=True)
#     new_df.to_csv(new_path, index=False, encoding="utf-8-sig")

#     # ---- 匹配元信息 ----
#     if "位置" in df_detail.columns:
#         match = df_detail[df_detail["位置"].str.contains(csv_path.name, na=False, regex=False)]
#     else:
#         match = pd.DataFrame()
#     meta = match.iloc[0].to_dict() if not match.empty else {}

#     # ---- 索引表记录 ----
#     row = {"csv_path": str(new_path)}
#     for col in ["发生点", "赫兹", "故障类", "尺寸", "负载", "转速", "采样点"]:
#         row[col] = meta.get(col, "")
#     rows.append(row)

# # ---- 保存索引表 ----
# df_index = pd.DataFrame(rows)
# df_index.to_csv(INDEX_OUT, index=False, encoding="utf-8-sig")

# print(f"✅ 已完成：所有文件已清洗到 {OUT_DIR}")
# print(f"✅ 索引表保存到 {INDEX_OUT}，共 {len(df_index)} 条记录")
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 定义原始数据目录和输出目录
ROOT = Path("第3问_CNN/inputs/目标域")            # 原始数据目录
OUT_DIR = Path("第3问_CNN/inputs/目标域1")        # 新目录（存处理后的csv）

# 创建输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 处理每一个 CSV 文件
for csv_path in tqdm(list(ROOT.rglob("*.csv"))):
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取失败 {csv_path}: {e}")
        continue

    # ---- 先将所有列名更改为 NO ----
    df.columns = ['NO'] * len(df.columns)

    # ---- 添加前三列并修改列名 ----
    # 创建三个新的列，填充空值或默认值
    df.insert(0, "DE", "")  # 在第一列插入 "DE"
    df.insert(1, "FE", "")  # 在第二列插入 "FE"
    df.insert(2, "BA", "")  # 在第三列插入 "BA"

    # ---- 保存到新目录 ----
    rel = csv_path.relative_to(ROOT)  # 生成相对路径
    new_path = OUT_DIR / rel  # 新路径
    new_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
    df.to_csv(new_path, index=False, encoding="utf-8-sig")  # 保存 CSV 文件

print(f"✅ 所有文件已处理并保存到 {OUT_DIR}")
