# 本程序及代码是在人工智能工具 ChatGPT (GPT-5，OpenAI，2025年3月) 辅助下完成的

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 输入/输出路径配置
SRC_DIR     = Path("第1问/1.1csv_32k")              # 已重采样的多列 CSV
DST_DIR     = Path("第1问/1.2csv_32k_DE")           # 输出的单列 CSV
DETAIL_CSV  = Path("第1问/1.1detail_32k.csv")       # 原始索引
OUT_DETAIL  = Path("第1问/1.2detail_32k_DE.csv")    # 新索引

DST_DIR.mkdir(parents=True, exist_ok=True)

df_index = pd.read_csv(DETAIL_CSV)
rows = []

for _, row in tqdm(df_index.iterrows(), total=len(df_index)):
    csv_path = Path(row["csv_path"])
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)

    # 提取 DE 通道列
    de_cols = [c for c in df.columns if "DE" in c.upper()]
    if not de_cols:
        continue

    # 默认保留所有检测到的 DE 列
    df_de = df[de_cols]
    # 如果只需第一个 DE 列，可改成： df_de = df[[de_cols[0]]]

    # 保存新 CSV
    rel_path = csv_path.relative_to(SRC_DIR)
    out_path = DST_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_de.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 更新索引信息
    new_meta = row.to_dict()
    new_meta["csv_path"] = str(out_path)
    new_meta["channels"] = ",".join(de_cols)
    rows.append(new_meta)

# 保存新的索引表
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

print(f"✅ 提取完成，共 {len(df_out)} 个文件")
print("新索引:", OUT_DETAIL)
