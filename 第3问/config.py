from pathlib import Path

DETAIL_CSV=Path("第3问/inputs/4.1detail.csv")
RAW_DIR=Path("第3问/inputs/4.1无处理csv")

ARTifacts=Path("第3问/artifacts_lgbm")

OUT_DIR=Path("第3问/outputs");OUT_DIR.mkdir(parents=True,exist_ok=True)

FS=32000
L=4096
HOP=2048
LOWC=100
HIGHC=min(5000,int(0.45*FS))
CHANNEL_INDEX=0
NORM_PM1=0
USE_CORAL=0
RANDOM_SEED=42
