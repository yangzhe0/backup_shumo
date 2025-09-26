from model import SimpleCNN
from data_preprocessing import load_and_clean_index
from train import train_model
from dataset import BearingDataset, TargetDataset
from torch.utils.data import DataLoader

INDEX_PATH = '第3问_CNN/index.csv'
MANIFEST_PATH = '第3问_CNN/source_manifest.csv'
TARGET_DOMAIN_PREFIX = '第3问_CNN/processed_images/target'

df_filtered = load_and_clean_index(INDEX_PATH, COLUMN_SEARCH_MAP, TARGET_DOMAIN_PREFIX)

source_dataset = BearingDataset(manifest_path=MANIFEST_PATH, fixed_width=4096)
source_loader = DataLoader(dataset=source_dataset, batch_size=32, shuffle=True)

target_dataset = TargetDataset(manifest_path='path/to/target_manifest.csv', fixed_width=4096)
target_loader = DataLoader(dataset=target_dataset, batch_size=32, shuffle=True)

model = SimpleCNN(num_classes=len(source_dataset.classes), input_shape=(1, 129, 4096))

trained_model = train_model(model, source_loader, target_loader)
