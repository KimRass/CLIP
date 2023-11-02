from torch.utils.data import DataLoader
from pathlib import Path

from utils import load_config
from cifar100 import CIFAR100Dataset

CONFIG = load_config(Path(__file__).parent/"config.yaml")

MEAN = (0.507, 0.487, 0.441)
STD = (0.267, 0.256, 0.276)
ds = CIFAR100Dataset(
    data_dir="/Users/jongbeomkim/Documents/datasets/cifar-100-python",
    img_size=224,
    mean=MEAN,
    std=STD,
    split="train",
)
dl = DataLoader(ds, batch_size=4, shuffle=True)