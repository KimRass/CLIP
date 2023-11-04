# Source: https://www.cs.toronto.edu/~kriz/cifar.html
# Reference: https://github.com/KimRass/ViT/blob/main/cifar100.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle

from utils import load_config

CONFIG = load_config(Path(__file__).parent/"config.yaml")


def _get_images_and_gts(data_path):
    with open(data_path, mode="rb") as f:
        data_dic = pickle.load(f, encoding="bytes")

    imgs = data_dic[b"data"]
    imgs = imgs.reshape(-1, 3, 32, 32)
    imgs = imgs.transpose(0, 2, 3, 1)

    # gts = data_dic[b"fine_labels"]
    gts = data_dic[b"filenames"]
    return imgs, gts


def _get_cifar100_images_and_gts(data_dir, split="train"):
    imgs, gts = _get_images_and_gts(Path(data_dir)/split)
    return imgs, gts


def get_cifar100_mean_and_std(data_dir, split="train"):
    imgs, _ = _get_cifar100_images_and_gts(data_dir=data_dir, split=split)

    imgs = imgs.astype("float") / 255
    n_pixels = imgs.size // 3
    sum_ = imgs.reshape(-1, 3).sum(axis=0)
    sum_square = (imgs ** 2).reshape(-1, 3).sum(axis=0)
    mean = (sum_ / n_pixels).round(3)
    std = (((sum_square / n_pixels) - mean ** 2) ** 0.5).round(3)
    return mean, std


class CIFAR100Dataset(Dataset):
    def __init__(self, data_dir, img_size, mean, std, split="train"):
        super().__init__()

        self.imgs, self.gts = _get_cifar100_images_and_gts(data_dir=data_dir, split=split)

        self.transform = T.Compose([
            T.Resize(size=img_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        image = Image.fromarray(img)
        image = self.transform(image)

        gt = self.gts[idx]
        gt = torch.tensor(gt).long()
        return image, gt


if __name__ == "__main__":
    data_path = "/Users/jongbeomkim/Documents/datasets/cifar-100-python/train"
    _, gts = _get_images_and_gts(data_path)
    gts[: 10]

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
    di = iter(dl)

    image, gt = next(di)
    gt