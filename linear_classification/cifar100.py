# References:
    # https://github.com/KimRass/ViT/blob/main/cifar100.py

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle


def _get_images_and_gts(data_path):
    with open(data_path, mode="rb") as f:
        data_dic = pickle.load(f, encoding="bytes")

    imgs = data_dic[b"data"]
    imgs = imgs.reshape(-1, 3, 32, 32)
    imgs = imgs.transpose(0, 2, 3, 1)

    gts = data_dic[b"fine_labels"]
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
    def __init__(self, data_dir, mean, std, split="train"):
        super().__init__()

        self.imgs, self.gts = _get_cifar100_images_and_gts(data_dir=data_dir, split=split)

        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)],
                p=0.5,
            ),
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
