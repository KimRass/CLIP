# Source: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/

from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
import json

from data_augmentation import get_val_transformer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_imagenet1k_classes(json_path):
    with open(json_path, mode="r") as f:
        classes = json.load(f)
    classes = {v[0]: (int(k), v[1]) for k, v in classes.items()}
    return classes


class ImageNet1kDataset(Dataset):
    def __init__(self, data_dir, img_size, classes, split=None):
        super().__init__()

        self.classes = classes

        self.data_dir = Path(data_dir)
        if split is not None:
            self.data_dir /= split

        self.img_paths = sorted(list(map(str, self.data_dir.glob("**/*.JPEG"))))
        self.transformer = get_val_transformer(img_size=img_size)

    def _img_path_to_gt(self, img_path, classes):
        gt = classes[Path(img_path).parent.name][0]
        return gt

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transformer(image)

        gt = self._img_path_to_gt(img_path, classes=self.classes)
        return image, gt


if __name__ == "__main__":
    ds = ImageNet1kDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/imagenet-mini",
        tokenizer=None,
        img_size=28,
        split="train",
    )
    image, prompt = ds[500]
    image.show()
    prompt
