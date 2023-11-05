# Source: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from pathlib import Path
import os
from PIL import Image

from utils import get_imagenet1k_classes
from flickr import encode
from data_augmentation import get_train_transformer, get_val_transformer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

CLASSES = get_imagenet1k_classes("/Users/jongbeomkim/Desktop/workspace/CLIP/imagenet1k_classes.json")


def _class_name_to_prompt(class_name):
    return f"""A photo a {class_name.replace("_", " ")}"""


def _img_path_to_prompt(img_path):
    dir_name = CLASSES[Path(img_path).parent.name]
    return _class_name_to_prompt(dir_name)


# class ImageNet1kDataset(Dataset):
#     def __init__(self, data_dir, tokenizer, img_size, split="train"):
#         super().__init__()

#         self.data_dir = Path(data_dir)/split
#         self.tokenizer = tokenizer

#         self.img_paths = sorted(list(map(str, self.data_dir.glob("**/*.JPEG"))))

#         self.transformer = get_val_transformer(img_size=img_size)

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         image = Image.open(img_path)
#         image = self.transformer(image)

#         prompt =_img_path_to_prompt(img_path)
#         token_ids = encode(prompt, tokenizer=self.tokenizer, max_len=self.max_len)
#         return image, token_ids


def get_imagenet1k_dataset(data_dir, img_size):
    ds = ImageFolder(root=data_dir, transform=get_val_transformer(img_size))
    return ds


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
