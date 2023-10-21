import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
import os
from PIL import Image
import re
from collections import defaultdict
import random

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# from transformers import DistilBertTokenizerFast
# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def _encode(text, tokenizer, max_len):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    token_ids = torch.tensor(encoding["input_ids"])
    return token_ids


class Flickr8kDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # self.unk_id = tokenizer.unk_token_id
        # self.cls_id = tokenizer.cls_token_id
        # self.sep_id = tokenizer.sep_token_id
        # self.pad_id = tokenizer.pad_token_id

        self.images_dir = self.data_dir/"Images"
        self.imgs = sorted(list(self.images_dir.glob("**/*.jpg")))

        self.transforms = T.Compose([
            T.Resize(size=224),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.captions = defaultdict(list)
        with open(self.data_dir/"captions.txt", mode="r") as f:
            for line in f:
                line = line.strip()
                if ".jpg" in line:
                    split_idx = re.search(pattern=r"(.jpg)", string=line).span()[1]
                    img_path = self.images_dir/line[: split_idx]
                    text = line[split_idx + 1:].replace(" .", ".")
                    self.captions[img_path].append(text)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = Image.open(img_path)
        image = self.transforms(image)

        texts = self.captions[img_path]
        text = random.choice(texts)

        token_ids = _encode(text, tokenizer=self.tokenizer, max_len=self.max_len)
        return image, token_ids


if __name__ == "__main__":
    flickr = Flickr8kDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k",
        tokenizer=tokenizer,
        max_len=64,
    )
    flickr[0]
