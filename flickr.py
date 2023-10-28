# Sources:
    # https://www.kaggle.com/datasets/adityajn105/flickr8k
    # https://www.kaggle.com/datasets/adityajn105/flickr30k

import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
import re
from collections import defaultdict
import random

from data_augmentation import get_image_transformer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def encode(text, tokenizer, max_len):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=max_len,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    return encoding["input_ids"]


class FlickrDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len, img_size):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.images_dir = self.data_dir/"Images"
        self.img_paths = sorted(list(map(str, self.images_dir.glob("**/*.jpg"))))

        self.transformer = get_image_transformer(img_size=img_size)

        self.captions = defaultdict(list)
        with open(self.data_dir/"captions.txt", mode="r") as f:
            for line in f:
                line = line.strip()
                if ".jpg" in line:
                    split_idx = re.search(pattern=r"(.jpg)", string=line).span()[1]
                    img_path = str(self.images_dir/line[: split_idx])
                    text = line[split_idx + 1:].replace(" .", ".")
                    if img_path in self.img_paths:
                        self.captions[img_path].append(text)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = self.transformer(image)

        texts = self.captions[img_path]
        text = random.choice(texts)
        token_ids = encode(text, tokenizer=self.tokenizer, max_len=self.max_len)
        return image, token_ids


class DataCollatorForDynamicPadding(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

    def _pad(self, token_ids, max_len):
        token_ids = token_ids[: max_len]
        token_ids += [self.pad_id] * (max_len - len(token_ids))
        return token_ids

    def _get_attention_mask(self, token_ids):
        return (token_ids != self.pad_id).long()
    
    def __call__(self, batch):
        images = list()
        ls_token_ids = list()
        max_len = 0
        for image, token_ids in batch:
            images.append(image)
            ls_token_ids.append(token_ids)

            token_ids_len = len(token_ids)
            if token_ids_len > max_len:
                max_len = token_ids_len

        image = torch.stack(images)
        ls_token_ids = [self._pad(token_ids=token_ids, max_len=max_len) for token_ids in ls_token_ids]
        token_ids = torch.as_tensor(ls_token_ids)
        attn_mask = self._get_attention_mask(token_ids)
        return image, token_ids, attn_mask
