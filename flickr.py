import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
import os
from PIL import Image
import re
from collections import defaultdict
import random
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def encode(text, tokenizer):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    return encoding["input_ids"]


class Flickr8kDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = 0

        self.images_dir = self.data_dir/"Images"
        self.img_paths = sorted(list(map(str, self.images_dir.glob("**/*.jpg"))))

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
                    img_path = str(self.images_dir/line[: split_idx])
                    text = line[split_idx + 1:].replace(" .", ".")
                    if img_path in self.img_paths:
                        self.captions[img_path].append(text)

        # for img_path, texts in self.captions.items():
        #     ls_token_ids = encode(texts, tokenizer=self.tokenizer)
        #     self.captions[img_path] = ls_token_ids

        #     for token_ids in ls_token_ids:
        #         token_ids_len = len(token_ids)
        #         if token_ids_len > self.max_len:
        #             self.max_len = token_ids_len
        # print(f"The maximum length of token IDs is {self.max_len:,}.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = self.transforms(image)

        # ls_token_ids = self.captions[img_path]
        # token_ids = random.choice(ls_token_ids)
        texts = self.captions[img_path]
        text = random.choice(texts)
        token_ids = encode(text, tokenizer=self.tokenizer)
        print(img_path.name, text)
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

        ls_token_ids = [self._pad(token_ids=token_ids, max_len=max_len) for token_ids in ls_token_ids]
        attn_mask = self._get_attention_mask(token_ids)
        return torch.stack(images), torch.as_tensor(ls_token_ids), attn_mask
