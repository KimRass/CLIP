import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
import numpy as np
from PIL import Image
import argparse
from time import time
from pathlib import Path
import wandb

from utils import load_config, get_device, image_to_grid
from flickr import Flickr8kDataset, DataCollatorForDynamicPadding
from train import get_clip

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

if __name__ == "__main__":
    clip = get_clip(config=CONFIG, device=DEVICE)
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/epoch_10.pth"
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    flickr = Flickr8kDataset(data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k_subset", tokenizer=tokenizer)
    collator = DataCollatorForDynamicPadding(tokenizer=tokenizer)
    test_dl = DataLoader(
        flickr,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    for _, token_ids, attn_mask in test_dl:
        tot_mat = torch.empty(size=(4, 0))
        tot_image = torch.empty(size=(0, 3, 224, 224))

        text_embed = text_enc(token_ids=token_ids, attn_mask=attn_mask)
        for image, _, _ in test_dl:
            img_embed = img_enc(image)
            mat = text_embed @ img_embed.T
            tot_mat = torch.cat([tot_mat, mat], dim=1)
            tot_image = torch.cat([tot_image, image], dim=0)

        tokenizer.decode(token_ids[0])
        torch.argmax(tot_mat, dim=1)
        grid = image_to_grid(image=tot_image, n_cols=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        grid.show()
