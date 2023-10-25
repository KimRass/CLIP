import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from pathlib import Path
# import torch.nn.functional as F
# import numpy as np
# from PIL import Image
# import argparse
# from time import time
# import wandb

from utils import load_config, get_device, image_to_grid
from flickr import Flickr8kDataset, DataCollatorForDynamicPadding, encode
from train import get_clip

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

if __name__ == "__main__":
    clip = get_clip(config=CONFIG, device=DEVICE)
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/epoch_32.pth"
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])
    temp = state_dict["temperature"]


    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    flickr = Flickr8kDataset(data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k_subset", tokenizer=tokenizer)
    collator = DataCollatorForDynamicPadding(tokenizer=tokenizer)
    test_dl = DataLoader(
        flickr,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )

    query = "A blonde horse and a blonde girl in a black sweatshirt are staring at a fire in a barrel ."
    token_ids = encode(query, tokenizer=tokenizer)
    attn_mask = [1] * len(token_ids)
    token_ids = torch.as_tensor(token_ids)[None, ...]
    attn_mask = torch.as_tensor(attn_mask)[None, ...]
    text_embed = text_enc(token_ids=token_ids, attn_mask=attn_mask)

    tot_sim = torch.empty(size=(0,))
    tot_image = torch.empty(size=(0, 3, 224, 224))
    for image, _, _ in test_dl:
        # pass
        img_embed = img_enc(image)
        mat = (text_embed @ img_embed.T) / temp
        tot_sim = torch.cat([tot_sim, mat[0]], dim=0)
        tot_image = torch.cat([tot_image, image], dim=0)
    # tot_sim
    # text_embed.shape, img_embed.shape, sim.shape
    print(tot_sim)
    print(torch.argmax(tot_sim, dim=0).item())
    grid = image_to_grid(image=tot_image, n_cols=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    grid.show()
