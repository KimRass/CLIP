import torch
from transformers import DistilBertTokenizerFast
import argparse
from time import time
from pathlib import Path
import wandb

from utils import load_config, get_device, get_elapsed_time
from flickr import Flickr8kDataset, DataCollatorForDynamicPadding
from train import get_clip

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

if __name__ == "__main__":
    clip = get_clip(config=CONFIG, device=DEVICE)
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])