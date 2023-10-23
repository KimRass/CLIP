import torch
import yaml
from datetime import timedelta
from time import time
from PIL import Image
from pathlib import Path

from clip import CLIP


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(image).save(str(path), quality=100)


def get_clip(config, device):
    clip = CLIP(
        img_size=config["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        patch_size=config["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
        img_n_layers=config["ARCHITECTURE"]["IMG_ENC"]["N_LAYERS"],
        img_n_heads=config["ARCHITECTURE"]["IMG_ENC"]["N_HEADS"],
        img_hidden_dim=config["ARCHITECTURE"]["IMG_ENC"]["HIDDEN_DIM"],
        img_mlp_dim=config["ARCHITECTURE"]["IMG_ENC"]["MLP_DIM"],
        vocab_size=config["ARCHITECTURE"]["TEXT_ENC"]["VOCAB_SIZE"],
        max_len=config["ARCHITECTURE"]["TEXT_ENC"]["MAX_LEN"],
        text_n_layers=config["ARCHITECTURE"]["TEXT_ENC"]["N_LAYERS"],
        text_n_heads=config["ARCHITECTURE"]["TEXT_ENC"]["N_HEADS"],
        text_hidden_dim=config["ARCHITECTURE"]["TEXT_ENC"]["HIDDEN_DIM"],
        text_mlp_dim=config["ARCHITECTURE"]["TEXT_ENC"]["MLP_DIM"],
        embed_dim=config["ARCHITECTURE"]["EMBED_DIM"],
    ).to(device)
    clip.train()
    return clip
