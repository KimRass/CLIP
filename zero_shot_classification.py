import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import faiss

from utils import load_config, get_device, image_to_grid
from tokenizer import get_tokenizer
from imagenet1k import ImageNet1kDataset, get_imagenet1k_dataset, _class_name_to_prompt
from train import get_clip
from semantic_search import load_faiss_index, save_faiss_index
from flickr import encode, pad


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    # "We use a very large minibatch size of 32,768."
    parser.add_argument("--batch_size", type=int, required=False, default=32_768)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--torch_compile", action="store_true", required=False)
    parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


def get_imagenet1k_classes(json_path):
    with open(json_path, mode="r") as f:
        classes = json.load(f)
    classes = {int(k): v[1] for k, v in classes.items()}
    return classes


if __name__ == "__main__":
    # CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    max_len = 128
    clip = get_clip(config=CONFIG, max_len=max_len, device=DEVICE)
    clip.eval()
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    # ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/clip_flickr_200.pth"
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])

    json_path = "/Users/jongbeomkim/Desktop/workspace/CLIP/imagenet1k_classes.json"
    CLASSES = get_imagenet1k_classes(json_path)
    prompts = [(k, _class_name_to_prompt(v)) for k, v in CLASSES.items()]
    tokenizer = get_tokenizer()

    faiss_idx = faiss.IndexFlatIP(768) # Inner product
    faiss_idx = faiss.IndexIDMap2(faiss_idx)

    text_embeds = list()
    gts = list()
    for gt, prompt in prompts:
        token_ids = encode(prompt, tokenizer=tokenizer, max_len=max_len)
        token_ids = pad(token_ids=token_ids, max_len=max_len, pad_id=0)
        text_embed = text_enc(token_ids)
        text_embeds.append(text_embed.detach().cpu().numpy())
    xb = np.concatenate(text_embeds)
    faiss.normalize_L2(xb)
    faiss_idx.add_with_ids(xb, np.concatenate(gts))

    # ds = ImageNet1kDataset(
    #     data_dir="/Users/jongbeomkim/Documents/datasets/imagenet-mini",
    #     tokenizer=None,
    #     img_size=28,
    #     split="train",
    # )
    ds = get_imagenet1k_dataset(args.data_dir)
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    for image, gt in dl:
        image = image.to(DEVICE)
        gt = gt.to(DEVICE)
