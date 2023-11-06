import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import faiss
from tqdm import tqdm

from utils import load_config, get_device, image_to_grid
from tokenizer import get_tokenizer
# from imagenet1k import get_imagenet1k_dataset, _class_name_to_prompt
from imagenet1k import ImageNet1kDataset
from train import get_clip
from semantic_search import load_faiss_index, save_faiss_index, _add_texts_to_faiss_index
from flickr import encode, pad, get_attention_mask


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    # parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--k", type=int, required=False, default=5)
    # parser.add_argument("--torch_compile", action="store_true", required=False)
    # parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


def get_imagenet1k_classes(json_path):
    with open(json_path, mode="r") as f:
        classes = json.load(f)
    classes = {int(k): v[1] for k, v in classes.items()}
    return classes


def get_number_of_correct_predictions(gt, nns, k):
    arr_gt = gt.detach().cpu().numpy()
    eq = np.equal(nns, np.repeat(arr_gt[:, None], repeats=k, axis=1))
    # acc = corr.sum(axis=1).astype("float").mean()
    return eq.sum(axis=1).sum()


if __name__ == "__main__":
    # CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    clip = get_clip(config=CONFIG, max_len=args.max_len, device=DEVICE)
    clip.eval()
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/clip_flickr.pth"
    # state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])

    json_path = "/Users/jongbeomkim/Desktop/workspace/CLIP/imagenet1k_classes.json"
    CLASSES = get_imagenet1k_classes(json_path)
    tokenizer = get_tokenizer()

    faiss_idx = faiss.IndexFlatIP(clip.embed_dim) # `IP`: Inner Product
    faiss_idx = faiss.IndexIDMap2(faiss_idx)
    
    _add_texts_to_faiss_index(
        faiss_idx=faiss_idx,
        idx2text=CLASSES,
        text_enc=text_enc,
        tokenizer=tokenizer,
        max_len=clip.max_len,
    )

    ds = ImageNet1kDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/imagenet-mini",
        tokenizer=None,
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        split="train",
    )
    # ds = get_imagenet1k_dataset(data_dir=args.data_dir, img_size=clip.img_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=False,
        drop_last=False,
    )
    n_corrs = 0
    for image, gt in tqdm(dl):
        image = image.to(DEVICE)
        gt = gt.to(DEVICE)

        query_embed = img_enc(image)
        xq = query_embed.detach().cpu().numpy()
        faiss.normalize_L2(xq)
        _, nns = faiss_idx.search(xq, k=args.k)
        corr = get_number_of_correct_predictions(gt=gt, nns=nns, k=args.k)
        n_corrs += corr
    acc = n_corrs / len(dl)
    print(acc)
