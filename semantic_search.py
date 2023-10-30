# References:
    # https://github.com/facebookresearch/faiss/wiki/Getting-started

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm

from utils import load_config, get_device, image_to_grid
from tokenizer import get_tokenizer
from flickr import ImageDataset, encode
from train import get_clip

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()


def _add_to_faiss_index(faiss_idx, dl, img_enc):
    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB")
    img_embeds = list()
    indices = list()
    for idx, image in enumerate(tqdm(dl)):
        img_embed = img_enc(image)
        img_embeds.append(img_embed.detach().cpu().numpy())
        indices.append(idx)
    xb = np.concatenate(img_embeds)
    faiss.normalize_L2(xb)
    faiss_idx.add_with_ids(xb, np.arange(xb.shape[0]))
    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB")


def save_faiss_index(dim, dl, img_enc, save_path):
    faiss_idx = faiss.IndexFlatIP(dim) # Inner product
    faiss_idx = faiss.IndexIDMap2(faiss_idx)
    _add_to_faiss_index(faiss_idx=faiss_idx, dl=dl, img_enc=img_enc)
    faiss.write_index(faiss_idx, save_path)


def text_to_embedding(text, text_enc):
    token_ids = encode(text, tokenizer=tokenizer, max_len=max_len)
    attn_mask = [1] * len(token_ids)

    token_ids = torch.as_tensor(token_ids)[None, ...]
    attn_mask = torch.as_tensor(attn_mask)[None, ...]
    text_embed = text_enc(token_ids=token_ids, attn_mask=attn_mask)
    return text_embed


if __name__ == "__main__":
    max_len = 128
    clip = get_clip(config=CONFIG, max_len=max_len, device=DEVICE)
    clip.eval()
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/clip_flickr_200.pth"
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])

    tokenizer = get_tokenizer()
    ds = ImageDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k",
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
    )
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    index_path = "/Users/jongbeomkim/Downloads/flickr8k.index"
    if Path(index_path).exists():
        faiss_idx = faiss.read_index(index_path)
    else:
        save_faiss_index(
            dim=CONFIG["ARCHITECTURE"]["EMBED_DIM"],
            dl=dl,
            img_enc=img_enc,
            save_path=index_path,
        )

    query = "The children are playing happily with water."
    query_text_embed = text_to_embedding(query, text_enc=text_enc)
    xq = query_text_embed.detach().cpu().numpy()
    faiss.normalize_L2(xq)
    distances, indices = faiss_idx.search(xq, 1)
    print(f"Cosine similarity: {distances[0][0]:.3f}")

    trg_image = ds[indices[0][0]]
    grid = image_to_grid(
        image=trg_image.unsqueeze(0), n_cols=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )
    grid.show()
