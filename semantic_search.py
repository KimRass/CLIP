# References:
    # https://github.com/facebookresearch/faiss/wiki/Getting-started

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image

from utils import load_config, get_device, image_to_grid, get_tokenizer, get_parent_dir
from data.data_augmentation import get_val_transformer
from data.flickr import ImageDataset, encode, pad, get_attention_mask
from train_clip import get_clip_model


def get_encoders_from_checkpoint(ckpt_path, config, max_len, device):
    clip = get_clip_model(config=config, max_len=max_len, device=device)
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    state_dict = torch.load(ckpt_path, map_location=device)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])
    return img_enc, text_enc


def add_texts_to_faiss_index(faiss_idx, idx2text, text_enc, tokenizer, max_len):
    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB.")

    text_embeds = list()
    for text in tqdm(idx2text.values()):
        token_ids = encode(text, tokenizer=tokenizer, max_len=max_len)
        token_ids = pad(token_ids=token_ids, max_len=max_len, pad_id=tokenizer.pad_token_id)
        token_ids = torch.tensor(token_ids)
        attn_mask = get_attention_mask(token_ids=token_ids, pad_id=tokenizer.pad_token_id)

        text_embed = text_enc(token_ids=token_ids, attn_mask=attn_mask)
        text_embeds.append(text_embed.detach().cpu().numpy())
    xb = np.concatenate(text_embeds)
    faiss.normalize_L2(xb)

    indices = np.array(list(idx2text.keys()))
    faiss_idx.add_with_ids(xb, indices)

    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB.")


def _add_images_to_faiss_index(faiss_idx, dl, img_enc):
    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB.")

    img_embeds = list()
    # indices = list()
    for idx, image in enumerate(tqdm(dl)):
        img_embed = img_enc(image)
        img_embeds.append(img_embed.detach().cpu().numpy())
        # indices.append(idx)
    xb = np.concatenate(img_embeds)
    faiss.normalize_L2(xb)
    indices = np.arange(xb.shape[0])
    faiss_idx.add_with_ids(xb, indices)

    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB.")


def load_faiss_index(index_path):
    if Path(index_path).exists():
        faiss_idx = faiss.read_index(index_path)
        return faiss_idx
    else:
        print(f"No file; '{index_path}'!")


def init_faiss_index(dim):
    faiss_idx = faiss.IndexFlatIP(dim) # `IP`: Inner Product
    faiss_idx = faiss.IndexIDMap2(faiss_idx)
    return faiss_idx


def save_faiss_index(dim, dl, img_enc, save_path):
    faiss_idx = init_faiss_index(dim)
    _add_images_to_faiss_index(faiss_idx=faiss_idx, dl=dl, img_enc=img_enc)
    faiss.write_index(faiss_idx, save_path)


def _text_to_embedding(text, text_enc):
    token_ids = encode(text, tokenizer=tokenizer, max_len=max_len)
    attn_mask = [1] * len(token_ids)

    token_ids = torch.as_tensor(token_ids)[None, ...]
    attn_mask = torch.as_tensor(attn_mask)[None, ...]
    text_embed = text_enc(token_ids=token_ids, attn_mask=attn_mask)
    return text_embed


def _image_to_embedding(img_path, img_enc, img_size):
    image = Image.open(img_path)
    image = get_val_transformer(img_size)(image)
    img_embed = img_enc(image.unsqueeze(0))
    return img_embed


# def perform_semantic_search(query, faiss_idx, k):
#     if Path(query).exists():
#         query_embed = _image_to_embedding(
#             img_path=img_path, img_enc=img_enc, img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
#         )
#     else:
#         query_embed = _text_to_embedding(query, text_enc=text_enc)

#     xq = query_embed.detach().cpu().numpy()
#     faiss.normalize_L2(xq)
#     dists, nns = faiss_idx.search(xq, k)
#     return dists, nns


def perform_semantic_search(query_embed, faiss_idx, k):
    xq = query_embed.detach().cpu().numpy()
    faiss.normalize_L2(xq)
    dists, nns = faiss_idx.search(xq, k)
    return dists, nns


def index_to_image(idx, ds):
    trg_image = ds[idx[0][0]]
    grid = image_to_grid(
        image=trg_image.unsqueeze(0), n_cols=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )
    return grid


if __name__ == "__main__":
    PARENT_DIR = get_parent_dir()
    CONFIG = load_config(PARENT_DIR/"configs/flickr.yaml")

    DEVICE = get_device()

    max_len = 128
    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/clip_flickr_200.pth"
    img_enc, text_enc = get_encoders_from_checkpoint(
        ckpt_path, config=CONFIG, max_len=max_len, device=DEVICE,
    )
    img_enc.eval()
    text_enc.eval()

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
    faiss_idx = load_faiss_index(index_path)
    save_faiss_index(
        dim=CONFIG["ARCHITECTURE"]["EMBED_DIM"],
        dl=dl,
        img_enc=img_enc,
        save_path=index_path,
    )

    query = "The children are playing happily with water."
    query_text_embed = _text_to_embedding(query, text_enc=text_enc)

    img_path = "/Users/jongbeomkim/Desktop/workspace/Gatys-et-al./examples/content_img2.jpg"
    query_img_embed = _image_to_embedding(
        img_path=img_path, img_enc=img_enc, img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
    )

    dists, nns = perform_semantic_search(query=img_path, faiss_idx=faiss_idx)
    print(f"Cosine similarity: {dists[0][0]:.3f}")

    out_image = index_to_image(idx=nns, ds=ds)
    out_image.show()
