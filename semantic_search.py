import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from utils import l2_norm
import faiss

from utils import load_config, get_device, image_to_grid
from tokenizer import get_tokenizer
from flickr import FlickrDataset, DataCollatorForDynamicPadding, encode
from train import get_clip

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()


def add_to_faiss_index(faiss_idx, dl, img_enc):
    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB")
    for image, _, _ in dl:
        img_embed = img_enc(image)
        faiss_idx.add(img_embed.detach().cpu().numpy())
    print(f"There are {faiss_idx.ntotal:,} vectors in total in the DB")


def text_to_embedding(text, text_enc):
    token_ids = encode(text, tokenizer=tokenizer, max_len=max_len)
    attn_mask = [1] * len(token_ids)

    token_ids = torch.as_tensor(token_ids)[None, ...]
    attn_mask = torch.as_tensor(attn_mask)[None, ...]
    text_embed = text_enc(token_ids=token_ids, attn_mask=attn_mask)
    return text_embed


if __name__ == "__main__":
    faiss_idx = faiss.IndexFlatL2(CONFIG["ARCHITECTURE"]["EMBED_DIM"])

    max_len = 128
    clip = get_clip(config=CONFIG, max_len=max_len, device=DEVICE)
    clip.eval()
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/epoch_128.pth"
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])
    temp = state_dict["temperature"]

    tokenizer = get_tokenizer()
    test_ds = FlickrDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k_subset",
        tokenizer=tokenizer,
        max_len=max_len,
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
    )
    collator = DataCollatorForDynamicPadding(tokenizer=tokenizer)
    test_dl = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )



    # tot_image = torch.empty(size=(0, 3, 224, 224))
        # tot_image = torch.cat([tot_image, image], dim=0)

    # query = "A German Shepherd chases another with a stick in his mouth ."
    # query = "A group of people paddle their blue inflatable raft down the rapids ."
    query = "A little blonde boy is petting a resting tiger ."
    query_text_embed = text_to_embedding(query, text_enc=text_enc)
        
    distances, indices = faiss_idx.search(text_embed.detach().cpu().numpy(), 1)

    rank = 1
    trg_image = tot_image[I[0][rank - 1]]
    grid = image_to_grid(image=trg_image.unsqueeze(0), n_cols=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    grid.show()
