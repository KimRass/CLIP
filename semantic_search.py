import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from utils import l2_norm

from utils import load_config, get_device, image_to_grid
from tokenizer import get_tokenizer
from flickr import FlickrDataset, DataCollatorForDynamicPadding, encode
from train import get_clip

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

if __name__ == "__main__":
    max_len = 128
    clip = get_clip(config=CONFIG, max_len=max_len, device=DEVICE)
    img_enc = clip.img_enc
    text_enc = clip.text_enc

    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/epoch_128.pth"
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    img_enc.load_state_dict(state_dict["image_encoder"])
    text_enc.load_state_dict(state_dict["text_encoder"])
    temp = state_dict["temperature"]

    tokenizer = get_tokenizer()
    test_ds = FlickrDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k_subset", tokenizer=tokenizer, max_len=max_len,
    )
    collator = DataCollatorForDynamicPadding(tokenizer=tokenizer)
    test_dl = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )

    # query = "A German Shepherd chases another with a stick in his mouth ."
    query = "A group of people paddle their blue inflatable raft down the rapids ."
    token_ids = encode(query, tokenizer=tokenizer, max_len=max_len)
    attn_mask = [1] * len(token_ids)
    token_ids = torch.as_tensor(token_ids)[None, ...]
    attn_mask = torch.as_tensor(attn_mask)[None, ...]
    text_embed = text_enc(token_ids=token_ids, attn_mask=attn_mask)
    text_embed = l2_norm(text_embed)

    tot_sim = torch.empty(size=(0,))
    tot_image = torch.empty(size=(0, 3, 224, 224))
    for image, _, _ in test_dl:
        img_embed = img_enc(image)
        img_embed = l2_norm(img_embed)
        sim_mat = (text_embed @ img_embed.T)

        tot_sim = torch.cat([tot_sim, sim_mat[0]], dim=0)
        tot_image = torch.cat([tot_image, image], dim=0)

    print(tot_sim)
    print(torch.max(tot_sim, dim=0))
    # print(torch.argmax(tot_sim, dim=0).item())
    tot_image[0, 0, 0, 0]
    grid = image_to_grid(image=tot_image, n_cols=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    grid.show()
