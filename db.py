import torch
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import faiss

from utils import load_config, get_device, image_to_grid
from flickr import ImageDataset
from semantic_search import init_faiss_index, get_encoders_from_checkpoint

CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")

DEVICE = get_device()

faiss_idx = init_faiss_index(256)

max_len = 128
ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/clip_flickr_200.pth"
img_enc, text_enc = get_encoders_from_checkpoint(
    ckpt_path, config=CONFIG, max_len=max_len, device=DEVICE,
)
img_enc.eval()
text_enc.eval()

db_path = "/Users/jongbeomkim/Desktop/workspace/CLIP/db/flickr.db"
conn = duckdb.connect(database=db_path, read_only=False)

create_sql = """
CREATE TABLE flickr(
    img_id INTEGER PRIMARY KEY,
    img_path TEXT
)
"""
conn.sql(create_sql)

ds = ImageDataset(data_dir="/Users/jongbeomkim/Documents/datasets/flickr30k", img_size=224)
for idx, (img_path, image) in enumerate(iter(ds), start=1):
    # img_embed = img_enc(image[None, :])
    # img_embed = img_embed.detach().cpu().numpy()
    xb = torch.randn(1, 256).numpy()
    faiss.normalize_L2(xb)
    indices = np.array([idx, ])
    faiss_idx.add_with_ids(xb, indices)

    insert_sql = f"""
    INSERT INTO flickr VALUES({idx + 100}, {Path(img_path).stem})
    """
    conn.sql(insert_sql)
conn.table("flickr")
