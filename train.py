import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
import argparse
import wandb

from utils import load_config
from flickr import Flickr8kDataset
from tokenizer import load_tokenizer
from clip import CLIP

CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--resume_from", type=str, required=False)

    args = parser.parse_args()
    return args


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

flickr = Flickr8kDataset(
    data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k",
    tokenizer=tokenizer,
    max_len=64,
)
train_dl = DataLoader(
    flickr,
    # batch_size=args.batch_size,
    batch_size=4,
    shuffle=True,
    # num_workers=args.n_cpus,
    num_workers=1,
    pin_memory=True,
    drop_last=True,
)
image, token_ids = next(iter(train_dl))

clip = CLIP(
    img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
    patch_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
    max_len=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["MAX_LEN"],
    n_heads=CONFIG["ARCHITECTURE"]["N_HEADS"],
    n_layers=CONFIG["ARCHITECTURE"]["N_LAYERS"],
    img_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_DIM"],
    text_dim=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["TEXT_DIM"],
    embed_dim=256,
    mlp_dim=CONFIG["ARCHITECTURE"]["MLP_DIM"],
)
# batch_size = 4
# image = torch.randn((batch_size, 3, 224, 224))
# token_ids = torch.randint(100, size=(batch_size, 32))
loss = clip.get_loss(image=image, token_ids=token_ids)
