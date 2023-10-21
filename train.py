import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import DistilBertTokenizerFast
import argparse
from time import time
from tqdm import tqdm
from pathlib import Path
import wandb

from utils import load_config, get_device, get_elapsed_time
from flickr import Flickr8kDataset
from tokenizer import load_tokenizer
from clip import CLIP

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=32_768)
    parser.add_argument("--resume_from", type=str, required=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # wandb.init(project="CLIP")
    # wandb.config.update(CONFIG)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    flickr = Flickr8kDataset(
        # data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k",
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_len=64,
    )
    train_dl = DataLoader(
        flickr,
        batch_size=args.batch_size,
        # batch_size=4,
        shuffle=True,
        num_workers=args.n_cpus,
        # num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    clip = CLIP(
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        patch_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
        max_len=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["MAX_LEN"],
        n_heads=CONFIG["ARCHITECTURE"]["N_HEADS"],
        n_layers=CONFIG["ARCHITECTURE"]["N_LAYERS"],
        img_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_DIM"],
        text_dim=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["TEXT_DIM"],
        embed_dim=CONFIG["ARCHITECTURE"]["EMBED_DIM"],
        mlp_dim=CONFIG["ARCHITECTURE"]["MLP_DIM"],
    )

    # "For the Vision Transformers we train a ViT-B/32, a ViT-B/16, and a ViT-L/14. We use the Adam optimizer (Kingma & Ba, 2014) with decoupled weight decay regularization (Loshchilov & Hutter, 2017) applied to all weights that are not gains
    # or biases, and decay the learning rate using a cosine schedule (Loshchilov & Hutter, 2016). Initial"
    optim = Adam(
        clip.parameters(),
        lr=CONFIG["TRAINING"]["LR"],
        betas=(CONFIG["OPTIMIZER"]["BETA1"], CONFIG["OPTIMIZER"]["BETA2"]),
        weight_decay=CONFIG["OPTIMIZER"]["WEIGHT_DECAY"],
    )

    scaler = GradScaler()

    init_epoch = 0
    for epoch in range(init_epoch + 1, CONFIG["TRAINING"]["N_EPOCHS"] + 1):
        start_time = time()
        for step, (image, token_ids) in enumerate(tqdm(train_dl), start=1):
            image = image.to(DEVICE)
            token_ids = token_ids.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
                enabled=True,
            ):
                img_loss, text_loss = clip.get_loss(image=image, token_ids=token_ids)
                tot_loss = img_loss + text_loss

            optim.zero_grad()
            scaler.scale(tot_loss).backward()
            scaler.step(optim)

            scaler.update()

            if step % 1 == 0:
                msg = f"[ {get_elapsed_time(start_time)} ]"
                msg += f"""[ {epoch}/{CONFIG["TRAINING"]["N_EPOCHS"]} ]"""
                msg += f"""[ {step}/{len(train_dl)} ]"""
                msg += f"""[ Image loss: {img_loss:.4f} ]"""
                msg += f"""[ Text loss: {text_loss:.4f} ]"""
                print(msg)