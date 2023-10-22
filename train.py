import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import DistilBertTokenizerFast
import argparse
from time import time
from tqdm import tqdm
from pathlib import Path
import wandb

from utils import load_config, get_device, get_elapsed_time
from flickr import Flickr8kDataset, DataCollatorForDynamicPadding
from tokenizer import load_tokenizer
from clip import CLIP

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    # "We use a very large minibatch size of 32,768."
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
    )
    collator = DataCollatorForDynamicPadding(tokenizer=tokenizer)
    train_dl = DataLoader(
        flickr,
        batch_size=args.batch_size,
        # batch_size=4,
        shuffle=True,
        num_workers=args.n_cpus,
        # num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    clip = CLIP(
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        patch_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
        img_n_layers=CONFIG["ARCHITECTURE"]["IMG_ENC"]["N_LAYERS"],
        img_n_heads=CONFIG["ARCHITECTURE"]["IMG_ENC"]["N_HEADS"],
        img_hidden_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["HIDDEN_DIM"],
        img_mlp_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["MLP_DIM"],
        vocab_size=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["VOCAB_SIZE"],
        max_len=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["MAX_LEN"],
        text_n_layers=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["N_LAYERS"],
        text_n_heads=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["N_HEADS"],
        text_hidden_dim=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["HIDDEN_DIM"],
        text_mlp_dim=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["MLP_DIM"],
        embed_dim=CONFIG["ARCHITECTURE"]["EMBED_DIM"],
    ).to(DEVICE)
    clip.train()

    # "We use the Adam optimizer with decoupled weight decay regularization (Loshchilov & Hutter, 2017) applied to all
    # weights that are not gains or biases, and decay the learning rate using a cosine schedule."
    optim = AdamW(
        clip.parameters(),
        lr=CONFIG["TRAINING"]["LR"],
        betas=(CONFIG["OPTIMIZER"]["BETA1"], CONFIG["OPTIMIZER"]["BETA2"]),
        weight_decay=CONFIG["OPTIMIZER"]["WEIGHT_DECAY"],
    )

    if DEVICE.type == "cuda":
        scaler = GradScaler()

    init_epoch = 0
    start_time = time()
    for epoch in range(init_epoch + 1, CONFIG["TRAINING"]["N_EPOCHS"] + 1):
        for step, (image, token_ids, attn_mask) in enumerate(train_dl, start=1):
            image = image.to(DEVICE)
            token_ids = token_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)

            # "Mixed-precision was used to accelerate training and save memory."
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
                enabled=True,
            ):
                img_loss, text_loss = clip.get_loss(image=image, token_ids=token_ids, attn_mask=attn_mask)
                tot_loss = img_loss + text_loss

            optim.zero_grad()
            if DEVICE.type == "cuda":
                scaler.scale(tot_loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                tot_loss.backward()
                optim.step()

            # "The learnable temperature parameter was clipped to prevent scaling the logits by more than 100
            # which we found necessary to prevent training instability."
            with torch.no_grad():
                clip.temp.clamp_(max=100)

            # if step % 10 == 0:
        msg = f"[ {get_elapsed_time(start_time)} ]"
        msg += f"""[ {epoch}/{CONFIG["TRAINING"]["N_EPOCHS"]} ]"""
        msg += f"""[ {step}/{len(train_dl)} ]"""
        msg += f"""[ Image loss: {img_loss:.4f} ]"""
        msg += f"""[ Text loss: {text_loss:.4f} ]"""
        # msg += f"""[ Temperature: {clip.temp.data} ]"""
        print(msg)

        start_time = time()
