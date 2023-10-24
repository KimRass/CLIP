import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import DistilBertTokenizerFast
import argparse
from time import time
from pathlib import Path
import wandb

from utils import load_config, get_device, get_elapsed_time
from flickr import Flickr8kDataset, DataCollatorForDynamicPadding
from clip import CLIP
from tokenizer import load_tokenizer

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

PARENT_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    # "We use a very large minibatch size of 32,768."
    parser.add_argument("--batch_size", type=int, required=False, default=32_768)
    parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


def get_clip(config, device):
    clip = CLIP(
        img_size=config["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        patch_size=config["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
        img_n_layers=config["ARCHITECTURE"]["IMG_ENC"]["N_LAYERS"],
        img_n_heads=config["ARCHITECTURE"]["IMG_ENC"]["N_HEADS"],
        img_hidden_dim=config["ARCHITECTURE"]["IMG_ENC"]["HIDDEN_DIM"],
        img_mlp_dim=config["ARCHITECTURE"]["IMG_ENC"]["MLP_DIM"],
        vocab_size=config["ARCHITECTURE"]["TEXT_ENC"]["VOCAB_SIZE"],
        max_len=config["ARCHITECTURE"]["TEXT_ENC"]["MAX_LEN"],
        text_n_layers=config["ARCHITECTURE"]["TEXT_ENC"]["N_LAYERS"],
        text_n_heads=config["ARCHITECTURE"]["TEXT_ENC"]["N_HEADS"],
        text_hidden_dim=config["ARCHITECTURE"]["TEXT_ENC"]["HIDDEN_DIM"],
        text_mlp_dim=config["ARCHITECTURE"]["TEXT_ENC"]["MLP_DIM"],
        embed_dim=config["ARCHITECTURE"]["EMBED_DIM"],
    ).to(device)
    clip.train()
    return clip


def train_single_step(image, token_ids, attn_mask, clip, optim, scaler):
    image = image.to(DEVICE)
    token_ids = token_ids.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    # "Mixed-precision was used to accelerate training and save memory."
    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
        enabled=True,
    ):
        img_loss, text_loss = clip.get_losses(image=image, token_ids=token_ids, attn_mask=attn_mask)
        tot_loss = (img_loss + text_loss) / 2

    optim.zero_grad()
    if DEVICE.type == "cuda" and scaler is not None:
        scaler.scale(tot_loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        tot_loss.backward()
        optim.step()

    # "The learnable temperature parameter was clipped to prevent scaling the logits by more than 100
    # which we found necessary to prevent training instability."
    # with torch.no_grad():
    #     clip.temp.clamp_(max=100)
    return img_loss, text_loss


def save_checkpoint(epoch, clip, optim, scaler, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "image_encoder": clip.img_enc.state_dict(),
        "text_encoder": clip.text_enc.state_dict(),
        "temperature": clip.temp.item(),
        "optimizer": optim.state_dict(),
    }
    if scaler is not None:
        state_dict["scaler"] = scaler.state_dict()
    torch.save(state_dict, str(save_path))
    # wandb.save(str(save_path), base_path=Path(save_path).parent)


if __name__ == "__main__":
    args = get_args()

    wandb.init(project="CLIP", resume=args.run_id)
    wandb.config.update(CONFIG, allow_val_change=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    flickr = Flickr8kDataset(data_dir=args.data_dir, tokenizer=tokenizer)
    collator = DataCollatorForDynamicPadding(tokenizer=tokenizer)
    train_dl = DataLoader(
        flickr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    clip = get_clip(config=CONFIG, device=DEVICE)

    # "We use the Adam optimizer with decoupled weight decay regularization (Loshchilov & Hutter, 2017) applied to all
    # weights that are not gains or biases, and decay the learning rate using a cosine schedule."
    optim = AdamW(
        clip.parameters(),
        lr=CONFIG["TRAINING"]["LR"],
        betas=(CONFIG["OPTIMIZER"]["BETA1"], CONFIG["OPTIMIZER"]["BETA2"]),
        weight_decay=CONFIG["OPTIMIZER"]["WEIGHT_DECAY"],
    )

    scaler = GradScaler() if DEVICE.type == "cuda" else None

    init_epoch = 0
    for epoch in range(init_epoch + 1, CONFIG["TRAINING"]["N_EPOCHS"] + 1):
        start_time = time()
        accum_img_loss = 0
        accum_text_loss = 0
        for step, (image, token_ids, attn_mask) in enumerate(train_dl, start=1):
            img_loss, text_loss = train_single_step(
                image=image,
                token_ids=token_ids,
                attn_mask=attn_mask,
                clip=clip,
                optim=optim,
                scaler=scaler,
            )
            accum_img_loss += img_loss.item()
            accum_text_loss += text_loss.item()

        accum_img_loss /= len(train_dl)
        accum_text_loss /= len(train_dl)

        msg = f"[ {get_elapsed_time(start_time)} ]"
        msg += f"""[ {epoch}/{CONFIG["TRAINING"]["N_EPOCHS"]} ]"""
        msg += f"""[ Image loss: {accum_img_loss:.4f} ]"""
        msg += f"""[ Text loss: {accum_text_loss:.4f} ]"""
        msg += f"""[ Temperature: {clip.temp.item():.4f} ]"""
        print(msg)

        wandb.log(
            {
                "Image loss": accum_img_loss,
                "Text loss": accum_text_loss,
                "Temperature": clip.temp.item(),
            },
            step=epoch,
        )

        if epoch == 10:
            save_checkpoint(
                epoch=epoch,
                clip=clip,
                optim=optim,
                scaler=scaler,
                save_path=PARENT_DIR/f"checkpoints/epoch_{epoch}.pth",
            )
