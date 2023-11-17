import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
import argparse
from time import time
from pathlib import Path
import wandb

from utils import (
    apply_seed,
    load_config,
    get_device,
    get_elapsed_time,
    modify_state_dict,
    get_tokenizer,
)
from flickr import FlickrDataset, DataCollatorForDynamicPadding
from data_augmentation import get_val_transformer
from clip import CLIP
from evaluate import CLIPTopKAccuracy

CONFIG = load_config(Path(__file__).parent/"configs/flickr.yaml")

DEVICE = get_device()

PARENT_DIR = Path(__file__).resolve().parent
SAVE_DIR = PARENT_DIR/"checkpoints"
WANDB_CKPT_PATH = SAVE_DIR/"checkpoint.tar"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--flickr8k_dir", type=str, required=True)
    parser.add_argument("--flickr30k_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True) # "We train all models for 32 epochs."
    parser.add_argument("--n_cpus", type=int, required=True)
    # "We use a very large minibatch size of 32,768."
    parser.add_argument("--batch_size", type=int, required=False, default=32_768)
    # "For computational efficiency, the max sequence length was capped at 76."
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--torch_compile", action="store_true", required=False)
    parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


def get_clip(config, batch_size, max_len, device, torch_compile=False):
    clip = CLIP(
        img_size=config["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        patch_size=config["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
        img_n_layers=config["ARCHITECTURE"]["IMG_ENC"]["N_LAYERS"],
        img_n_heads=config["ARCHITECTURE"]["IMG_ENC"]["N_HEADS"],
        img_hidden_dim=config["ARCHITECTURE"]["IMG_ENC"]["HIDDEN_DIM"],
        img_mlp_dim=config["ARCHITECTURE"]["IMG_ENC"]["MLP_DIM"],
        vocab_size=config["ARCHITECTURE"]["TEXT_ENC"]["VOCAB_SIZE"],
        max_len=max_len,
        text_n_layers=config["ARCHITECTURE"]["TEXT_ENC"]["N_LAYERS"],
        text_n_heads=config["ARCHITECTURE"]["TEXT_ENC"]["N_HEADS"],
        text_hidden_dim=config["ARCHITECTURE"]["TEXT_ENC"]["HIDDEN_DIM"],
        text_mlp_dim=config["ARCHITECTURE"]["TEXT_ENC"]["MLP_DIM"],
        embed_dim=config["ARCHITECTURE"]["EMBED_DIM"],
        batch_size=batch_size,
    ).to(device)
    if torch_compile:
        clip = torch.compile(clip)
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
        img_embed, text_embed = clip(image=image, token_ids=token_ids, attn_mask=attn_mask)
        loss = clip.get_loss(img_embed=img_embed, text_embed=text_embed)

    optim.zero_grad()
    if DEVICE.type == "cuda" and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()
    return loss


@torch.no_grad()
def validate(val_dl, clip, metric):
    clip.eval()

    accum_acc = 0
    for image, token_ids, attn_mask in val_dl:
        image = image.to(DEVICE)
        token_ids = token_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)

        img_embed, text_embed = clip(image=image, token_ids=token_ids, attn_mask=attn_mask)
        acc = metric(img_embed=img_embed, text_embed=text_embed)

        accum_acc += acc
    avg_acc = accum_acc / len(val_dl)

    clip.train()
    return avg_acc


def save_checkpoint(clip, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "image_encoder": modify_state_dict(clip.img_enc.state_dict()),
        "text_encoder": modify_state_dict(clip.text_enc.state_dict()),
    }
    torch.save(state_dict, str(save_path))
    print("Saved the checkpoint.")


def save_wandb_checkpoint(epoch, clip, optim, scaler, max_avg_acc, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "image_encoder": modify_state_dict(clip.img_enc.state_dict()),
        "text_encoder": modify_state_dict(clip.text_enc.state_dict()),
        "optimizer": optim.state_dict(),
        "max_average_accuracy": max_avg_acc,
    }
    if scaler is not None:
        state_dict["scaler"] = scaler.state_dict()
    torch.save(state_dict, str(save_path))
    wandb.save(str(save_path), base_path=Path(save_path).parent)


def get_dls(flickr8k_dir, flickr30k_dir, tokenizer, max_len, img_size, batch_size, n_cpus):
    ds1 = FlickrDataset(data_dir=flickr8k_dir, tokenizer=tokenizer, max_len=max_len, img_size=img_size)
    ds2 = FlickrDataset(data_dir=flickr30k_dir, tokenizer=tokenizer, max_len=max_len, img_size=img_size)
    ds = ds1 + ds2

    train_size = round(len(ds) * 0.9)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    val_ds.transformer = get_val_transformer

    collator = DataCollatorForDynamicPadding(tokenizer=tokenizer)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    print(f"Train set size: {train_size:,}, validation set size: {val_size:,}")
    return train_dl, val_dl


if __name__ == "__main__":
    apply_seed(CONFIG.SEED)

    args = get_args()

    run = wandb.init(project="CLIP", resume=args.run_id)
    if args.run_id is None:
        args.run_id = wandb.run.name
    wandb.config.update({"max_len": args.max_len, "run_id": args.run_id}, allow_val_change=True)
    wandb.config.update(CONFIG, allow_val_change=True)
    print(wandb.config)

    tokenizer = get_tokenizer()
    train_dl, val_dl = get_dls(
        flickr8k_dir=args.flickr8k_dir,
        flickr30k_dir=args.flickr30k_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        batch_size=args.batch_size,
        n_cpus=args.n_cpus,
    )

    clip = get_clip(
        config=CONFIG,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=DEVICE,
        torch_compile=args.torch_compile,
    )
    metric = CLIPTopKAccuracy(k=1, batch_size=args.batch_size)

    # "We use the Adam optimizer with decoupled weight decay regularization (Loshchilov & Hutter, 2017) applied to all
    # weights that are not gains or biases, and decay the learning rate using a cosine schedule."
    optim = AdamW(
        clip.parameters(),
        lr=CONFIG["TRAINING"]["LR"],
        betas=(CONFIG["OPTIMIZER"]["BETA1"], CONFIG["OPTIMIZER"]["BETA2"]),
        weight_decay=CONFIG["OPTIMIZER"]["WEIGHT_DECAY"],
    )

    scaler = GradScaler(enabled=True if DEVICE.type == "cuda" else False)

    ### Resume
    if wandb.run.resumed:
        state_dict = torch.load(str(WANDB_CKPT_PATH), map_location=DEVICE)
        init_epoch = state_dict["epoch"]
        clip.img_enc.load_state_dict(state_dict["image_encoder"])
        clip.text_enc.load_state_dict(state_dict["text_encoder"])
        optim.load_state_dict(state_dict["optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        max_avg_acc = state_dict["max_average_accuracy"]

        prev_ckpt_path = str(WANDB_CKPT_PATH)

        print(f"Resuming from epoch {init_epoch + 1}...")
    else:
        init_epoch = 0
        prev_ckpt_path = ".pth"
        max_avg_acc = 0

    for epoch in range(init_epoch + 1, args.n_epochs + 1):
        start_time = time()
        accum_loss = 0
        for image, token_ids, attn_mask in train_dl:
            loss = train_single_step(
                image=image,
                token_ids=token_ids,
                attn_mask=attn_mask,
                clip=clip,
                optim=optim,
                scaler=scaler,
            )
            accum_loss += loss.item()

        avg_loss = accum_loss / len(train_dl)

        avg_acc = validate(val_dl=val_dl, clip=clip, metric=metric)

        msg = f"[ {get_elapsed_time(start_time)} ]"
        msg += f"""[ {epoch}/{args.n_epochs} ]"""
        msg += f"""[ Loss: {avg_loss:.4f} ]"""
        msg += f"""[ Accuracy: {avg_acc:.4f} ]"""
        print(msg)

        wandb.log(
            {"Loss": avg_loss, "Validation accuracy": avg_acc},
            step=epoch,
        )

        if avg_acc > max_avg_acc:
            cur_ckpt_path = SAVE_DIR/f"{args.run_id}_epoch_{epoch}.pth"
            save_checkpoint(clip=clip, save_path=cur_ckpt_path)

            max_avg_acc = avg_acc
            Path(prev_ckpt_path).unlink(missing_ok=True)
            prev_ckpt_path = cur_ckpt_path

        save_wandb_checkpoint(
            epoch=epoch,
            clip=clip,
            optim=optim,
            scaler=scaler,
            max_avg_acc=max_avg_acc,
            save_path=WANDB_CKPT_PATH
        )
