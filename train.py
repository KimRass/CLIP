import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.cuda.amp import GradScaler
import argparse
from time import time
from pathlib import Path
import wandb

from utils import load_config, get_device, get_elapsed_time
from tokenizer import get_tokenizer
from flickr import FlickrDataset, DataCollatorForDynamicPadding
from clip import CLIP
from loss import CLIPLoss
from evaluate import TopKAccuracy

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

PARENT_DIR = Path(__file__).resolve().parent
SAVE_DIR = PARENT_DIR/"checkpoints"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--flickr8k_dir", type=str, required=True)
    parser.add_argument("--flickr30k_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True) # "We train all models for 32 epochs."
    parser.add_argument("--n_cpus", type=int, required=True)
    # "We use a very large minibatch size of 32,768."
    parser.add_argument("--batch_size", type=int, required=False, default=32_768)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


def get_clip(config, max_len, device):
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
    ).to(device)
    clip.train()
    return clip


def train_single_step(image, token_ids, attn_mask, clip, crit, optim, scaler):
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
        img_loss, text_loss = crit(img_embed=img_embed, text_embed=text_embed)
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
    with torch.no_grad():
        clip.temp.clamp_(max=100)
    return img_loss, text_loss


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

        accum_acc += acc.item()
    avg_acc = accum_acc / len(val_dl)

    clip.train()
    return avg_acc


def save_checkpoint(epoch, clip, optim, scaler, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        # "epoch": epoch,
        "image_encoder": clip.img_enc.state_dict(),
        "text_encoder": clip.text_enc.state_dict(),
        # "temperature": clip.temp.item(),
        # "optimizer": optim.state_dict(),
    }
    # if scaler is not None:
    #     state_dict["scaler"] = scaler.state_dict()
    torch.save(state_dict, str(save_path))
    # wandb.save(str(save_path), base_path=Path(save_path).parent)
    print("Saved the checkpoint.")


def get_dls(flickr8k_dir, flickr30k_dir, tokenizer, max_len, batch_size, n_cpus):
    ds1 = FlickrDataset(data_dir=flickr8k_dir, tokenizer=tokenizer, max_len=max_len)
    ds2 = FlickrDataset(data_dir=flickr30k_dir, tokenizer=tokenizer, max_len=max_len)
    ds = ConcatDataset(ds1, ds2)
    train_size = round(len(ds) * 0.9)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
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
    return train_dl, val_dl


if __name__ == "__main__":
    args = get_args()

    run = wandb.init(project="CLIP", resume=args.run_id)
    if args.run_id is None:
        args.run_id = wandb.run.name
    wandb.config.update(CONFIG, allow_val_change=True)
    print(wandb.config)

    tokenizer = get_tokenizer()
    train_dl, val_dl = get_dls(
        flickr8k_dir=args.flickr8k_dir,
        flickr30k_dir=args.flickr30k_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        n_cpus=args.n_cpus,
    )

    clip = get_clip(config=CONFIG, max_len=args.max_len, device=DEVICE)
    crit = CLIPLoss(batch_size=args.batch_size, temp=clip.temp)
    metric = TopKAccuracy(k=1, batch_size=args.batch_size)

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
    max_avg_acc = 0
    for epoch in range(init_epoch + 1, args.n_epochs + 1):
        start_time = time()
        accum_img_loss = 0
        accum_text_loss = 0
        for image, token_ids, attn_mask in train_dl:
            img_loss, text_loss = train_single_step(
                image=image,
                token_ids=token_ids,
                attn_mask=attn_mask,
                clip=clip,
                crit=crit,
                optim=optim,
                scaler=scaler,
            )
            accum_img_loss += img_loss.item()
            accum_text_loss += text_loss.item()

        avg_img_loss = accum_img_loss / len(train_dl)
        avg_text_loss = accum_text_loss / len(train_dl)

        avg_acc = validate(val_dl=val_dl, clip=clip, metric=metric)

        msg = f"[ {get_elapsed_time(start_time)} ]"
        msg += f"""[ {epoch}/{args.n_epochs} ]"""
        msg += f"""[ Image loss: {avg_img_loss:.4f} ]"""
        msg += f"""[ Text loss: {avg_text_loss:.4f} ]"""
        msg += f"""[ Temperature: {clip.temp.item():.4f} ]"""
        msg += f"""[ Accuracy: {avg_acc:.4f} ]"""
        print(msg)

        wandb.log(
            {
                "Image loss": avg_img_loss,
                "Text loss": avg_text_loss,
                "Temperature": clip.temp.item(),
                "Accuracy": avg_acc,
            },
            step=epoch,
        )

    if avg_acc > max_avg_acc:
        save_checkpoint(
            epoch=epoch,
            clip=clip,
            optim=optim,
            scaler=scaler,
            save_path=SAVE_DIR/"epoch_{epoch}.pth",
        )
