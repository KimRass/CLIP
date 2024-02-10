import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import argparse
from torch.optim import Adam
from torch.cuda.amp import GradScaler
import time

from utils import get_parent_dir, get_config, get_elapsed_time, apply_seed
from model import LinearClassifier
from data.data_augmentation import get_train_transformer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def get_dls(data_dir, img_size, batch_size, n_cpus):
    transformer = get_train_transformer(img_size=img_size)
    train_ds = ImageFolder(Path(data_dir)/"train", transform=transformer)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
    )
    val_ds = ImageFolder(Path(data_dir)/"val", transform=transformer)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
    )
    return train_dl, val_dl


def train_single_step(model, image, gt, optim, scaler, device):
    image = image.to(device)
    gt = gt.to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=True if device.type == "cuda" else False,
    ):
        pred = model(image)
        loss = model.get_loss(pred=pred, gt=gt)

    optim.zero_grad()
    if CONFIG["DEVICE"].type == "cuda" and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()
    return loss.item()


@torch.no_grad()
def validate(val_dl, model, device):
    model.eval()
    batch_size = val_dl.batch_size
    sum_corr = 0
    for image, gt in val_dl:
        image = image.to(device)
        gt = gt.to(device)

        pred = model(image)
        acc = model.get_top_k_acc(pred=pred, gt=gt, k=5)
        sum_corr += acc * batch_size
    avg_acc = sum_corr / (batch_size * len(val_dl))
    model.train()
    return avg_acc


if __name__ == "__main__":
    PARENT_DIR = get_parent_dir()
    args = get_args()
    CONFIG = get_config(config_path=PARENT_DIR/"cifar100.yaml", args=args)

    apply_seed(CONFIG["SEED"])

    model = LinearClassifier(
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        patch_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
        n_layers=CONFIG["ARCHITECTURE"]["IMG_ENC"]["N_LAYERS"],
        n_heads=CONFIG["ARCHITECTURE"]["IMG_ENC"]["N_HEADS"],
        hidden_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["HIDDEN_DIM"],
        mlp_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["MLP_DIM"],
        embed_dim=CONFIG["ARCHITECTURE"]["EMBED_DIM"],
        n_classes=CONFIG["IMAGENET1K"]["N_CLASSES"],
    ).to(CONFIG["DEVICE"])
    state_dict = torch.load(CONFIG["CKPT_PATH"], map_location=CONFIG["DEVICE"])
    model.img_enc.load_state_dict(state_dict["image_encoder"])

    optim = Adam(
        model.parameters(),
        lr=CONFIG["TRAINING"]["LR"],
        betas=(CONFIG["OPTIMIZER"]["BETA1"], CONFIG["OPTIMIZER"]["BETA2"]),
        weight_decay=CONFIG["OPTIMIZER"]["WEIGHT_DECAY"],
    )
    scaler = GradScaler(enabled=True if CONFIG["DEVICE"].type == "cuda" else False)

    train_dl, val_dl = get_dls(
        data_dir=CONFIG["DATA_DIR"],
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        n_cpus=CONFIG["N_CPUS"],
    )

    best_avg_acc = 0
    for epoch in range(1, CONFIG["N_EPOCHS"] + 1):
        start_time = time.time()
        cum_loss = 0
        for step, (image, gt) in enumerate(train_dl, start=1):
            loss = train_single_step(
                model=model,
                image=image,
                gt=gt,
                optim=optim,
                scaler=scaler,
                device=CONFIG["DEVICE"],
            )
            cum_loss += loss
        avg_loss = cum_loss / len(train_dl)

        avg_acc = validate(val_dl=val_dl, model=model, device=CONFIG["DEVICE"])
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc

        msg = f"[ {get_elapsed_time(start_time)} ]"
        msg += f"""[ {epoch}/{CONFIG["N_EPOCHS"]} ]"""
        msg += f"""[ Loss: {avg_loss:.4f} ]"""
        msg += f"""[ Accuracy: {avg_acc:.4f} ]"""
        msg += f"""[ Best accuracy: {best_avg_acc:.4f} ]"""
        print(msg)
