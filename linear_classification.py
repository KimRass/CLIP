import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import argparse
from tqdm import tqdm
from torch.optim import Adam
from torch.cuda.amp import GradScaler

from utils import load_config, get_device
from model import LinearClassifier
from data_augmentation import get_train_transformer
from loss import ClassificationLoss


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    CONFIG = load_config(Path(__file__).parent/"CONFIG.yaml")

    args = get_args()

    DEVICE = get_device()

    model = LinearClassifier(
        img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
        patch_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
        n_layers=CONFIG["ARCHITECTURE"]["IMG_ENC"]["N_LAYERS"],
        n_heads=CONFIG["ARCHITECTURE"]["IMG_ENC"]["N_HEADS"],
        hidden_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["HIDDEN_DIM"],
        mlp_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["MLP_DIM"],
        embed_dim=CONFIG["ARCHITECTURE"]["EMBED_DIM"],
        n_classes=1000,
    )
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    model.img_enc.load_state_dict(state_dict["image_encoder"])
    model.img_enc.eval()

    optim = Adam(
        model.parameters(),
        # lr=CONFIG.BASE_LR,
        lr=0.0001
        # betas=(CONFIG.BETA1, CONFIG.BETA2),
        # weight_decay=CONFIG.WEIGHT_DECAY,
    )
    scaler = GradScaler(enabled=True if DEVICE.type == "cuda" else False)

    crit = ClassificationLoss(n_classes=1000)

    transformer = get_train_transformer(img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"])
    train_ds = ImageFolder(args.data_dir, transform=transformer)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=False,
        drop_last=False,
    )
    init_epoch = 0
    for epoch in range(init_epoch + 1, 32 + 1):
        for step, (image, gt) in tqdm(enumerate(train_dl, start=1)):
            image = image.to(DEVICE)
            gt = gt.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
            ):
                pred = model(image)
                loss = crit(pred, gt)

            optim.zero_grad()
            if DEVICE.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
