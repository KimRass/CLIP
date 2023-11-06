import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import faiss
from tqdm import tqdm

from utils import load_config, get_device, image_to_grid
from tokenizer import get_tokenizer
from imagenet1k import get_imagenet1k_classes, ImageNet1kDataset
from train import get_clip
from semantic_search import (
    get_encoders_from_checkpoint,
    add_texts_to_faiss_index,
    perform_semantic_search,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--max_len", type=int, required=False, default=128)
    parser.add_argument("--k", type=int, required=False, default=5)
    # parser.add_argument("--torch_compile", action="store_true", required=False)
    # parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


def init_faiss_index(dim):
    faiss_idx = faiss.IndexFlatIP(dim) # `IP`: Inner Product
    faiss_idx = faiss.IndexIDMap2(faiss_idx)
    return faiss_idx


def get_number_of_correct_preds(gt, nns, k):
    arr_gt = gt.detach().cpu().numpy()
    eq = np.equal(nns, np.repeat(arr_gt[:, None], repeats=k, axis=1))
    # acc = corr.sum(axis=1).astype("float").mean()
    return eq.sum(axis=1).sum()


if __name__ == "__main__":
    # CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/clip_flickr.pth"
    img_enc, text_enc = get_encoders_from_checkpoint(
        ckpt_path, config=CONFIG, max_len=args.max_len, device=DEVICE,
    )
    img_enc.eval()
    text_enc.eval()

    CLASSES = get_imagenet1k_classes(Path(__file__).resolve().parent/"imagenet1k_classes.json")
    
    faiss_idx = init_faiss_index(img_enc.embed_dim)
    tokenizer = get_tokenizer()
    add_texts_to_faiss_index(
        faiss_idx=faiss_idx,
        idx2text={i[0]: i[1] for i in CLASSES.values()},
        text_enc=text_enc,
        tokenizer=tokenizer,
        max_len=text_enc.max_len,
    )

    ds = ImageNet1kDataset(data_dir=args.data_dir, img_size=img_enc.img_size, classes=CLASSES)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=False,
        drop_last=False,
    )
    n_corrs = 0
    for image, gt in tqdm(dl):
        image = image.to(DEVICE)
        gt = gt.to(DEVICE)

        query_embed = img_enc(image)
        _, nns = perform_semantic_search(query_embed=query_embed, faiss_idx=faiss_idx, k=args.k)

        corr = get_number_of_correct_preds(gt=gt, nns=nns, k=args.k)
        n_corrs += corr
    acc = n_corrs / len(ds)
    print(acc)
