from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm

from utils import get_parent_dir, get_config, get_tokenizer
from data.imagenet1k import get_imagenet1k_classes, ImageNet1kDataset
from semantic_search import (
    init_faiss_index,
    get_encoders_from_checkpoint,
    add_texts_to_faiss_index,
    perform_semantic_search,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--max_len", type=int, required=False, default=128)
    parser.add_argument("--k", type=int, required=False, default=5)

    args = parser.parse_args()
    return args


def get_number_of_correct_preds(gt, nns, k):
    arr_gt = gt.detach().cpu().numpy()
    eq = np.equal(nns, np.repeat(arr_gt[:, None], repeats=k, axis=1))
    return eq.sum(axis=1).sum()


if __name__ == "__main__":
    args = get_args()
    PARENT_DIR = get_parent_dir()
    CONFIG = get_config(config_path=PARENT_DIR/"configs/imagenet1k.yaml", args=args)
    CLASSES = get_imagenet1k_classes(PARENT_DIR/"imagenet1k_classes.json")

    img_enc, text_enc = get_encoders_from_checkpoint(
        ckpt_path=CONFIG["CKPT_PATH"],
        config=CONFIG,
        max_len=CONFIG["MAX_LEN"],
        device=CONFIG["DEVICE"],
    )
    img_enc.eval()
    text_enc.eval()

    
    faiss_idx = init_faiss_index(img_enc.embed_dim)
    tokenizer = get_tokenizer()
    add_texts_to_faiss_index(
        faiss_idx=faiss_idx,
        idx2text={i[0]: i[1] for i in CLASSES.values()},
        text_enc=text_enc,
        tokenizer=tokenizer,
        max_len=text_enc.max_len,
    )

    ds = ImageNet1kDataset(data_dir=CONFIG["DATA_DIR"], img_size=img_enc.img_size, classes=CLASSES)
    dl = DataLoader(
        ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["N_CPUS"],
        pin_memory=False,
        drop_last=False,
    )
    n_corrs = 0
    for image, gt in tqdm(dl):
        image = image.to(CONFIG["DEVICE"])
        gt = gt.to(CONFIG["DEVICE"])

        query_embed = img_enc(image)
        _, nns = perform_semantic_search(query_embed=query_embed, faiss_idx=faiss_idx, k=CONFIG["K"])

        corr = get_number_of_correct_preds(gt=gt, nns=nns, k=CONFIG["K"])
        n_corrs += corr
    acc = n_corrs / len(ds)
    print(acc)
