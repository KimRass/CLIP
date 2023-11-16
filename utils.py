import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import yaml
from datetime import timedelta
from time import time
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import random
import numpy as np
import os
import json
from transformers import DistilBertTokenizerFast
from copy import deepcopy


def apply_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def get_parent_dir():
    return Path(__file__).resolve().parent


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def _args_to_config(args, config):
    copied = deepcopy(config)
    for k, v in vars(args).items():
        copied[k.upper()] = v
    return copied


def get_config(config_path, args=None):
    config = load_config(config_path)
    if args is not None:
        config = _args_to_config(args=args, config=config)

    config["DEVICE"] = get_device()
    return config


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def get_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return tokenizer


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(image).save(str(path), quality=100)


def denorm(tensor, mean, std):
    tensor *= torch.Tensor(std)[None, :, None, None]
    tensor += torch.Tensor(mean)[None, :, None, None]
    return tensor


def image_to_grid(image, n_cols, mean, std):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor, mean=mean, std=std)
    grid = make_grid(tensor, nrow=n_cols, padding=2, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def l2_norm(x):
    return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)


def modify_state_dict(state_dict, keyword="_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith(keyword):
            new_key = old_key[len(keyword):]
        else:
            new_key = old_key
        new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict


def get_imagenet1k_classes(json_path):
    json_path = "/Users/jongbeomkim/Desktop/workspace/CLIP/imagenet1k_classes.json"
    with open(json_path, mode="r") as f:
        classes = json.load(f)
    classes = {k[0]: class_name for dir_name, class_name in classes.items()}
    return classes


def set_requires_grad(models, requires_grad):
    for model in models:
        for p in model.parameters():
            p.requires_grad = requires_grad
