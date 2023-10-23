import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import DistilBertTokenizerFast
import argparse
from time import time
from pathlib import Path
import wandb

from utils import load_config, get_device, get_elapsed_time, get_clip
from flickr import Flickr8kDataset, DataCollatorForDynamicPadding
from tokenizer import load_tokenizer

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()
