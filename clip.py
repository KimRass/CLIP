import torch
import torch.nn as nn

from utils import load_config
from model import ImageEncoder, TextEncoder

CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")


# class CLIP(object):
img_enc = ImageEncoder(
    img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
    patch_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
    n_heads=CONFIG["ARCHITECTURE"]["N_HEADS"],
    n_layers=CONFIG["ARCHITECTURE"]["N_LAYERS"],
    hidden_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_DIM"],
    mlp_dim=CONFIG["ARCHITECTURE"]["MLP_DIM"],
)
text_enc = TextEncoder(
    max_len=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["MAX_LEN"],
    n_heads=CONFIG["ARCHITECTURE"]["N_HEADS"],
    n_layers=CONFIG["ARCHITECTURE"]["N_LAYERS"],
    hidden_dim=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["TEXT_DIM"],
    mlp_dim=CONFIG["ARCHITECTURE"]["MLP_DIM"],
)
img_proj = nn.Linear(img_dim, embed_dim)
text_proj = nn.Linear(text_dim, embed_dim)

ce = nn.CrossEntropyLoss()

batch_size = 4
image = torch.randn((batch_size, 3, 224, 224))
img_embed = img_enc.encode_img(image)
img_embed.shape

token_ids = torch.randint(100, size=(batch_size, 32))
text_embed = text_enc.encode_text(token_ids)
text_embed.shape

# temp = nn.Parameter()
logits = torch.matmul(img_embed, text_embed.T)
labels = torch.arange(batch_size)
img_loss = ce(logits, labels)
text_loss = ce(logits.T, labels)
# return img_loss, text_loss