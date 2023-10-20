import torch
import torch.nn as nn

from utils import load_config
from model import ImageEncoder, TextEncoder

CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")


class CLIP(object):
    def __init__(
        self,
        img_size,
        patch_size,
        max_len,
        n_heads,
        n_layers,
        img_dim,
        text_dim,
        embed_dim,
        mlp_dim,
    ):
        self.img_enc = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_heads=n_heads,
            n_layers=n_layers,
            hidden_dim=img_dim,
            mlp_dim=mlp_dim,
        )
        self.text_enc = TextEncoder(
            max_len=max_len,
            n_heads=n_heads,
            n_layers=n_layers,
            hidden_dim=text_dim,
            mlp_dim=mlp_dim,
        )
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        self.ce = nn.CrossEntropyLoss()

    def get_loss(self, image, token_ids):
        b, _, _, _ = image.shape

        img_embed = self.img_enc.encode_img(image)
        img_embed = self.img_proj(img_embed)

        text_embed = self.text_enc.encode_text(token_ids)
        text_embed = self.text_proj(text_embed)

        # temp = nn.Parameter()
        logits = torch.matmul(img_embed, text_embed.T)
        labels = torch.arange(b)
        img_loss = self.ce(logits, labels)
        text_loss = self.ce(logits.T, labels)
        return img_loss, text_loss


clip = CLIP(
    img_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_SIZE"],
    patch_size=CONFIG["ARCHITECTURE"]["IMG_ENC"]["PATCH_SIZE"],
    max_len=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["MAX_LEN"],
    n_heads=CONFIG["ARCHITECTURE"]["N_HEADS"],
    n_layers=CONFIG["ARCHITECTURE"]["N_LAYERS"],
    img_dim=CONFIG["ARCHITECTURE"]["IMG_ENC"]["IMG_DIM"],
    text_dim=CONFIG["ARCHITECTURE"]["TEXT_ENC"]["TEXT_DIM"],
    embed_dim=256,
    mlp_dim=CONFIG["ARCHITECTURE"]["MLP_DIM"],
)
batch_size = 4
image = torch.randn((batch_size, 3, 224, 224))
token_ids = torch.randint(100, size=(batch_size, 32))
loss = clip.get_loss(image=image, token_ids=token_ids)


path = "/Users/jongbeomkim/Downloads/QR_20230817_1476_0_LLEZC2WBGOZSTWPXAAQ0.png"
for _ in range(100):
    img = load_image(path)
    save_image(img, path=path)