import torch
import torch.nn as nn
from pathlib import Path

from utils import load_config
from model import ImageEncoder, TextEncoder

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")


# class CLIP(object):
class CLIP(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        max_len,
        # n_heads,
        # n_layers,
        img_dim,
        text_dim,
        mlp_dim,
        embed_dim,
    ):
        super().__init__()

        self.img_enc = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            # n_heads=n_heads,
            # n_layers=n_layers,
            img_dim=img_dim,
            mlp_dim=mlp_dim,
            embed_dim=embed_dim,
        )
        self.text_enc = TextEncoder(
            max_len=max_len,
            # n_heads=n_heads,
            # n_layers=n_layers,
            text_dim=text_dim,
            mlp_dim=mlp_dim,
            embed_dim=embed_dim,
        )

        # "The learnable temperature parameter was initialized to the equivalent of 0.07 from and clipped
        # to prevent scaling the logits by more than 100 which we found necessary to prevent training instability."
        self.temp = nn.Parameter(torch.tensor((0.07,)))

        self.ce = nn.CrossEntropyLoss()

    def _l2_norm(self, x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    def get_loss(self, image, token_ids):
        b, _, _, _ = image.shape

        img_embed = self.img_enc(image)
        img_embed = self._l2_norm(img_embed)

        text_embed = self.text_enc(token_ids)
        text_embed = self._l2_norm(text_embed)

        # "To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016), half-precision Adam statistics (Dhariwal et al., 2020), and half-precision stochastically rounded text encoder weights were used."
        # The calculation of embedding similarities was also sharded with individual GPUs computing only the subset of the pairwise similarities necessary for their local batch of embeddings."
        logits = torch.matmul(img_embed, text_embed.T) * torch.exp(self.temp)
        labels = torch.arange(b).to(image.device)
        img_loss = self.ce(logits, labels) / 2
        text_loss = self.ce(logits.T, labels) / 2
        return img_loss, text_loss


if __name__ == "__main__":
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
    img_loss, text_loss = clip.get_loss(image=image, token_ids=token_ids)
    img_loss, text_loss
