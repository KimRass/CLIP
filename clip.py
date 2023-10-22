import torch
import torch.nn as nn
from pathlib import Path

from utils import load_config
from model import ImageEncoder, TextEncoder

# CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml")
CONFIG = load_config(Path(__file__).parent/"config.yaml")


class CLIP(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        img_n_layers,
        img_n_heads,
        img_hidden_dim,
        img_mlp_dim,
        vocab_size,
        max_len,
        text_n_layers,
        text_n_heads,
        text_hidden_dim,
        text_mlp_dim,
        embed_dim,
    ):
        super().__init__()

        self.img_enc = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_layers=img_n_layers,
            n_heads=img_n_heads,
            hidden_dim=img_hidden_dim,
            mlp_dim=img_mlp_dim,
            embed_dim=embed_dim,
        )
        self.text_enc = TextEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=text_n_layers,
            n_heads=text_n_heads,
            hidden_dim=text_hidden_dim,
            mlp_dim=text_mlp_dim,
            embed_dim=embed_dim,
        )

        # "The learnable temperature parameter was initialized to the equivalent of 0.07."
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

        # "To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016),
        # half-precision Adam statistics (Dhariwal et al., 2020), and half-precision stochastically rounded
        # text encoder weights were used."
        # The calculation of embedding similarities was also sharded with individual GPUs computing only the subset
        # of the pairwise similarities necessary for their local batch of embeddings."
        logits = torch.matmul(img_embed, text_embed.T) * torch.exp(self.temp)
        labels = torch.arange(b).to(image.device)
        img_loss = self.ce(logits, labels) / 2
        text_loss = self.ce(logits.T, labels) / 2
        return img_loss, text_loss
