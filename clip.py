# References:
    # https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py

# "To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016),
# half-precision Adam statistics (Dhariwal et al., 2020), and half-precision stochastically rounded
# text encoder weights were used."
# The calculation of embedding similarities was also sharded with individual GPUs computing only the subset
# of the pairwise similarities necessary for their local batch of embeddings."

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        batch_size,
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

        # "The learnable temp parameter was initialized to the equivalent of 0.07."
        self.temp = nn.Parameter(torch.tensor((0.07,)))

        self.gt = torch.arange(batch_size, device=self.device)

    def _l2_norm(self, x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    def forward(self, image, token_ids, attn_mask):
        # b, _, _, _ = image.shape

        img_embed = self.img_enc(image)
        text_embed = self.text_enc(token_ids=token_ids, attn_mask=attn_mask)

        img_embed = self._l2_norm(img_embed)
        text_embed = self._l2_norm(text_embed)

        sim_mat = torch.matmul(img_embed, text_embed.T) * torch.exp(self.temp)
        # print(F.softmax(sim_mat, dim=1).argmax(dim=1))

        # gt = torch.arange(b, device=image.device)
        img_loss = F.cross_entropy(sim_mat, self.gt)
        text_loss = F.cross_entropy(sim_mat.T, self.gt)
        return img_loss, text_loss


if __name__ == "__main__":
    def _l2_norm(x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    b = 4
    img_embed = torch.randn(b, 256)
    text_embed = torch.randn(b, 256)

    # img_embed = _l2_norm(img_embed)
    # text_embed = _l2_norm(text_embed)
    
    mat = (img_embed @ text_embed.T)
    -F.log_softmax(mat, dim=1)
