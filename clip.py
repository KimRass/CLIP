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

from model import ImageEncoder, TextEncoder
from utils import l2_norm


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

        self.img_size= img_size
        self.max_len = max_len
        self.embed_dim = embed_dim

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

        self.gt = torch.arange(batch_size)

    def forward(self, image, token_ids, attn_mask):
        img_embed = self.img_enc(image)
        text_embed = self.text_enc(token_ids=token_ids, attn_mask=attn_mask)
        return img_embed, text_embed

    def get_loss(self, img_embed, text_embed):
        img_embed = l2_norm(img_embed)
        text_embed = l2_norm(text_embed)

        sim_mat = torch.matmul(img_embed, text_embed.T)

        self.gt = self.gt.to(img_embed.device)
        img_loss = F.cross_entropy(sim_mat, self.gt, reduction="mean")
        text_loss = F.cross_entropy(sim_mat.T, self.gt, reduction="mean")
        tot_loss = (img_loss + text_loss) / 2
        return tot_loss


if __name__ == "__main__":
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    a = l2_norm(a)
    b = l2_norm(b)
    torch.matmul(a, b.T)
    F.cosine_similarity(a, b)
