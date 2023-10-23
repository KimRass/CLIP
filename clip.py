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

        self.ce = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _l2_norm(self, x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    def get_losses(self, image, token_ids, attn_mask):
        b, _, _, _ = image.shape

        img_embed = self.img_enc(image)
        text_embed = self.text_enc(token_ids=token_ids, attn_mask=attn_mask)

        logits = (img_embed @ text_embed.T) / self.temp
        labels = torch.arange(b).to(image.device)
        img_loss = self.ce(logits, labels) / 2
        text_loss = self.ce(logits.T, labels) / 2
        return img_loss, text_loss

        # logits = (img_embed @ text_embed.T) / self.temp
        # img_sim = img_embed @ img_embed.T
        # text_sim = text_embed @ text_embed.T
        # targets = F.softmax((img_sim + text_sim) / 2 * self.temp, dim=-1)
        # img_loss = (-targets * self.log_softmax(logits)).sum(dim=1)
        # text_loss = (-targets.T * self.log_softmax(logits.T)).sum(dim=1)
        # return img_loss.mean(), text_loss.mean()

        # img_embed = self._l2_norm(img_embed)
        # text_embed = self._l2_norm(text_embed)

        # cos_sim_mat = img_embed @ text_embed.T # $[-1, 1]$
        # mat = (cos_sim_mat + 1) / 2 # $[0, 1]$
        # labels = torch.arange(b).to(image.device)
        # img_loss = self.ce(logits, labels)
        # text_loss = self.ce(logits.T, labels)


if __name__ == "__main__":
    # def _l2_norm( x):
#     return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    # img_embed = torch.randn(4, 256)
    # text_embed = torch.randn(4, 256)
    # logits = (img_embed @ text_embed.T)
    # labels = torch.arange(4)
    # logits
    # labels
    # img_loss = nn.CrossEntropyLoss()(logits, labels)
    # text_loss = nn.CrossEntropyLoss()(logits.T, labels)
    # img_loss, text_loss
    # logits = (img_embed @ text_embed.T)
    # img_sim = img_embed @ img_embed.T
    # text_sim = text_embed @ text_embed.T
    # targets = F.softmax((img_sim + text_sim) / 2, dim=-1)
    # img_loss = (-targets * nn.LogSoftmax(dim=-1)(logits)).sum(dim=1)
    # img_loss
