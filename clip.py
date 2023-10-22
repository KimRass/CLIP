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

    def _l2_norm(self, x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    def get_losses(self, image, token_ids, attn_mask):
        b, _, _, _ = image.shape

        img_embed = self.img_enc(image)
        img_embed = self._l2_norm(img_embed)

        text_embed = self.text_enc(token_ids=token_ids, attn_mask=attn_mask)
        text_embed = self._l2_norm(text_embed)

        # "To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016),
        # half-precision Adam statistics (Dhariwal et al., 2020), and half-precision stochastically rounded
        # text encoder weights were used."
        # The calculation of embedding similarities was also sharded with individual GPUs computing only the subset
        # of the pairwise similarities necessary for their local batch of embeddings."
        # cos_sim_mat = torch.matmul(img_embed, text_embed.T)
        # labels = torch.arange(b).to(image.device)
        # img_loss = self.ce(logits, labels)
        # text_loss = self.ce(logits.T, labels)
        logits = (img_embed @ text_embed.T) / self.temp
        img_sim = img_embed @ img_embed.T
        text_sim = text_embed @ text_embed.T
        targets = F.softmax((img_sim + text_sim) / 2 * self.temp, dim=-1)
        img_loss = cross_entropy(logits, targets, reduction="none")
        text_loss = cross_entropy(logits.T, targets.T, reduction="none")
        return img_loss, text_loss



def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


if __name__ == "__main__":
    token_ids = torch.randint(0, 100, size=(2, 12))
    attn_mask = torch.randint(0, 2, size=(2, 12))
    clip.text_enc(token_ids=token_ids, attn_mask=attn_mask)
