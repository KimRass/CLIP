# References:
    # https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py
    # https://github.com/KimRass/ViT/blob/main/eval.py

# "To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016),
# half-precision Adam statistics (Dhariwal et al., 2020), and half-precision stochastically rounded
# text encoder weights were used."
# The calculation of embedding similarities was also sharded with individual GPUs computing only the subset
# of the pairwise similarities necessary for their local batch of embeddings."

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VisionTransformer
from transformers import DistilBertConfig, DistilBertModel

from utils import set_requires_grad, l2_norm


class ImageEncoder(nn.Module):
    def __init__(self, img_size, patch_size, n_layers, n_heads, hidden_dim, mlp_dim, embed_dim):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        self.model = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_layers=n_layers,
            num_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )
        self.model.heads = nn.Identity()
        self.img_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.img_proj(x)
        return x


# "The text encoder is a Transformer with the architecture modifications described in Radford et al. (2019)."
# As a base size we use a 63M-parameter 12-layer 512-wide model with 8 attention heads. The text sequence is bracketed with [SOS] and [EOS] tokens and the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text which is layer normalized and then linearly projected into the multi-modal embedding space. Masked self-attention was used in the text encoder to preserve"
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, n_heads, hidden_dim, mlp_dim, embed_dim):
        super().__init__()

        self.max_len = max_len

        self.model = DistilBertModel(
            DistilBertConfig(
                vocab_size=vocab_size,
                max_position_embeddings=max_len,
                n_heads=n_heads,
                n_layers=n_layers,
                dim=hidden_dim,
                hidden_dim=mlp_dim,
                attention_dropout=0.1,
            )
        )
        self.text_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, token_ids, attn_mask):
        x = self.model(input_ids=token_ids, attention_mask=attn_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.text_proj(x)
        return x


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

    @staticmethod
    def _l2_norm(x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    def get_top_k_acc(self, img_embed, text_embed, k):
        img_embed = self._l2_norm(img_embed)
        text_embed = self._l2_norm(text_embed)

        mat = torch.matmul(img_embed, text_embed.T)
        _, topk = torch.topk(mat, k=k, dim=1)
        corr = torch.eq(topk, self.gt.to(img_embed.device).unsqueeze(1).repeat(1, k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc


class ClsHead(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super().__init__()

        self.n_classes = n_classes

        self.head_proj = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x = x[:, 0, :]
        x = self.head_proj(x)
        x = x.view(-1, self.n_classes)
        return x


class CELossWithLabelSmoothing(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

    def forward(self, pred, gt, label_smoothing=0):
        assert 0 <= label_smoothing <= 1, "The argument `label_smoothing` must be between 0 and 1!"

        if gt.ndim == 1:
            gt = torch.eye(self.n_classes, device=gt.device)[gt]
            return self(pred, gt, label_smoothing=label_smoothing)
        elif gt.ndim == 2:
            log_prob = F.log_softmax(pred, dim=1)
            ce_loss = -torch.sum(gt * log_prob, dim=1)
            loss = (1 - label_smoothing) * ce_loss
            loss += label_smoothing * -torch.sum(log_prob, dim=1)
            return torch.mean(loss)


class ClsTopKAccuracy(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, pred, gt):
        _, topk = torch.topk(pred, k=self.k, dim=1)
        corr = torch.eq(topk, gt.unsqueeze(1).repeat(1, self.k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc


class LinearClassifier(nn.Module):
    def __init__(self, img_size, patch_size, n_layers, n_heads, hidden_dim, mlp_dim, embed_dim, n_classes):
        super().__init__()

        self.img_enc = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            embed_dim=embed_dim,
        )
        self.cls_head = ClsHead(hidden_dim=embed_dim, n_classes=n_classes)

        self.loss_fn = CELossWithLabelSmoothing(n_classes=n_classes)

        # Freeze parameters.
        set_requires_grad(models=[self.img_enc], requires_grad=False)
        self.img_enc.eval()

    def forward(self, x):
        x = self.img_enc(x)
        x = self.cls_head(x.detach())
        return x

    def get_loss(self, pred, gt, label_smoothing):
        return self.loss_fn(pred=pred, gt=gt, label_smoothing=label_smoothing)

    def get_top_k_acc(self, pred, gt, k):
        _, topk = torch.topk(pred, k=k, dim=1)
        corr = torch.eq(topk, gt.unsqueeze(1).repeat(1, k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc
