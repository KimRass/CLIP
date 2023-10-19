# Source:
    # https://github.com/KimRass/ViT/blob/main/model.py
    # https://github.com/KimRass/BERT/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import config


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_size, drop_prob=config.DROP_PROB):
        super().__init__()

        self.patch_size = patch_size
        dim = (patch_size ** 2) * 3

        self.norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, hidden_size)
        self.drop = nn.Dropout(drop_prob)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = rearrange(
            x,
            pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.norm1(x)
        x = self.proj(x)
        x = self.drop(x)
        x = self.norm2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_prob=config.DROP_PROB):
        super().__init__()

        self.head_size = hidden_size // n_heads
        self.n_heads = n_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * n_heads * self.head_size, bias=False)
        self.drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _get_attention_score(self, q, k):
        attn_score = torch.einsum("bhnd,bhmd->bhnm", q, k)
        return attn_score

    def forward(self, x, mask=None):
        q, k, v = torch.split(
            self.qkv_proj(x), split_size_or_sections=self.n_heads * self.head_size, dim=2,
        )
        q = rearrange(q, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        k = rearrange(k, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        v = rearrange(v, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9)
        attn_weight = F.softmax(attn_score / (self.head_size ** 0.5), dim=3)
        x = torch.einsum("bhnm,bhmd->bhnd", attn_weight, v)
        x = rearrange(x, pattern="b h n d -> b n (h d)")
        x = self.out_proj(x)
        x = self.drop(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = self.norm(x)
        x = sublayer(x)
        x += skip
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_size):
        super().__init__()

        self.proj1 = nn.Linear(hidden_size, mlp_size)
        self.drop1 = nn.Dropout(0.1)
        self.proj2 = nn.Linear(mlp_size, hidden_size)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.drop1(x)
        x = self.proj2(x)
        x = F.gelu(x)
        x = self.drop2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, mlp_size, n_heads):
        super().__init__()

        self.self_attn = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads)
        self.self_attn_resid = ResidualConnection(hidden_size=hidden_size)
        self.mlp = PositionwiseFeedForward(hidden_size=hidden_size, mlp_size=mlp_size)
        self.mlp_resid = ResidualConnection(hidden_size=hidden_size)

    def forward(self, x):
        x = self.self_attn_resid(x=x, sublayer=self.self_attn)
        x = self.mlp_resid(x=x, sublayer=self.mlp)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, hidden_size, mlp_size, n_heads):
        super().__init__()

        self.enc_stack = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size=hidden_size, mlp_size=mlp_size, n_heads=n_heads)
                for _ in range(n_layers)]
        )

    def forward(self, x):
        for enc_layer in self.enc_stack:
            x = enc_layer(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        n_layers=12,
        hidden_size=768,
        mlp_size=3072,
        n_heads=12,
        drop_prob=config.DROP_PROB,
        n_classes=0,
    ):
        super().__init__()

        self.n_classes = n_classes

        assert img_size % patch_size == 0, "`img_size` must be divisible by `patch_size`!"

        cell_size = img_size // patch_size
        n_patches = cell_size ** 2

        self.patch_embed = PatchEmbedding(patch_size=patch_size, hidden_size=hidden_size)
        self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_size)))
        self.pos_embed = nn.Parameter(torch.randn((1, n_patches + 1, hidden_size)))
        self.drop1 = nn.Dropout(drop_prob)
        self.tf_enc = TransformerEncoder(
            n_layers=n_layers, hidden_size=hidden_size, mlp_size=mlp_size, n_heads=n_heads,
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, n_classes)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x):
        b, _, _, _ = x.shape

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.repeat(b, 1, 1), x), dim=1)
        x += self.pos_embed
        x = self.drop1(x)
        x = self.tf_enc(x)

        if self.n_classes == 0:
            x = x.mean(dim=1)
        else:
            x = x[:, 0, :]
            x = self.norm(x)
            x = self.proj(x)
            x = self.drop2(x)
        return x

if __name__ == "__main__":
    # image = torch.randn((4, 3, 32, 32))
    # img_enc = ImageEncoder(
    #     img_size=32,
    #     patch_size=16,
    #     n_layers=12,
    #     hidden_size=192,
    #     n_heads=12,
    #     n_classes=100,
    # )
    # img_out = img_enc(image)
    # print(img_out.shape)

    text_enc = TextEncoder(
        vocab_size=30000, max_len=64, pad_id=0,
    )
    seq = torch.randint(high=1000, size=(4, 32))
    # token_embed = TokenEmbedding(vocab_size=3000, hidden_size=768, pad_id=0)
    # pos_embed = LearnedPositionalEmbedding(max_len=64, embed_size=768, pad_id=0)
    # token_embed(seq)
    # pos_embed(seq)
    text_out = text_enc(seq)
    print(text_out.shape)
