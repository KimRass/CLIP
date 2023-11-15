import torch.nn as nn
from torchvision.models import VisionTransformer
from transformers import DistilBertConfig, DistilBertModel


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
        # Freeze parameters.
        # self.img_enc.img_proj = ClsHead(hidden_dim=hidden_dim, n_classes=n_classes)
        self.cls_head = ClsHead(hidden_dim=embed_dim, n_classes=n_classes)

    def forward(self, x):
        x = self.img_enc(x)
        x = self.cls_head(x)
        return x
