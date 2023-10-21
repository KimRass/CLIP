import torch
import torch.nn as nn
from torchvision.models import VisionTransformer
from transformers import DistilBertConfig, DistilBertModel
from copy import copy


class ImageEncoder(nn.Module):
    def __init__(self, img_size, patch_size, n_heads, n_layers, img_dim, mlp_dim, embed_dim):
        super().__init__()

        self.model = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_heads=n_heads,
            num_layers=n_layers,
            hidden_dim=img_dim,
            mlp_dim=mlp_dim,
        )
        self.model.heads = nn.Identity()
        self.img_proj = nn.Linear(img_dim, embed_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.img_proj(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, max_len, n_heads, n_layers, text_dim, mlp_dim, embed_dim):
        super().__init__()

        self.model = DistilBertModel(
            DistilBertConfig(
                max_position_embeddings=max_len,
                n_heads=n_heads,
                n_layers=n_layers,
                dim=text_dim,
                hidden_dim=mlp_dim,
                attention_dropout=0.1,
            )
        )
        self.text_proj = nn.Linear(text_dim, embed_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.last_hidden_state[:, 0, :]
        x = self.text_proj(x)
        return x


if __name__ == "__main__":
    img_enc = ImageEncoder()
    image = torch.randn((4, 3, 224, 224))
    img_embed = img_enc.encode_img(image)
    img_embed.shape

    text_enc = TextEncoder()
    token_ids = torch.randint(100, size=(4, 32))
    text_embed = text_enc.encode_text(token_ids)
    text_embed.shape
