import torch
import torch.nn as nn
from torchvision.models import VisionTransformer
# from torchvision.models import vit_b_32
from transformers import DistilBertConfig, DistilBertModel


class ImageEncoder(nn.Module):
    def __init__(self, img_size, patch_size, n_layers, n_heads, hidden_dim, mlp_dim, embed_dim):
        super().__init__()

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


# "The text encoder is a Transformer with the architecture modifications described in Radford et al. (2019).
# As a base size we use a 63M-parameter 12-layer 512-wide model with 8 attention heads. For computational efficiency, the max sequence length was capped at 76. The text sequence is bracketed with [SOS] and [EOS] tokens and the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text which is layer normalized and then linearly projected into the multi-modal embedding space. Masked self-attention was used in the text encoder to preserve"
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, n_heads, hidden_dim, mlp_dim, embed_dim):
        super().__init__()

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
