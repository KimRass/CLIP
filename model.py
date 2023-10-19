import torch
from torchvision.models import VisionTransformer
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast
from copy import copy


class ImageEncoder(object):
    def __init__(self, img_size, patch_size, n_heads, n_layers, hidden_dim, mlp_dim):
        self.model = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_heads=n_heads,
            num_layers=n_layers,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )
        self.model.heads.head.register_forward_hook(self.get_img_embed())

    def get_img_embed(self):
        def forward_hook_fn(model, input, output):
            self.img_embed = input[0]
        return forward_hook_fn

    def encode_img(self, image):
        self.model(image)
        return copy(self.img_embed)


class TextEncoder(object):
    def __init__(self, max_len, n_heads, n_layers, hidden_dim, mlp_dim):
        self.model = DistilBertModel(
            DistilBertConfig(
                max_position_embeddings=max_len,
                n_heads=n_heads,
                n_layers=n_layers,
                dim=hidden_dim,
                hidden_dim=mlp_dim,
                attention_dropout=0.1,
            )
        )
        # self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def encode_text(self, token_ids):
        out = self.model(token_ids)
        text_embed = out.last_hidden_state[:, 0, :]
        return text_embed


if __name__ == "__main__":
    img_enc = ImageEncoder()
    image = torch.randn((4, 3, 224, 224))
    img_embed = img_enc.encode_img(image)
    img_embed.shape

    text_enc = TextEncoder()
    token_ids = torch.randint(100, size=(4, 32))
    text_embed = text_enc.encode_text(token_ids)
    text_embed.shape
