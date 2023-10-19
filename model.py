import torch
from torchvision.models import vit_b_16

img_enc = vit_b_16()
image = torch.randn((4, 3, 224, 224))
img_enc(image).shape