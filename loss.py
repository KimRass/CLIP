import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, batch_size, temp):
        super().__init__()

        self.temp = temp

        self.gt = torch.arange(batch_size)

    def _l2_norm(self, x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    def forward(self, img_embed, text_embed):
        img_embed = self._l2_norm(img_embed)
        text_embed = self._l2_norm(text_embed)

        sim_mat = torch.matmul(img_embed, text_embed.T) * torch.exp(self.temp)

        self.gt = self.gt.to(img_embed.device)
        img_loss = F.cross_entropy(sim_mat, self.gt, reduction="mean")
        text_loss = F.cross_entropy(sim_mat.T, self.gt, reduction="mean")
        return img_loss, text_loss
