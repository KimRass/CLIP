# References:
    # https://github.com/KimRass/ViT/blob/main/evaluate.py

import torch
import torch.nn as nn


class CLIPTopKAccuracy(nn.Module):
    def __init__(self, k, batch_size):
        super().__init__()

        self.k = k
        self.gt = torch.arange(batch_size)

    def _l2_norm(self, x):
        return x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    def forward(self, img_embed, text_embed):
        img_embed = self._l2_norm(img_embed)
        text_embed = self._l2_norm(text_embed)

        mat = torch.matmul(img_embed, text_embed.T)
        _, topk = torch.topk(mat, k=self.k, dim=1)
        corr = torch.eq(topk, self.gt.to(img_embed.device).unsqueeze(1).repeat(1, self.k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc


class ClsTopKAccuracy(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, pred, gt):
        _, topk = torch.topk(pred, k=self.k, dim=1)
        corr = torch.eq(topk, gt.unsqueeze(1).repeat(1, self.k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc
