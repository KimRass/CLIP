import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import l2_norm


class CLIPLoss(nn.Module):
    def __init__(self, batch_size, temp):
        super().__init__()

        self.temp = temp

        self.gt = torch.arange(batch_size)

    def forward(self, img_embed, text_embed):
        img_embed = l2_norm(img_embed)
        text_embed = l2_norm(text_embed)

        sim_mat = torch.matmul(img_embed, text_embed.T) * torch.exp(self.temp)

        self.gt = self.gt.to(img_embed.device)
        img_loss = F.cross_entropy(sim_mat, self.gt, reduction="mean")
        text_loss = F.cross_entropy(sim_mat.T, self.gt, reduction="mean")
        tot_loss = (img_loss + text_loss) / 2
        return tot_loss


if __name__ == "__main__":
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    a = l2_norm(a)
    b = l2_norm(b)
    torch.matmul(a, b.T)
    F.cosine_similarity(a, b)
