import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cosine_similarity(anchor, positive)
        neg_sim = self.cosine_similarity(anchor, negative)

        # Contrastive loss
        loss = torch.relu(self.margin - pos_sim + neg_sim).mean()

        return loss

contrastive_loss = ContrastiveLoss()
