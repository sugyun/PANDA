import numpy as np
import torch
from torch import nn


def IsNaN(pred):
    return torch.isnan(pred).sum() > 0


def binary_accuracy_from_probs(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy().reshape(-1)
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    y_pred_label = (y_pred_np >= 0.5).astype(np.float32)
    return float((y_pred_label == y_true_np).mean())


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


class CSG2AFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=64, dropout=0.1):
        super(CSG2AFeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
