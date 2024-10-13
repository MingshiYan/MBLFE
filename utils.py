#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author:
# @Date  : 2023/9/23 11:26
# @Desc  :
import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = 0
        for embedding in embeddings:
            tmp = torch.norm(embedding, p=self.norm)
            tmp = tmp / embedding.shape[0]
            emb_loss += tmp
        return emb_loss


def distcorr(a, b):

    n = a.shape[-1]
    X = a.unsqueeze(-1)
    X = torch.cdist(X, X)
    A = X - X.mean(axis=-2).unsqueeze(-2) - X.mean(axis=-1).unsqueeze(-1) + X.mean(axis=(-2, -1), keepdims=True)

    Y = b.unsqueeze(-1)
    Y = torch.cdist(Y, Y)
    B = Y - Y.mean(axis=-2).unsqueeze(-2) - Y.mean(axis=-1).unsqueeze(-1) + Y.mean(axis=(-2, -1), keepdims=True)

    dcov2_xy = (A * B).sum(dim=(-2, -1)) / float(n * n)
    dcov2_xx = (A * A).sum(dim=(-2, -1)) / float(n * n)
    dcov2_yy = (B * B).sum(dim=(-2, -1)) / float(n * n)
    dcor = torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy) + 1e-8)
    return dcor
