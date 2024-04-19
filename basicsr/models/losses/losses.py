# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class JSDivergence(nn.Module):
    def __init__(self, eps=1e-12):
        """
        Initialize the Jensen-Shannon Divergence module.
        
        Args:
        - eps (float): A small value to ensure numerical stability.
        """
        super(JSDivergence, self).__init__()
        self.eps = eps
        self.gamma = 2.0

    def forward(self, logits, target_probs, weighted=False):
        """
        Calculate the Jensen-Shannon Divergence between logits and target probabilities.
        
        Args:
        - logits (Tensor): Logits from the model.
        - target_probs (Tensor): Target probability distribution.
        
        Returns:
        - JS divergence (Tensor)
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(logits, dim=1)
        
        # Calculate the mean distribution M
        M = 0.5 * (pred_probs + target_probs)

        # KL divergence for each distribution against M
        kl_div1 = self.kl_divergence(target_probs, M, weighted=weighted)
        kl_div2 = self.kl_divergence(pred_probs, M, weighted=weighted)
        
        # Jensen-Shannon Divergence
        js_div = 0.5 * kl_div1 + 0.5 * kl_div2
        return js_div

    def kl_divergence(self, p, q, weighted):
        """
        Calculate the Kullback-Leibler divergence D(P || Q).
        
        Args:
        - p (Tensor): True probability distribution.
        - q (Tensor): Approximated probability distribution.
        
        Returns:
        - KL divergence (Tensor)
        """
        p = p + self.eps  # Ensure numerical stability
        q = q + self.eps  # Ensure numerical stability
        if weighted ==False:
            return (p * (p / q).log()).sum(dim=1).mean()
        else:
            error = torch.abs(p - q)
            weights = torch.exp(self.gamma * error) - 1.0
            #print(torch.min(weights), torch.max(weights))
            return (weights * p *  (p / q).log()).sum(dim=1).mean()
        

class FocalMSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocalMSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.gamma = 1.0

    def forward(self, pred, target, weight=None, focal=False, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if not focal:
            return self.loss_weight * mse_loss(
                pred, target, weight, reduction=self.reduction)
        else:
            error = torch.abs(pred- target)
            weights_focal = torch.exp(self.gamma * error) - 1.0

            return self.loss_weight * mse_loss(
                pred, target, weight*weights_focal, reduction=self.reduction)



class CosineSimilarityLoss(nn.Module):
    """Cosine Similarity Loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        if reduction not in ['mean']: #['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        target_normalized = target / target.norm(dim=1, keepdim=True)
        pred_normalized= pred / pred.norm(dim=1, keepdim=True)

        cosine_sim = F.cosine_similarity(pred_normalized, target_normalized, dim=1)
        
        loss = 1 - cosine_sim.mean()
        
        return self.loss_weight*loss