import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import pdb


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction 

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label. # 0-3
    c: (n_batches, 1)
        The censoring status indicator. # 0 or 1
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """

    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h) 

    S = torch.cumprod(1 - hazards, dim=1) 
    S_padded = torch.cat([torch.ones_like(c), S], 1) 

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    neg_l = censored_loss + uncensored_loss

    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


def orthogonal_loss(paths_postSA_embed, wsi_postSA_embed, lambda1=0.5, lambda2=0.5):
    embd1 = paths_postSA_embed
    embd2 = wsi_postSA_embed
    dot_product = torch.dot(embd1, embd2)
    loss = lambda1 * torch.abs(dot_product) + lambda2 * dot_product**2
    return loss



class NllAddOrthogonalSurvLoss(nn.Module):
   
    def __init__(self, paths_postSA_embed, wsi_postSA_embed, alpha=0.0, eps=1e-7, reduction='sum', lambda1=0.5, lambda2=0.5):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction 
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        loss_nll = nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)
        loss_orthogonal = orthogonal_loss(paths_postSA_embed, wsi_postSA_embed, lambda1=self.lambda1, lambda2=self.lambda2)

        return loss_nll + loss_orthogonal


class MAELossFunction(nn.Module):
    def __init__(self, eps=1e-7, reduction='sum'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def __call__(self, all_hazards_which, y_disc_which, all_censorships):
        return mae_loss(all_hazards_which, y_disc_which, all_censorships, eps=self.eps, reduction=self.reduction)

def mae_loss(all_hazards_which, y_disc_which, all_censorships, eps=1e-7, reduction='sum'):
    y = y_disc_which.type(torch.int64)
    c = all_censorships.type(torch.int64)

    hazards = torch.sigmoid(all_hazards_which)

    if c == 0:
        uncensored_hazard = hazards[0][y[0]].detach().cpu()
        uncensored_y_disc = y_disc_which[0].detach().cpu()
        uncensored_mae = 0
        if uncensored_hazard > uncensored_y_disc:
            uncensored_mae = torch.log(abs(uncensored_hazard - uncensored_y_disc)) + 1.0
        loss = uncensored_mae
    else:
        censored_hazard = hazards[0][y[0]].detach().cpu()
        censored_y_disc = y_disc_which[0].detach().cpu()
        censored_mae = 0
        if censored_hazard < censored_y_disc:
            censored_mae = torch.log(abs(censored_y_disc - censored_hazard)) + 1.0
        loss = censored_mae
    return loss

   
class CombinedLoss(nn.Module):
    def __init__(self, nll_loss_instance, mae_loss_instance, nll_weight=1.0, mae_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.nll_loss = nll_loss_instance
        self.mae_loss = mae_loss_instance
        self.nll_weight = nll_weight
        self.mae_weight = mae_weight

    def forward(self, h, y, t, c):
        y = y.type(torch.int64)
        c = c.type(torch.int64)
        nll = self.nll_loss(h, y, t, c)

        hazards = torch.sigmoid(h)
        all_hazards_which = torch.argmax(hazards, dim=-1)  

        y_disc_which = y 
        mae = self.mae_loss(all_hazards_which, y_disc_which, c.detach().cpu().numpy())

        return self.nll_weight * nll + self.mae_weight * mae
