import torch


class RelL2Loss():
    """Relative L2 loss for PDEs adopted from https://github.com/BaratiLab/FactFormer/blob/main/loss_fn.py"""

    def __init__(self, dim=-2, eps=1e-5, reduction='sum', reduce_all=True):
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
        self.reduce_all = reduce_all
    
    def __call__(self, y_hat, y):
        assert y_hat.shape == y.shape

        reduce_fn = torch.mean if self.reduction == 'mean' else torch.sum

        y_norm = reduce_fn((y ** 2), dim=self.dim)
        mask = y_norm < self.eps
        y_norm[mask] = self.eps
        diff = reduce_fn((y_hat - y) ** 2, dim=self.dim)
        diff = diff / y_norm  # [b, c]
        
        if self.reduce_all:
            diff = diff.sqrt().mean() # mean across channels and batch and any other dimensions
        else:
            diff = diff.sqrt() # do nothing
        return diff


class CombinedLoss():
    """Computes a combined loss by summing the surface and volume losses.

    The loss function is applied to the full surface tensor and the full volume
    tensor. This keeps the training objective consistent even when the dataset
    has multiple channels per field group (for example NACA4 now has surface
    pressure + normals and volume pressure + sdf + velocity).
    """

    def __init__(self, loss_fn, fields):
        self.loss_fn = loss_fn
        self.fields = fields

    def __call__(self, y_hat_surf, y_hat_vol, y_surf, y_vol):
        """Compute the combined surface and volume loss."""
        loss_surf = self.loss_fn(y_hat_surf, y_surf)
        loss_vol = self.loss_fn(y_hat_vol, y_vol)
        return loss_surf + loss_vol
