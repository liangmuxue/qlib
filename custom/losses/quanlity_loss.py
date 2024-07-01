import torch
from torch import nn
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.utils import _check_quantiles, raise_if_not

from cus_utils.common_compute import pairwise_distances

class QuanlityLoss(QuantileRegression):
    """分位数损失函数"""
    
    def __init__(self,device=None):
        super(QuanlityLoss, self).__init__()
        self.quantiles = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
        ]     
        self.quantiles_tensor = torch.tensor(self.quantiles).to(device)
        self.device = device

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor):
        """
        We are re-defining a custom loss (which is not a likelihood loss) compared to Likelihood

        Parameters
        ----------
        model_output
            must be of shape (batch_size, n_timesteps, n_target_variables, n_quantiles)
        target
            must be of shape (n_samples, n_timesteps, n_target_variables)
        """

        dim_q = 3

        # test if torch model forward produces correct output and store quantiles tensor
        if self.first:
            raise_if_not(
                len(model_output.shape) == 4
                and len(target.shape) == 3
                and model_output.shape[:2] == target.shape[:2],
                "mismatch between predicted and target shape",
            )
            raise_if_not(
                model_output.shape[dim_q] == len(self.quantiles),
                "mismatch between number of predicted quantiles and target quantiles",
            )
            self.first = False
        
        self.quantiles_tensor = self.quantiles_tensor.to(self.device)
        errors = target.unsqueeze(-1) - model_output
        losses = torch.max(
            (self.quantiles_tensor - 1) * errors, self.quantiles_tensor * errors
        )

        return losses.sum(dim=dim_q).mean()
    
