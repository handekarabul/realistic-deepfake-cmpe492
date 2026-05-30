import torch
import torch.nn.functional as F

from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="weighted_cross_entropy")
class WeightedCrossEntropyLoss(AbstractLossClass):
    """Cross-entropy with per-class weights: index 0=real, 1=fake."""

    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            self.register_buffer("class_weight", torch.tensor(weight, dtype=torch.float32))
        else:
            self.class_weight = None

    def forward(self, inputs, targets):
        weight = self.class_weight.to(inputs.device) if self.class_weight is not None else None
        return F.cross_entropy(inputs, targets, weight=weight)
