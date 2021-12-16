import sys

sys.path.append("../")
from ..nn_distance.chamfer_loss import ChamferLoss

import torch.nn as nn


class PriorLoss(nn.Module):
    def __init__(self, cd_wt, classify_wt):
        super(PriorLoss, self).__init__()
        self.cd_wt = cd_wt
        self.classify_wt = classify_wt
        self.chamferloss = ChamferLoss()
        self.classify_loss = nn.CrossEntropyLoss()

    def forward(self, inst_shape, model, pred, gt):
        cd_loss, _, _ = self.chamferloss(inst_shape, model)
        classify_loss = self.classify_loss(pred, gt)
        total_loss = self.cd_wt * cd_loss + self.classify_wt * classify_loss
        return total_loss, cd_loss, classify_loss

