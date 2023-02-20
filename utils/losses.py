import torch
from torch import nn
from torch import functional as F
from utils.dataConvert import one_hot


def Dice_loss(inputs, target):
    inputs_hot = one_hot(inputs.argmax(dim=1))
    target_hot = one_hot(target)
    inter = inputs_hot * target_hot
    unin = inputs_hot.sum(dim=3) + target_hot.sum(dim=3)
    scores = 2 * inter.sum(dim=3) / unin

    dice_loss = 1 - scores.mean()
    return dice_loss


def Focal_Loss(inputs, target, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def loss(inputs, target):
    return Focal_Loss(inputs, target) + Dice_loss(inputs, target)
