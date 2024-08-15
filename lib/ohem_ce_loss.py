#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCELoss(nn.Module):

    def __init__(self, thresh, lb_ignore=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.lb_ignore = lb_ignore
        # self.weight_CE = torch.FloatTensor([1, 22, 11265, 2606,520,8602,150,423]).to('cuda')
        # self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none',weight=self.weight_CE)
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.lb_ignore].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)




#FocalLoss
def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, lb_ignore=255, gamma=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.lb_ignore = lb_ignore

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', ignore_index=self.lb_ignore), self.gamma)


#PolyFocalLoss
#PolyFocalLoss
class PolyFocalLoss(nn.Module):
    # def __init__(self, alpha=0.25, gamma=2, class_number=8,lb_ignore=255):
    def __init__(self, alpha=0.25, gamma=2, lb_ignore=255):
        super( PolyFocalLoss, self, ).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        # self.classnum = class_number
        self.lb_ignore = lb_ignore

    def forward(self, input, target):
        focal_loss_func = FocalLoss(gamma=self.gamma,lb_ignore=255)
        focal_loss = focal_loss_func(input, target)

        p = torch.sigmoid(input)
        # labels = torch.nn.functional.one_hot(target, self.classnum)
        labels = torch.nn.functional.one_hot(target, input.shape[1]).permute(0, 3, 1, 2)
        # labels = torch.tensor(labels, dtype=torch.float32)
        labels=labels.clone()
        poly = labels * p + (1 - labels) * (1 - p)
        poly_focal_loss = focal_loss + torch.mean(self.epsilon * torch.pow(1 - poly, 2 + 1), dim=-1)
        return poly_focal_loss.mean()


# ----------------- DICE Loss--------------------
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, mask=False):
        assert len(logits.shape) == 4, "inputs shape must be NCHW"
        targets = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        num = targets.size(0)
        smooth = 1.

        probs = torch.sigmoid(logits)
        m1 = probs.contiguous().view(num, -1)
        m2 = targets.contiguous().view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

# ----------------- Generalized Dice Loss--------------------
class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, pred, target,epsilon=1e-6):
        """compute the weighted dice_loss
         Args:
             pred (tensor): prediction after softmax, shape(bath_size, channels, height, width)
             target (tensor): gt, shape(bath_size, channels, height, width)
         Returns:
             gldice_loss: loss value
         """
        pred=torch.softmax(pred,dim=1)
        target=torch.unsqueeze(target,dim=1)
        wei = torch.sum(target, axis=[0, 2, 3])  # (n_class,)
        wei = 1 / (wei ** 2 + epsilon)
        intersection = torch.sum(wei * torch.sum(pred * target, axis=[0, 2, 3]))
        union = torch.sum(wei * torch.sum(pred + target, axis=[0, 2, 3]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)
        return gldice_loss

# -------------------- BCELoss -----------------------
class BCELoss(nn.Module):
    """binary bceloss with sigmoid"""

    def __init__(self,lb_ignore=255):
        super(BCELoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, inputs, targets, weights=None, mask=False):
        assert len(inputs.shape) == 4, "inputs shape must be NCHW"
        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2).float()
        if mask:
            inputs = inputs * targets
        # losses = F.binary_cross_entropy_with_logits(inputs, targets, weights,reduction='none')
        losses =self.criteria(inputs, targets)
        return losses

class CELoss(nn.Module):
    def __init__(self, lb_ignore=255, gamma=2):
        super(CELoss, self).__init__()
        self.lb_ignore = lb_ignore

    def forward(self, input, target):
        return torch.mean(F.cross_entropy(input, target, reduction='none', ignore_index=self.lb_ignore))

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2).float()

        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class DiceWithOhemCELoss(nn.Module):
    def __init__(self):
        super(DiceWithOhemCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        ohemloss = self.ohem_loss(preds, targets)
        diceloss = self.dice_loss(preds, targets)
        return ohemloss + 3*diceloss

class DiceWithFocalLoss(nn.Module):
    def __init__(self):
        super( DiceWithFocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    def forward(self, preds, targets):
        focalloss = self.focal_loss(preds, targets)
        diceloss = self.dice_loss(preds, targets)
        return diceloss+2*focalloss

class OhemWithIoULoss(nn.Module):
    def __init__(self):
        super(OhemWithIoULoss, self).__init__()
        self.iou_loss = IoULoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        iouloss = self.iou_loss(preds, targets)
        ohemloss = self.ohem_loss(preds, targets)
        return 2*iouloss + ohemloss

class OhemWithFocalLoss(nn.Module):
    def __init__(self):
        super(OhemWithFocalLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        focalloss = self.focal_loss(preds, targets)
        ohemloss = self.ohem_loss(preds, targets)
        return focalloss + ohemloss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2).float()

        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class GDiceWithOhemCELoss(nn.Module):
    def __init__(self):
        super(GDiceWithOhemCELoss, self).__init__()
        self.gdice_loss = GeneralizedDiceLoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        ohemloss = self.ohem_loss(preds, targets)
        gdiceloss = self.gdice_loss(preds, targets)
        return ohemloss + gdiceloss


if __name__ == '__main__':
    pass

