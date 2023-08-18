from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]  # [N,D]


# def _neg_loss(pred, gt):
#     ''' Modified focal loss. Exactly the same as CornerNet.
#         Runs faster and costs a little bit more memory
#     Arguments:
#         pred (batch x c x h x w)
#         gt_regr (batch x c x h x w)
#     '''
#     pos_inds = gt.eq(1).float()
#     neg_inds = gt.lt(1).float()
#
#     neg_weights = torch.pow(1 - gt, 4)
#
#     loss = 0
#
#     pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
#     neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
#
#     num_pos = pos_inds.float().sum()
#     pos_loss = pos_loss.sum()
#     neg_loss = neg_loss.sum()
#
#     if num_pos == 0:
#         loss = loss - neg_loss
#     else:
#         loss = loss - (pos_loss + neg_loss) / num_pos
#     return loss
#
#
# class FocalLoss(nn.Module):
#     '''nn.Module warpper for focal loss'''
#
#     def __init__(self):
#         super(FocalLoss, self).__init__()
#         self.neg_loss = _neg_loss
#
#     def forward(self, out, target):
#         return self.neg_loss(out, target)
#
#
# class FocalLoss(nn.Module):
#
#     def __init__(self, focusing_param=2, balance_param=0.25):
#         super(FocalLoss, self).__init__()
#
#         self.focusing_param = focusing_param
#         self.balance_param = balance_param
#
#     def forward(self, output, target):
#         cross_entropy = F.cross_entropy(output, target)
#         cross_entropy_log = torch.log(cross_entropy)
#         logpt = - F.cross_entropy(output, target)
#         pt = torch.exp(logpt)
#
#         focal_loss = -((1 - pt) ** self.focusing_param) * logpt
#
#         balanced_focal_loss = self.balance_param * focal_loss
#
#         return balanced_focal_loss


# class FocalLoss(nn.Module):
    # def __init__(self, num_classes=20):
    #     super(FocalLoss, self).__init__()
    #     self.num_classes = num_classes

    # def focal_loss(self, x, y):
    #     """Focal loss.
    #     Args:
    #       x: (tensor) sized [N,D].
    #       y: (tensor) sized [N,].
    #     Return:
    #       (tensor) focal loss.
    #     """
    #     alpha = 0.25
    #     gamma = 2

    #     t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)  # [N,21]
    #     t = t[:, 1:]  # exclude background
    #     t = Variable(t).cuda()  # [N,20]

    #     p = x.sigmoid()
    #     pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    #     w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    #     w = w * (1 - pt).pow(gamma)
    #     return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    # def focal_loss_alt(self, x, y):
    #     """Focal loss alternative.
    #     Args:
    #       x: (tensor) sized [N,D].
    #       y: (tensor) sized [N,].
    #     Return:
    #       (tensor) focal loss.
    #     """
    #     alpha = 0.25

    #     t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)
    #     t = t[:, 1:]
    #     t = Variable(t).cuda()

    #     xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
    #     pt = (2 * xt + 1).sigmoid()

    #     w = alpha * t + (1 - alpha) * (1 - t)
    #     loss = -w * pt.log() / 2
    #     return loss.sum()

    # def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
    #     """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
    #     Args:
    #       loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
    #       loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
    #       cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
    #       cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
    #     loss:
    #       (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
    #     """
    #     batch_size, num_boxes = cls_targets.size()
    #     pos = cls_targets > 0  # [N,#anchors]
    #     num_pos = pos.data.long().sum()

    #     ################################################################
    #     # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
    #     ################################################################
    #     mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
    #     masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
    #     masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
    #     loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

    #     ################################################################
    #     # cls_loss = FocalLoss(loc_preds, loc_targets)
    #     ################################################################
    #     pos_neg = cls_targets > -1  # exclude ignored anchors
    #     mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
    #     masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
    #     cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

    #     print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0] / num_pos, cls_loss.data[0] / num_pos), end=' | ')
    #     loss = (loc_loss + cls_loss) / num_pos
    #     return loss

# balance_cof = 4
# def _neg_loss(pred, gt):
#     ''' Modified focal loss. Exactly the same as CornerNet.
#         Runs faster and costs a little bit more memory
#     Arguments:
#         pred (batch x c x h x w)
#         gt_regr (batch x c x h x w)
#     '''
#     # focal_inds = pred.gt(1.4013e-45) * pred.lt(1-1.4013e-45)
#     pred = torch.clamp(pred, 1.4013e-45, 1)
#     fos = torch.sum(gt, 1)
#     pos_inds = gt.eq(1).float()
#     neg_inds = gt.lt(1).float()
#     fos_inds = torch.unsqueeze(fos.gt(0).float(), 1)
#     # fos_inds = fos_inds.expand(-1, 3, -1, -1, -1)
#     fos_inds = fos_inds.expand(-1, 3, -1, -1)
#     neg_inds = neg_inds + (balance_cof - 1) * fos_inds * (gt.eq(0).float())

#     neg_weights = torch.pow(1 - gt, 4)
#     loss = 0
#     # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * focal_inds
#     # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * focal_inds
#     pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * balance_cof
#     neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

#     num_pos  = pos_inds.float().sum()
#     pos_loss = pos_loss.sum()
#     neg_loss = neg_loss.sum()

#     if num_pos == 0:
#         loss = loss - neg_loss
#     else:
#         loss = loss - (pos_loss + neg_loss) / num_pos

#     return loss
def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pred = torch.clamp(pred, 1.4013e-6, 1)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    # zro_inds = gt.eq(0).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    else:

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    # zro_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * zro_inds

    # num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    # zro_loss = zro_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        # loss = loss - (pos_loss + neg_loss + zro_loss) / num_pos
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss

def _MSE_loss(pred, gt):
    # mse_inds = pred.le(1.4013e-45) + pred.ge(1 - 1.4013e-45)
    pos_inds = gt.eq(1).float()
    num_pos = pos_inds.float().sum()
    mse_loss = torch.pow(pred - gt, 2)
    mse_loss = mse_loss.sum()

    if num_pos > 0:
        mse_loss = mse_loss / num_pos

    return mse_loss
class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss
        # nn.MSELoss()
        # self.mse_loss = nn.BCELoss()
    def forward(self, out, target):
        return self.neg_loss(out, target) #+ self.mse_loss(out, target)
class FocalLoss1(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss1, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.mse_loss = nn.MSELoss()
    def forward(self, input, target):

        ce_loss = nn.BCELoss(reduction=self.reduction,weight=self.weight)(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean() + self.mse_loss(input, target)
        return focal_loss
class Dice_loss(nn.Module):
    def __init__(self, beta=1, smooth=1e-15):
        super(Dice_loss, self).__init__()
        self.beta = beta
        self.smooth = smooth
    def forward(self, inputs, target):
        n, c, h, w = inputs.size()
        target = target.permute(0, 2, 3, 1)
        nt, ht, wt, ct = target.size()
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
            
        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1) #inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c)
        temp_target = target.view(n, -1, ct)
        
        #--------------------------------------------#
        #   计算dice loss
        #--------------------------------------------#
        tp = torch.sum(temp_target * temp_inputs, axis=[0,1])
        fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
        fn = torch.sum(temp_target              , axis=[0,1]) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        dice_loss = 1 - torch.mean(score)
        return dice_loss
# class SoftDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
 
#     def forward(self, probs, targets):
#         num = targets.size(0)
#         smooth = 1e-45
#         score0 = 0
#         # probs = F.sigmoid(logits)
#         for i in range(3):
#             m1 = probs[:, i, :, :]
#             m2 = targets[:, i, :, :]
#             m1 = m1.view(num, -1)
#             m2 = m2.view(num, -1)
#             intersection = (m1 * m2)
    
#             score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
#             score = 1 - score.sum() / num
#             score0 = score0 + score
#         return score0/3

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1e-15
        
        # probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

def Dice_loss_1(inputs, target, beta=1, smooth = 1e-15):
    n, c, h, w = inputs.size()
    target = target.permute(0, 2, 3, 1)
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c) #torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
    
    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
# bce_loss = nn.BCELoss(size_average=True)
def muti_bce_loss_fusion(d1, d2, d3,  d6, labels_v):
    bce_loss = nn.BCELoss(reduction='mean')
    # loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    # loss4 = bce_loss(d4,labels_v)
    # loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss =  loss1 + loss2 + loss3  + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss