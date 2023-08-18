import torch
import numpy
a = torch.ones((2, 3,4))
print(a)
numpy.save(./results/1.npy, 1)

# a= a.view(2,-1)
# # a1 =  torch.sum(a, axis = [0,1])
# # a2 =  torch.sum(a, dim=(0,1))

    

# print(a.size())
# # print(a1)
# # print(a2)
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss1(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss1, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = nn.BCELoss(reduction=self.reduction,weight=self.weight)(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss