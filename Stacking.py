# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:51:07 2018

@author: Enyan
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
class Stack(nn.Module):
    def __init__(self,inputDim,outputDim):
        super(Stack, self).__init__()
        self.weight = nn.Linear(inputDim,outputDim)
#        self.hidden = nn.Linear(10,outputDim)
    def forward(self,inputTensor):
#        a1 = inputTensor[:,0]
#        a2 = inputTensor[:,1]
#        b1 = inputTensor[:,2]
#        b2 = inputTensor[:,3]
#        feature = torch.cat([inputTensor,(a1*b1).view([-1,1]),(a1*b2).view([-1,1]),(a2*b1).view([-1,1]),(a2*b2).view([-1,1])],dim = 1)
        scores = self.weight(inputTensor)
        scores = F.log_softmax(scores,dim = 1)
        return scores