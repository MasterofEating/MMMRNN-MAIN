import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch, input_length, channel = x.shape
        res_set = torch.zeros((batch, self.pred_len, channel)).cuda()
        for i in range(0, channel):
            seq = x[:, :, i]  # seq: [Batch, Input length, 1]
            res = self.Linear(seq)
            res_set[:, :, i] = res

        return res_set