import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelBlock(nn.Module):
    def __init__(self, configs, ptl1, ptl2, state):
        super(ModelBlock, self).__init__()
        if state == 1:              # When state=1, input the sequence with the finest granularity
            self.P = configs.pl[0]  # length of periods
        if state == 2:
            self.P = configs.pl[1]
        self.map_len = configs.map_len
        self.hidR = configs.hidRNN
        self.RD = configs.RD
        self.pred_len = configs.pred_len

        if state == 1:
            self.ptl = ptl1
        if state == 2:
            self.ptl = ptl2

        self.embedding_layer = nn.Linear(self.P, self.map_len)
        self.GRU1 = nn.GRU(self.map_len, self.hidR)  # SSGRU
        self.GRU2 = nn.GRU(self.ptl, self.hidR)  # SDGRU
        self.GRU3 = nn.GRU(1, self.hidR)  # STGRU
        self.FC = nn.Linear(self.hidR * 4, self.pred_len)

    def forward(self, x):  # shape： batch_size*n_val*seq_len
        hw_input = x[:, 0, :]  # batch_size*seq_len

        r = x.permute(2, 0, 1).contiguous()
        _, h2 = self.GRU2(r)

        embedding = self.embedding_layer(x)
        embedding = embedding.permute(1, 0, 2).contiguous()  # n_val*batch_size*map_len

        h_set, h = self.GRU1(embedding)  # n_val*batch_size*hidR       1*batch_size*hidR
        ht = h
        h_set = h_set.permute(1, 2, 0).contiguous()  # batch_size*hidR*n_val
        h = h.permute(1, 0, 2).contiguous()  # batch_size*1*hidR
        score2 = h @ h_set  # batch_size*1*n_val

        att2_w = F.softmax(score2, dim=2)  # batch_size*1*n_val
        h_set = h_set.permute(0, 2, 1).contiguous()  # batch_size*n_val*hidR
        vt = att2_w @ h_set  # batch_size*1*hidR

        lst_p = hw_input.permute(1, 0).contiguous()
        lst_p = lst_p.unsqueeze(dim=2)
        _, ht3 = self.GRU3(lst_p)  # 1*batch_size*hidR

        vt = vt.squeeze()  # batch_size*hidR
        ht1 = ht.squeeze()  # batch_size*hidR
        ht2 = h2.squeeze()
        ht3 = ht3.squeeze()

        vh = torch.concat((vt, ht1, ht2, ht3), dim=1)
        res = self.FC(vh)

        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.pl1 = configs.pl[0]
        self.pl2 = configs.pl[1]
        self.ptl1 = int(self.seq_len / self.pl1)
        self.ptl2 = int(self.ptl1 / self.pl2)
        self.mg = configs.mg
        self.RD = configs.RD

        if self.mg == 1:
            self.f_grained = ModelBlock(self.configs, self.ptl1, self.ptl2, state=1)
        elif self.mg == 2:
            self.f_grained = ModelBlock(self.configs, self.ptl1, self.ptl2, state=1)
            self.c_grained = ModelBlock(self.configs, self.ptl1, self.ptl2, state=2)

    def forward(self, x):
        # x: [Batch, Input length, Channel]

        batch, input_length, channel = x.shape

        x = x.permute(0, 2, 1)  # seq: [Batch, Input length]
        x = x.reshape(batch*channel, -1)

        # hi_mean represents the evaluation value of the most recent observed seq
        if self.RD == 1:
            if self.pred_len <= input_length:
                hi_mean = x[:, -self.pred_len:].mean(dim=1).reshape(-1, 1)
            else:
                hi_mean = x.mean(dim=1).reshape(-1, 1)
            x = x - hi_mean
        input_seq = x.reshape(batch*channel, self.ptl1, self.pl1)  # input_seq: [Batch, ptl1, pl1]

        if self.mg == 1:
            o1 = self.f_grained(input_seq)
            res = o1
            if self.RD == 1:
                res += hi_mean
        else:
            o1 = self.f_grained(input_seq)
            c_grained_x = input_seq.mean(dim=2)  # shape: [Batch, ptl1, 1]
            c_grained_x = c_grained_x.view(-1, self.ptl2, self.pl2)  # shape: [Batch, ptl2, pl2]
            o2 = self.c_grained(c_grained_x)
            res = (o1 + o2) / 2
            if self.RD == 1:
                res = res + hi_mean

        res = res.reshape(batch,channel,-1)
        res = res.permute(0, 2, 1)

        return res
