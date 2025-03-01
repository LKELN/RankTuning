from torch import Tensor
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from einops import rearrange, reduce, repeat
import torch
from torch import nn

class Global_ranktuning(nn.Module):
    def __init__(self, train_batch_size=25, N=4, sort_num_heads=4, encoder_layer=2, sort_layers=4, num_corr=15,
                 sort_dim=64, num_class=2,
                 ):
        super(Global_ranktuning, self).__init__()
        self.train_batch_size = train_batch_size
        self.N = N
        self.num_corr = num_corr
        self.sort_dim = sort_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, sort_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.dim_increase = nn.Linear(self.num_corr, sort_dim, bias=True)
        self.dim_decrease = nn.Linear(sort_dim, num_class, bias=True)
        sort_transformer = nn.TransformerEncoderLayer(d_model=sort_dim, nhead=sort_num_heads,
                                                      dim_feedforward=sort_dim * 4, activation="gelu",
                                                      batch_first=True, dropout=0)
        self.sort = nn.TransformerEncoder(sort_transformer, num_layers=sort_layers)

        encoder_transformer = nn.TransformerEncoderLayer(d_model=sort_dim, nhead=sort_num_heads,
                                                         dim_feedforward=sort_dim * 4, activation="gelu",
                                                         batch_first=True, dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_transformer, num_layers=encoder_layer)

    def forward(self, x: Tensor, args=None):  # b,65,128
        if self.training:
            x = x.reshape(args.train_batch_size, args.negs_num_per_query + args.pos_num_per_query + 1,
                          -1,
                          args.features_dim)
        x = x.unsqueeze(0)
        correlation = torch.matmul(x[:, :1], x[:, 1:].permute(0, 1, 3, 2))
        q_sort = torch.argsort(correlation, dim=3, descending=True)
        k_sort = torch.argsort(correlation, dim=2, descending=True)

        q_info = torch.gather(input=correlation, index=q_sort[:, :, :, :self.num_corr],
                                dim=3)  # (1, 500, num_corr, 7)
        k_info = torch.gather(input=correlation, index=k_sort[:, :, :self.num_corr, :], dim=2)
        k_info = torch.cat([q_info, k_info.permute((0, 1, 3, 2))], dim=2)
        x = k_info
        b_s, num_reference, L, C = x.shape
        x = x.reshape(b_s * num_reference, L, C)
        x = self.dim_increase(x)
        x = torch.cat((self.cls_token.repeat(b_s * num_reference, 1, 1), x), dim=1)
        x = self.sort(x)
        x = x[:, 0, :]
        x = x.reshape(b_s, num_reference, self.sort_dim)
        x = self.encoder(x)
        x = self.dim_decrease(x)  # batch_size,99
        x = x.softmax(dim=-1)[:, :, 1]
        return x


