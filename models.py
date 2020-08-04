import torch as torch
from torch import nn
from torch.nn import init
import dgl
EPS = 1e-15
class DeepGraphInfomax(nn.Module):
    """ DeepGraphInfomax"""
    def __init__(self, hidden_feats, encoder, summary, corruption, weight=True):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_feats = hidden_feats
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        if weight:
            self.weight = nn.Parameter(torch.Tensor(hidden_feats, hidden_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        def reset(nn):
            def _reset(item):
                if hasattr(item, 'reset_parameters'):
                    item.reset_parameters()
            if nn is not None:
                if hasattr(nn, 'children') and len(list(nn.children())) > 0:
                    for item in nn.children():
                        _reset(item)
            else:
                _reset(nn)
        reset(self.encoder)
        reset(self.summary)
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, g, feat):
        with g.local_scope():
            pos_z = self.encoder(g, feat)
            cor = self.corruption(feat)
    #         cor = cor if isinstance(cor, tuple) else (cor, )
            neg_z = self.encoder(g, cor)
            summary = self.summary(pos_z)
            return pos_z, neg_z, summary

    def discriminate(self, z, summary, sigmoid=True):

        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary,
                                             sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z,
                                                 summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss
