import dgl
import dgl.function as fn
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

class DeepGCNLayer(nn.Module):
    def __init__(self, conv=None, norm=None, act=None, block='res+',
                 dropout=0., ckpt_grad=False):
        super(GCNLayer, self).__init__()
        self.conv = conv
        self.norm = norm
        self.activation = activation
        self.block = block.lower()
        if self.block is not in ['res+','res','dense','plain']:
            self.block ='res+'
        self.dropout= dropout

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()
    def forward(self, g, feature):
        with g.local_scope():
#             g.ndata['h'] = feature
#             g.update_all(gcn_msg, gcn_reduce)
#             h = g.ndata['h']
            if self.block =='res+':
                if self.norm is not None:
                    h = self.norm(x)
                if self.activation is not None:
                    h = self.activation(x)
                h = F.dropout(h, p=self.dropout,training=self.training)
                if self.conv is  None:
                    h = self.conv(h)

            return feature+h
