import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear, LayerNorm, ReLU,Sequential, Dropout
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

class DeepGCNLayer(torch.nn.Module):
    def __init__(self, conv=None, norm=None, activation=None, block='res+',
                 dropout=0., ckpt_grad=False):
        super(DeepGCNLayer, self).__init__()
        self.conv = conv
        self.norm = norm
        self.activation = activation
        self.block = block.lower()
        if self.block  in ['res+','res','dense','plain']:
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

class LayerNorm(torch.nn.Module):
    def __init__(self, in_feats, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.in_feats=in_feats
        self.eps=eps

        if elementwise_affine:
            self.weight = Parameter(torch.Tensor([in_feats]))
            self.bias = Parameter(torch.Tensor([in_feats]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)
    def forward(self,x,batch=None):
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)

        else:
            batch_size = int(batch.max()) + 1
            raise NotImplementedError

#         var = var/norm
#         out = x / (var.sqrt()[batch] + self.eps)
#         if self.weight is not None and self.bias is not None:
#             output= output*self.weight +self.bias

        return output

class DeeperGCN(torch.nn.Module):
    def __init__(self, g,hidden_features, out_features,num_layers):
        super(DeeperGCN, self).__init__()
        assert g.ndata['h'] is not None
        self.node_enc = nn.Linear(g.ndata['h'].size(-1),hidden_features)
        if 'w' in g.edata:
            self.edge_enc =  nn.Linear(g.edata['w'].size(-1),hidden_features)
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_features, hidden_features, aggr='softmax',
               t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_features, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_features, out_features)
    def forward(self,g, weight=None):
        graph = graph.local_var()
        graph.ndata['h'] = self.node_enc(graph.ndata['h'])
        if edge_enc is not None:
            graph.edata['w'] = self.edge_enc(graph.edata['w'])
        for layer in self.layers[1:]:
            h = layer(h)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h,p=0.1,training=self.training)
        return self.lin(h)

class GENConv(torch.nn.Module):
    def __init__(self, in_feats, out_feats,aggr= 'softmax', t=1.0, learn_t= False,
                 p = 1.0, learn_p = False, msg_norm = False,
                 learn_msg_scale = False, norm= 'batch',
                 num_layers = 2, eps = 1e-7):
        super(GENConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggr = aggr
        self.eps = eps
        if aggr in ['softmax', 'softmax_sg', 'power']:
            aggr='softmax'
        channels =[in_feats]
        for i in range(num_layers-1):
            channels.append(in_feats*2)
        channels.append(out_feats)
        self.mlp = MLP(channels, norm=norm)
        self.init_t = t
        self.init_p = p
        if learn_t and aggr == 'softmax':
            self.t = Parameter(torch.Tensor([t]), requires_grad=True)
        else:
            self.t = t

        if learn_p:
            self.p = Parameter(torch.Tensor([p]), requires_grad=True)
        else:
            self.p = p
        def reset_parameter(self):
            reset(self.mlp)
            if self.msg_norm is not None:
                self.msg_norm.reset_parameters()
            if self.t and isinstance(self.t, Tensor):
                self.t.data.fill_(self.initial_t)
            if self.p and isinstance(self.p, Tensor):
                self.p.data.fill_(self.initial_p)
        def forward(self,g,feats):
            g.update_all(message_func=message_fn(src='',out='m'),
                         reduce_func=aggregate_fn (msg='m',out='rst'))
            h = g.ndata['rst']
            if self.msg_norm is not None:
                output= self.msg_norm(x,out)
            output +=x_r
            return self.mlp(rst)

        def message_fn(self,edges):
            if edges.data['w'] is not None:
                msg = edges.data['w']+ edges.src['h']
            else:
                msg = edges.src['h']
            return {'msg':F.relu(msg)+self.eps }

        def aggregate_fn(self,nodes):
            softmax_nn = nn.Softmax(dim=1)
            if self.aggr=="softmax":
                out = softmax_nn(nodes.mailbox['msg']*self.t)
                rst=  nodes.mailbox['msg']*out
            return{'rst':rst }

class MLP(Sequential):
    def __init__(self, channels, norm= None,
                 bias= True, dropout= 0):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super(MLP, self).__init__(*m)



class MessageNorm(torch.nn.Module):
    def __init__(self, learn_scale= False):
        super(MessageNorm, self).__init__()
        self.scale = Parameter(torch.Tensor([1.0]), requires_grad=learn_scale)

    def reset_parameters(self):
        self.scale.data.fill_(1.0)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=-1)
        x_norm = x.norm(p=p, dim=-1, keepdim=True)
        return msg * x_norm * self.scale
