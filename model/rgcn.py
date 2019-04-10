import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

class RGCNLayer(nn.Module):
    def __init__(self, config, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.config = config
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat).to(config.device))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases).to(config.device))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat).to(config.device))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight


        def message_func(edges):

            w = weight[edges.data['rel_type'].long().to(self.config.device)]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm'].to(self.config.device)
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation is not None:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class DepRGCN(nn.Module):
    def __init__(self, config, input_dim):
        super(DepRGCN, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.h_dim = config.dep_hidden_dim
        self.num_rels = len(config.deplabels)
        self.num_bases = len(config.deplabels)
        self.num_hidden_layers = config.num_gcn_layers

        # create rgcn layers
        self.build_model()


    def build_model(self):
        self.layers = nn.ModuleList()
        # # input to hidden
        # i2h = self.build_hidden_layer(True)
        # self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(False)
            self.layers.append(h2h)
        # hidden to output


    def build_hidden_layer(self, is_input_layer):
        return RGCNLayer(self.config, self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=is_input_layer)

    def forward(self, feats, g): ##is a batched graph.
        """

        :param g: a batched graph
        :param feats: batch_size x sent_len x input_size
        :return:
        """
        batch_size, sent_len, self.input_dim = feats.size()
        g.ndata['h'] = feats.contiguous().view(-1, self.input_dim)
        for layer in self.layers:
            layer(g)
        output = g.ndata.pop('h').view(batch_size, sent_len, -1)
        return output
