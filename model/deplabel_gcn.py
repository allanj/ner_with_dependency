# 
# @author: Allan
#

import torch
import torch.nn as nn
import torch.nn.functional as F



class GCN(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()

        self.gcn_hidden_dim = config.dep_hidden_dim
        self.num_gcn_layers = config.num_gcn_layers
        self.gcn_mlp_layers = config.gcn_mlp_layers
        # gcn layer
        self.layers = self.num_gcn_layers
        self.device = config.device
        self.mem_dim = self.gcn_hidden_dim
        # self.in_dim = config.hidden_dim + config.dep_emb_size  ## lstm hidden dim
        self.in_dim = input_dim  ## lstm hidden dim

        print("[Model Info] GCN Input Size: {}, # GCN Layers: {}, #MLP: {}".format(self.in_dim, self.num_gcn_layers, config.gcn_mlp_layers))
        self.gcn_drop = nn.Dropout(config.gcn_dropout).to(self.device)

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim).to(self.device))

        self.dep_emb = nn.Embedding(len(config.deplabels), 1)

        # output mlp layers
        in_dim = config.hidden_dim
        layers = [nn.Linear(in_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]
        for _ in range(self.gcn_mlp_layers - 1):
            layers += [nn.Linear(self.gcn_hidden_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]

        self.out_mlp = nn.Sequential(*layers).to(self.device)



    def forward(self, gcn_inputs, word_seq_len, adj_matrix, dep_label_matrix):

        # print(adj_matrix.size())

        batch_size, sent_len, input_dim = gcn_inputs.size()

        denom = adj_matrix.sum(2).unsqueeze(2) + 1

        ##dep_label_matrix: NxN
        ##dep_emb.
        dep_embs = self.dep_emb(dep_label_matrix)  ## B x N x N x 1
        dep_embs = dep_embs.squeeze(3) * adj_matrix
        dep_denom = dep_embs.sum(2).unsqueeze(2) + 1

        # gcn_biinput = gcn_inputs.view(batch_size, sent_len, 1, input_dim).expand(batch_size, sent_len, sent_len, input_dim) ## B x N x N x h
        # weighted_gcn_input = (dep_embs + gcn_biinput).sum(2)

        for l in range(self.layers):

            Ax = adj_matrix.bmm(gcn_inputs) ## N x N  times N x h  = Nxh
            AxW = self.W[l](Ax)   ## N x m
            AxW = AxW + self.W[l](gcn_inputs)  # self loop  Nxh
            AxW = AxW / denom

            Bx = dep_embs.bmm(gcn_inputs)
            BxW = self.W_label[l](Bx)
            BxW = BxW + self.W_label[l]


            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW


        outputs = self.out_mlp(gcn_inputs)
        return outputs



