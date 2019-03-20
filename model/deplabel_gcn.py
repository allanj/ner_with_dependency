# 
# @author: Allan
#

import torch
import torch.nn as nn
import torch.nn.functional as F



class DepLabeledGCN(nn.Module):
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
        self.self_dep_label_id = torch.tensor(config.deplabel2idx[config.self_label]).long().to(self.device)

        print("[Model Info] GCN Input Size: {}, # GCN Layers: {}, #MLP: {}".format(self.in_dim, self.num_gcn_layers, config.gcn_mlp_layers))
        self.gcn_drop = nn.Dropout(config.gcn_dropout).to(self.device)

        # gcn layer
        self.W = nn.ModuleList()
        self.W_label = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
            self.W_label.append(nn.Linear(input_dim, self.mem_dim).to(self.device))

        self.dep_emb = nn.Embedding(len(config.deplabels), 1).to(config.device)

        # output mlp layers
        in_dim = config.hidden_dim
        layers = [nn.Linear(in_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]
        for _ in range(self.gcn_mlp_layers - 1):
            layers += [nn.Linear(self.gcn_hidden_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]

        self.out_mlp = nn.Sequential(*layers).to(self.device)



    def forward(self, gcn_inputs, word_seq_len, adj_matrix, dep_label_matrix):

        """

        :param gcn_inputs:
        :param word_seq_len:
        :param adj_matrix: should already contain the self loop
        :param dep_label_matrix:
        :return:
        """

        batch_size, sent_len, input_dim = gcn_inputs.size()

        denom = adj_matrix.sum(2).unsqueeze(2) + 1

        ##dep_label_matrix: NxN
        ##dep_emb.
        dep_embs = self.dep_emb(dep_label_matrix)  ## B x N x N x 1
        dep_embs = dep_embs.squeeze(3) * adj_matrix
        #
        self_val = self.dep_emb(self.self_dep_label_id)
        dep_denom = dep_embs.sum(2).unsqueeze(2) + self_val

        # gcn_biinput = gcn_inputs.view(batch_size, sent_len, 1, input_dim).expand(batch_size, sent_len, sent_len, input_dim) ## B x N x N x h
        # weighted_gcn_input = (dep_embs + gcn_biinput).sum(2)

        for l in range(self.layers):

            Ax = adj_matrix.bmm(gcn_inputs)  ## N x N  times N x h  = Nxh
            AxW = self.W[l](Ax)   ## N x m
            AxW = AxW + self.W[l](gcn_inputs)  ## self loop  N x h
            AxW = AxW / denom

            Bx = dep_embs.bmm(gcn_inputs)
            BxW = self.W_label[l](Bx)
            BxW = BxW + self.W_label[l](gcn_inputs * self_val)
            BxW = BxW / dep_denom


            gAxW = F.relu(AxW + BxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW


        outputs = self.out_mlp(gcn_inputs)
        return outputs



