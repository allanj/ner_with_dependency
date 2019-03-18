# 
# @author: Allan
#
import torch
import torch.nn as nn
import torch.nn.functional as F



class SyntacticGCN(nn.Module):
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

        print("[Model Info] Syntactic GCN Input Size: {}, # GCN Layers: {}, #MLP: {}".format(self.in_dim, self.num_gcn_layers, config.gcn_mlp_layers))
        self.gcn_drop = nn.Dropout(config.gcn_dropout).to(self.device)

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append([nn.Linear(input_dim, self.mem_dim,bias=False).to(self.device), nn.Linear(input_dim, self.mem_dim,bias=False).to(self.device)
                            ,nn.Linear(input_dim, self.mem_dim,bias=False).to(self.device)] ) ##because of ->, self loop, <-

        label_bias = nn.Parameter(torch.randn(len(config.deplabels), self.mem_dim))

        # output mlp layers
        in_dim = config.hidden_dim
        layers = [nn.Linear(in_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]
        for _ in range(self.gcn_mlp_layers - 1):
            layers += [nn.Linear(self.gcn_hidden_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]

        self.out_mlp = nn.Sequential(*layers).to(self.device)



    def forward(self, gcn_inputs, word_seq_len, out_matrix, in_matrix, ):
        """

        :param gcn_inputs: batch_size x N x input_size
        :param word_seq_len:
        :param out_matrix:
        :param in_matrix:
        :return:
        """

        # print(adj_matrix.size())

        denom = adj_matrix.sum(2).unsqueeze(2) + 1 ##because of self loop plus 1

        for l in range(self.layers):
            Ax = adj_matrix.bmm(gcn_inputs)  ## batch_size x N x input_size
            AxW = self.W[l](Ax)  ## with size..  batch_size x N x self.mem_dim.
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            ## syntactic bias should be batch_size x N x self.mem_dim
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW


        outputs = self.out_mlp(gcn_inputs)
        return outputs



