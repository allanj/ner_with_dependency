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
        self.edge_gate = config.edge_gate
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
        self.W_in = nn.ModuleList()
        self.W_out = nn.ModuleList()
        self.W_self = nn.ModuleList()

        self.biases = nn.ModuleList()

        if self.edge_gate:
            print("[Info] Labeled GCN model will be added edge-wise gating.")
            self.gates = nn.ModuleList()
            self.g_in = nn.ModuleList()
            self.g_out = nn.ModuleList()
            self.g_self = nn.ModuleList()

            self.gbiases = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W_in.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
            self.W_out.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
            self.W_self.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
            self.biases.append(nn.Embedding(len(config.deplabels), self.mem_dim).to(config.device))
            if self.edge_gate:
                self.g_in.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
                self.g_out.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
                self.g_self.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
                self.gbiases.append(nn.Embedding(len(config.deplabels), self.mem_dim).to(config.device))



        # output mlp layers
        in_dim = config.hidden_dim
        layers = [nn.Linear(in_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]
        for _ in range(self.gcn_mlp_layers - 1):
            layers += [nn.Linear(self.gcn_hidden_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]

        self.out_mlp = nn.Sequential(*layers).to(self.device)



    def forward(self, gcn_inputs, word_seq_len, adj_matrix_in, adj_matrix_out, dep_label_matrix):

        """

        :param gcn_inputs:
        :param word_seq_len:
        :param adj_matrix: should already contain the self loop
        :param dep_label_matrix:
        :return:
        """

        batch_size, sent_len, input_dim = gcn_inputs.size()

        denom = adj_matrix_in.sum(2).unsqueeze(2) + adj_matrix_out.sum(2).unsqueeze(2) + 1

        ##dep_label_matrix: NxN
        ##dep_emb.

        for l in range(self.layers):

            Ax = adj_matrix_in.bmm(gcn_inputs)  ## N x N  times N x h  = Nxh
            AxW = self.W_in[l](Ax)   ## N x m

            Bx = adj_matrix_out.bmm(gcn_inputs)
            BxW = self.W_out[l](Bx)

            self_out = self.W_self[l](gcn_inputs)


            res = (AxW + BxW + self_out) / denom

            dep_embs = self.biases[l](dep_label_matrix)  ## B x N x N x hidden_size.
            ## masking step.
            total_bias = (dep_embs * adj_matrix_in.unsqueeze(3) ).sum(1) + (dep_embs * adj_matrix_out.unsqueeze(3)).sum(2)

            res += total_bias

            if self.edge_gate:
                gAxW = self.g_in[l](Ax)  ## N x m

                gBxW = self.g_out[l](Bx)

                gself_out = self.g_self[l](gcn_inputs)

                gres = (gAxW + gBxW + gself_out) / denom

                gdep_embs = self.gbiases[l](dep_label_matrix)  ## B x N x N x hidden_size.
                ## masking step.
                gtotal_bias = (gdep_embs * adj_matrix_in.unsqueeze(3)).sum(1) + (
                        gdep_embs * adj_matrix_out.unsqueeze(3)).sum(2)

                gres += gtotal_bias

                gres = torch.sigmoid(gres)

                res = F.relu(gres * res)
            else:
                res = F.relu(res)

            gcn_inputs = self.gcn_drop(res) if l < self.layers - 1 else res


        outputs = self.out_mlp(gcn_inputs)
        return outputs



