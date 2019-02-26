# 
# @author: Allan
#
import torch.nn as nn
import torch
import torch.nn.functional as F

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, config, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.device = config.device
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        """
        :param inputs: emb size
        :param child_c:
        :param child_h:
        :return:
        """
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward_recursive(self, tree, inputs, final_h):
        """
        :param tree: tree object
        :param inputs: sentlen x hidden size.
        :return:
        """
        for child in tree.children:
            self.forward_recursive(child, inputs, final_h)

        if len(tree.children) == 0:
            # child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            # child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_c = torch.zeros(1, self.mem_dim).to(self.device)
            child_h = torch.zeros(1, self.mem_dim).to(self.device)
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.pos], child_c, child_h)
        _, final_h[tree.pos] = tree.state

    def forward(self, tree, inputs):
        inputs = inputs.squeeze(0)
        num_words = inputs.size(0)
        final_h = torch.zeros(num_words, self.mem_dim)
        self.forward_recursive(tree, inputs, final_h)
        return final_h