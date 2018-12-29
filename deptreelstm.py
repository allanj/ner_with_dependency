# 
# @author: Allan
#
import dynet as dy

class DepTreeLSTM():

    def __init__(self, model, config):
        wdim = config.embedding_dim
        hdim = config.tree_hdim

        self.WS = [model.add_parameters((hdim, wdim)) for _ in "ifou"]
        self.BS = [model.add_parameters((wdim)) for _ in "ifou"]
        self.US = [model.add_parameters((hdim, hdim)) for _ in "ifou"]

    def expr_for_tree(self, tree, seq_h, final_h):
        x = seq_h[tree.pos]
        if tree.is_leaf():
            i = dy.logistic(dy.affine_transform([self.BS[0], self.WS[0], x]))
            o = dy.logistic(dy.affine_transform([self.BS[2], self.WS[2], x]))
            u = dy.tanh(dy.affine_transform([self.BS[3], self.WS[3], x]))
            c = dy.cmult(i, u)
            h = dy.cmult(o, dy.tanh(c))
            final_h[tree.pos] = h
            return h, c

        uis = dy.concatenate([self.US[0]] * len(tree.children), 1)
        ufs = dy.concatenate([self.US[1]] * len(tree.children), 1)
        uos = dy.concatenate([self.US[2]] * len(tree.children), 1)
        uus = dy.concatenate([self.US[3]] * len(tree.children), 1)

        hs = []
        cs = []
        for child in tree.children:
            h, c = self.expr_for_tree(child, seq_h)
            hs.append(h)
            cs.append(c)
        hs = dy.concatenate(hs, 1)
        i = dy.logistic(dy.affine_transform([self.BS[0], self.WS[0], x, uis, hs]))
        fs = [dy.logistic(dy.affine_transform([self.BS[1], self.WS[1], x, ufs, hs])) ] * len(tree.children)
        o = dy.logistic(dy.affine_transform([self.BS[2], self.WS[2], x, uos, hs]))
        u = dy.tanh(dy.affine_transform([self.BS[3], self.WS[3], x, uus, hs]))
        c = dy.cmult(i, u) + dy.sum_dim(dy.concatenate([dy.cmult(f, c) for f,c in zip(fs, cs)], 1), [1])
        h = dy.cmult(o, dy.tanh(c))
        final_h[tree.pos] = h
        return h, c