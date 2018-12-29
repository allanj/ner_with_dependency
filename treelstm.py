import dynet as dy
import numpy as np
import os

class TreeLSTMBuilder():
    def __init__(self, pc_param, pc_embed, word_vocab, wdim, hdim, word_embed=None):
        self.WS = [pc_param.add_parameters((hdim, wdim)) for _ in "iou"]
        self.US = [pc_param.add_parameters((hdim, 2 * hdim)) for _ in "iou"]
        self.UFS = [pc_param.add_parameters((hdim, 2 * hdim)) for _ in "ff"]
        self.BS = [pc_param.add_parameters(hdim) for _ in "iouf"]
        self.E = pc_embed.add_lookup_parameters((len(word_vocab), wdim), init=word_embed)
        self.w2i = word_vocab

    def expr_for_tree(self, tree, decorate=False, training=True):
        if tree.isleaf(): raise RuntimeError('Tree structure error: meet with leaves')
        if len(tree.children) == 1:
            if not tree.children[0].isleaf(): raise RuntimeError(
                'Tree structure error: tree nodes with one child should be a leaf')
            emb = self.E[self.w2i.get(tree.children[0].label, 0)]
            # Wi, Wo, Wu = [dy.parameter(w) for w in self.WS]
            # bi, bo, bu, _ = [dy.parameter(b) for b in self.BS]
            i = dy.logistic(dy.affine_transform([self.BS[0], self.WS[0], emb]))
            o = dy.logistic(dy.affine_transform([self.BS[1], self.WS[1], emb]))
            u = dy.tanh(dy.affine_transform([self.BS[2], self.WS[2], emb]))
            c = dy.cmult(i, u)
            h = dy.cmult(o, dy.tanh(c))
            if decorate: tree._e = h
            return h, c
        if len(tree.children) != 2: raise RuntimeError('Tree structure error: only binary trees are supported.')
        e1, c1 = self.expr_for_tree(tree.children[0], decorate)
        e2, c2 = self.expr_for_tree(tree.children[1], decorate)
        # Ui, Uo, Uu = [dy.parameter(u) for u in self.US]
        # Uf1, Uf2 = [dy.parameter(u) for u in self.UFS]
        # bi, bo, bu, bf = [dy.parameter(b) for b in self.BS]
        e = dy.concatenate([e1, e2])
        i = dy.logistic(dy.affine_transform([self.BS[0], self.US[0], e]))
        o = dy.logistic(dy.affine_transform([self.BS[1], self.US[1], e]))
        f1 = dy.logistic(dy.affine_transform([self.BS[3], self.UFS[0], e]))
        f2 = dy.logistic(dy.affine_transform([self.BS[3], self.UFS[1], e]))
        u = dy.tanh(dy.affine_transform([self.BS[2], self.US[2], e]))
        c = dy.cmult(i, u) + dy.cmult(f1, c1) + dy.cmult(f2, c2)
        h = dy.cmult(o, dy.tanh(c))
        if decorate: tree._e = h
        return h, c
