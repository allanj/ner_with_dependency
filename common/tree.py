# 
# @author: Allan
#


class Tree(object):
    def __init__(self, pos, children):
        self.pos = pos
        self.children = children

    def __str__(self):
        if self.children is None: return self.pos
        return "[%s %s]" % (self.pos, " ".join([str(c) for c in self.children]))

    def isleaf(self):
        return self.children is None

    def leaves_iter(self):
        if self.isleaf():
            yield self
        else:
            for c in self.children:
                for l in c.leaves_iter(): yield l

    def leaves(self):
        return list(self.leaves_iter())

    def nonterms_iter(self):
        if not self.isleaf():
            yield self
            for c in self.children:
                for n in c.nonterms_iter(): yield n

    def nonterms(self):
        return list(self.nonterms_iter())
