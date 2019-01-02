# 
# @author: Allan
#


class Tree():
    def __init__(self, pos):
        self.pos = pos
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def sort_children(self):
        self.children.sort(key=lambda x : x.pos)

    def __str__(self):
        if self.children is None: return self.pos
        return "[%s %s]" % (self.pos, " ".join([str(c) for c in self.children]))

    def is_leaf(self):
        return len(self.children) == 0

    def leaves_iter(self):
        if self.is_leaf():
            yield self.pos
        else:
            for c in self.children:
                for l in c.leaves_iter(): yield l

    def leaves(self):
        return list(self.leaves_iter())

    def nonterms_iter(self):
        if not self.is_leaf():
            yield self
            for c in self.children:
                for n in c.nonterms_iter(): yield n

    def nonterms(self):
        return list(self.nonterms_iter())

    def __eq__(self, other):
        return other and self.pos == other.pos and self.children == other.children

    def __hash__(self):
        return hash((self.pos, self.children))

if __name__ == "__main__":

    '''###read the tree
    '''

    from config.reader import Reader
    reader = Reader()
    insts = reader.read_conll("../data/abc/train.conllx", number=1)

    for inst in insts:
        nodes = [Tree(pos) for pos in range(len(inst.input.words))]
        root = Tree(-1)
        for pos, head in enumerate(inst.input.heads):
            if head != -1:
                nodes[head].add_child(nodes[pos])
            else:
                root.add_child(nodes[pos])
        inst.nodes = nodes
        for node in nodes:
            node.sort_children()
        print(root.leaves())
        for pos, node in enumerate(nodes):
            if node.is_leaf():
                print("{}".format(pos))






