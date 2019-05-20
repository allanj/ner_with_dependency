# 
# @author: Allan
#


from config.reader import Reader
import numpy as np

import matplotlib.pyplot as plt
import random
from config.eval import evaluate, Span



def use_iobes(insts):
    for inst in insts:
        output = inst.output
        for pos in range(len(inst)):
            curr_entity = output[pos]
            if pos == len(inst) - 1:
                if curr_entity.startswith("B-"):
                    output[pos] = curr_entity.replace("B-", "S-")
                elif curr_entity.startswith("I-"):
                    output[pos] = curr_entity.replace("I-", "E-")
            else:
                next_entity = output[pos + 1]
                if curr_entity.startswith("B-"):
                    if next_entity.startswith("O") or next_entity.startswith("B-"):
                        output[pos] = curr_entity.replace("B-", "S-")
                elif curr_entity.startswith("I-"):
                    if next_entity.startswith("O") or next_entity.startswith("B-"):
                        output[pos] = curr_entity.replace("I-", "E-")


dataset = "conll2003"
train = "../data/"+dataset+"/train.sud.conllx"
dev = "../data/"+dataset+"/dev.sud.conllx"
test = "../data/"+dataset+"/test.sud.conllx"
digit2zero = False
reader = Reader(digit2zero)

insts = reader.read_conll(train, -1, True)
insts += reader.read_conll(dev, -1, False)
insts += reader.read_conll(test, -1, False)
use_iobes(insts)
L = 3


def get_spans(output):
    output_spans = set()
    start = -1
    for i in range(len(output)):
        if output[i].startswith("B-"):
            start = i
        if output[i].startswith("E-"):
            end = i
            output_spans.add(Span(start, end, output[i][2:]))
        if output[i].startswith("S-"):
            output_spans.add(Span(i, i, output[i][2:]))
    return output_spans

count_all = 0
count_have_sub = 0
count_grand = 0
for inst in insts:
    output = inst.output
    spans = get_spans(output)
    # print(spans)
    for span in spans:
        if span.right - span.left + 1 < L:
            continue
        count_dep = 0
        count_all += 1
        has_grand = False
        for i in range(span.left, span.right + 1):
            if inst.input.heads[i] >= span.left and inst.input.heads[i] <= span.right:
                count_dep += 1
            if inst.input.heads[i] >= span.left and inst.input.heads[i] <= span.right:
                head_i = inst.input.heads[i]
                if head_i != -1 and inst.input.heads[head_i] >= span.left and inst.input.heads[head_i] <= span.right:

                    has_grand = True
        if has_grand:
            count_grand += 1
        if count_dep == (span.right - span.left):
            count_have_sub += 1
        else:
            print(inst.input.words)


print(count_have_sub, count_all, count_have_sub/count_all*100)
print(count_grand, count_all, count_grand/count_all*100)

