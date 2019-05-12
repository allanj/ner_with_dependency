from config.reader import Reader

from common.sentence import Sentence
from common.instance import Instance
from typing import List
from tqdm import tqdm
import numpy as np

import seaborn as sns; sns.set(font_scale=0.8)
import matplotlib.pyplot as plt
import random


def read_results(res_file: str, number: int = -1) -> List[Instance]:
    print("Reading file: " + res_file)
    insts = []
    # vocab = set() ## build the vocabulary
    with open(res_file, 'r', encoding='utf-8') as f:
        words = []
        heads = []
        deps = []
        labels = []
        tags = []
        preds = []
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if line == "":
                inst = Instance(Sentence(words, heads, deps, tags), labels)
                inst.prediction = preds
                insts.append(inst)
                words = []
                heads = []
                deps = []
                labels = []
                tags = []
                preds = []

                if len(insts) == number:
                    break
                continue
            vals = line.split()
            word = vals[1]
            pos = vals[2]
            head = int(vals[3])
            dep_label = vals[4]

            label = vals[5]
            pred_label = vals[6]

            words.append(word)
            heads.append(head)  ## because of 0-indexed.
            deps.append(dep_label)
            tags.append(pos)
            labels.append(label)
            preds.append(pred_label)
    print("number of sentences: {}".format(len(insts)))
    return insts


# file = "data/ontonotes/test.sd.conllx"
# digit2zero = False
# reader = Reader(digit2zero)
#
# insts = reader.read_conll(file, -1, True)

file = "final_results/lstm_2_200_crf_ontonotes_sd_-1_dep_feat_emb_elmo_none_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
insts = read_results(file)  ##change inst.output -> inst.prediction

entities = set([ label[2:] for inst in insts for label in inst.output if len(label)>1])
print(entities)
dep_labels = set([ dep for inst in insts for label, dep in zip(inst.prediction, inst.input.dep_labels) if len(label)>1]  )
print(len(dep_labels), dep_labels)

ent2idx = {}
ents = list(entities)
ents.sort()
for i, label in enumerate(ents):
    ent2idx[label] = i


dep2idx = {}
deps = list(dep_labels)
deps.sort()
for i, label in enumerate(deps):
    dep2idx[label] = i

ent_dep_mat = np.zeros((len(entities), len(dep_labels)))
print(ent_dep_mat.shape)
for inst in insts:
    for label, dep in zip(inst.prediction, inst.input.dep_labels):
        if label == "O":
            continue
        ent_dep_mat[ent2idx[label[2:]]] [dep2idx[dep]] += 1

sum_labels = [ sum(ent_dep_mat[i]) for i in range(ent_dep_mat.shape[0])]
ent_dep_mat =   np.stack([ (ent_dep_mat[i]/sum_labels[i]) * 100 for i in range(ent_dep_mat.shape[0])], axis=0)
print(ent_dep_mat.shape)

indexs = [i for i in range(ent_dep_mat.shape[1]) if len(ent_dep_mat[:,i][ ent_dep_mat[:,i] >5.0 ]) ]
print(np.asarray(deps)[indexs])

xlabels = [deps[i] for i in indexs]
# cmap = sns.light_palette("#2ecc71", as_cmap=True)
# cmap = sns.light_palette("#8e44ad", as_cmap=True)
cmap = sns.cubehelix_palette(8,as_cmap=True)
ax = sns.heatmap(ent_dep_mat[:, indexs], annot=True, vmin=0, vmax=100, cmap=cmap,fmt='.0f', xticklabels=xlabels, yticklabels=ents, cbar=True)
                # ,annot_kws = {"size": 10})
                # , cbar_kws={'label': 'percentage (%)'})
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.show()