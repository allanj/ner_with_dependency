# 
# @author: Allan
#


from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
from typing import List
from config.eval import evaluate
import random

def read_conll(res_file: str, number: int = -1) -> List[Instance]:
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

res1 = "../final_results/lstm_2_200_crf_ontonotes_chinese_sd_-1_dep_feat_emb_elmo_elmo_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
insts1 = read_conll(res1)

res2 = "../final_results/lstm_1_200_crf_ontonotes_chinese_sd_-1_dep_none_elmo_elmo_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
insts2 = read_conll(res2)


sample_num = 10000

p = 0
for i in range(sample_num):
    sinsts = []
    sinsts_2 = []
    for _ in range(len(insts1)):
        n = random.randint(0, len(insts1) - 1)
        sinsts.append(insts1[n])
        sinsts_2.append(insts2[n])

    f1 = evaluate(sinsts)[2]
    f2= evaluate(sinsts_2)[2]

    if f1 > f2:
        p += 1

    p_val = (i + 1 - p) / (i+1)
    print("current p value: {}".format(p_val))



