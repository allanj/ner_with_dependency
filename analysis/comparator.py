# 
# @author: Allan
#
from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
from typing import List



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



lgcn_file = "../final_results/lstm_200_crf_ontonotes_sd_-1_dep_lstm_lgcn_elmo_elmo_sgd_gate_0_epoch_100_lr_0.01.results"
elmo_file = "../final_results/lstm_200_crf_ontonotes_.sd_-1_dep_none_elmo_elmo_sgd_gate_0_epoch_100_lr_0.01.results"
lgcn_res = read_conll(lgcn_file)
elmo_res = read_conll(elmo_file)




total = 0
total_word = 0
for dep_inst, inst in zip(lgcn_res, elmo_res):
    gold = inst.output
    normal_pred = inst.prediction
    dep_pred = dep_inst.prediction
    words = inst.input.words
    heads = inst.input.heads
    dep_labels = inst.input.dep_labels
    have_error= False
    for idx in range(len(gold)):
        if normal_pred[idx] != dep_pred[idx]:
            if gold[idx] == dep_pred[idx]:
                print("{}\t{}\t{}\t{}\t{}\t{}\t".format(idx, words[idx], heads[idx] + 1, dep_labels[idx], gold[idx], normal_pred[idx]))
    print("")
