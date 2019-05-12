# 
# @author: Allan
#


from config.reader import Reader
import numpy as np

import matplotlib.pyplot as plt
import random

from common.sentence import Sentence
from common.instance import Instance
from typing import List
from config.eval import Span
from tqdm import tqdm

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

def evaluate(insts):

    p = {}
    total_entity = {}
    total_predict = {}

    for inst in insts:

        output = inst.output
        prediction = inst.prediction
        #convert to span
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
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))

        # total_entity += len(output_spans)
        # total_predict += len(predict_spans)
        # p += len(predict_spans.intersection(output_spans))

        for span in output_spans:
            length = span.right - span.left + 1
            if length >= 9:
                length = 9
            if length in total_entity:
                total_entity[length] += 1
            else:
                total_entity[length] = 1

        for span in predict_spans:
            length = span.right - span.left + 1
            if length >= 9:
                length = 9
            if length in total_predict:
                total_predict[length] += 1
            else:
                total_predict[length] = 1

        for span in predict_spans.intersection(output_spans):
            length = span.right - span.left + 1
            if length >= 9:
                length = 9
            if length in p:
                p[length] += 1
            else:
                p[length] = 1

    max_len = max([key for key in p])
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    f = {}
    for length in range(1, max_len + 1):
        if length not in p:
            continue
        precision = p[length] * 1.0 / total_predict[length] * 100 if total_predict[length] != 0 else 0
        recall = p[length] * 1.0 / total_entity[length] * 100 if total_entity[length] != 0 else 0
        f[length] = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return f


res1 = "./final_results/lstm_2_200_crf_ontonotes_sd_-1_dep_feat_emb_elmo_none_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
insts1 = read_conll(res1)

res2 = "./final_results/lstm_2_200_crf_ontonotes_sd_-1_dep_none_elmo_none_sgd_gate_0_base_-1_epoch_100_lr_0.01.results"
insts2 = read_conll(res2)

print(evaluate(insts1))
print(evaluate(insts2))
