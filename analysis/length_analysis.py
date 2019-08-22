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

def evaluate(insts, maximum_length = 4):

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
            if length >= maximum_length:
                length = maximum_length
            if length in total_entity:
                total_entity[length] += 1
            else:
                total_entity[length] = 1

        for span in predict_spans:
            length = span.right - span.left + 1
            if length >= maximum_length:
                length = maximum_length
            if length in total_predict:
                total_predict[length] += 1
            else:
                total_predict[length] = 1

        for span in predict_spans.intersection(output_spans):
            length = span.right - span.left + 1
            if length >= maximum_length:
                length = maximum_length
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


def grand_child(insts1, insts2):
    num = 0
    gc_num = 0
    ld_num = 0
    for i in range(len(insts1)):

        first = insts1[i]
        second = insts2[i]
        inst = insts1[i]
        gold_spans = get_spans(first.output)

        pred_first = get_spans(first.prediction)
        pred_second = get_spans(second.prediction)

        # for span in pred_first:
        #     if span in gold_spans and (span not in pred_second):
        for span in gold_spans:
            if span in pred_first and (span not in pred_second):
                if span.right - span.left < 2:
                    continue
                num += 1
                # print(span.to_str(first.input.words))
                has_grand = False
                has_ld = False
                for k in range(span.left, span.right + 1):
                    if inst.input.heads[k] >= span.left and inst.input.heads[k] <= span.right:
                        head_i = inst.input.heads[k]
                        if abs(head_i - k) > 1:
                            has_ld = True
                        if head_i != -1 and inst.input.heads[head_i] >= span.left and inst.input.heads[
                            head_i] <= span.right:
                            has_grand = True

                if has_grand:
                    gc_num +=1
                if has_ld:
                    ld_num += 1
    return gc_num, ld_num, num


## Chinese Comparison
res1 = "./final_results/lstm_2_200_crf_ontonotes_chinese_sd_-1_dep_none_elmo_elmo_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
insts1 = read_conll(res1)

res2 = "./final_results/lstm_2_200_crf_ontonotes_chinese_sd_-1_dep_feat_emb_elmo_elmo_sgd_gate_0_base_-1_epoch_100_lr_0.01_doubledep_0_comb_3.results"
insts2 = read_conll(res2)

# Catalan Comparison
# res1 = "./final_results/lstm_2_200_crf_semca_sd_-1_dep_none_elmo_elmo_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
# insts1 = read_conll(res1)
#
# res2 = "./final_results/lstm_2_200_crf_semca_sd_-1_dep_feat_emb_elmo_elmo_sgd_gate_0_base_-1_epoch_300_lr_0.01_doubledep_0_comb_3.results"
# insts2 = read_conll(res2)


## Spanish Comparison
# res1 = "./final_results/lstm_2_200_crf_semes_sd_-1_dep_none_elmo_elmo_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
# insts1 = read_conll(res1)
#
# res2 = "./final_results/lstm_2_200_crf_semes_sd_-1_dep_feat_emb_elmo_elmo_sgd_gate_0_base_-1_epoch_300_lr_0.01_doubledep_0_comb_3.results"
# insts2 = read_conll(res2)




maximum_length = 6
print(evaluate(insts1, maximum_length))
print(evaluate(insts2, maximum_length))

print(grand_child(insts1, insts2))
