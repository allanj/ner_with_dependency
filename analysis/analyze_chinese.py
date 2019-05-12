# 
# @author: Allan
#


from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
from typing import List
from config.eval import evaluate, Span
import random

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

res1 = "../final_results/lstm_2_200_crf_ontonotes_sd_-1_dep_feat_emb_elmo_elmo_sgd_gate_0_base_-1_epoch_1000_lr_0.01.results"
insts1 = read_conll(res1)

res2 = "../final_results/lstm_1_200_crf_ontonotes_sd_-1_dep_feat_emb_elmo_none_sgd_gate_0_base_-1_epoch_150_lr_0.01.results"
insts2 = read_conll(res2)

print(evaluate(insts1))
print(evaluate(insts2))
num = 0
total_entity = 0
type2num = {}
length2num = {}
dep_label2num = {}
gc2num = {}
for i in range(len(insts1)):

    first = insts1[i]
    second = insts2[i]
    gold_spans = get_spans(first.output)

    pred_first = get_spans(first.prediction)
    pred_second = get_spans(second.prediction)


    # for span in pred_first:
    #     if span in gold_spans and (span not in pred_second):
    for span in gold_spans:
        if span in pred_first and (span not in pred_second):
            num += 1
            print(span.to_str(first.input.words))
            if span.type in type2num:
                type2num[span.type] +=1
            else:
                type2num[span.type] = 1
            length = span.right - span.left + 1
            if length in length2num:
                length2num[length] += 1
            else:
                length2num[length] = 1

            for k in range(span.left, span.right + 1):
                if first.input.heads[k] == -1 or (first.input.heads[k] > span.right or first.input.heads[k] < span.left):
                    if first.input.dep_labels[k] in dep_label2num:
                        dep_label2num[first.input.dep_labels[k]] +=1
                    else:
                        dep_label2num[first.input.dep_labels[k]] = 1

                if first.input.heads[k]!= -1 and first.input.heads[first.input.heads[k]] != -1:
                    h = first.input.heads[first.input.heads[k]]
                    if first.input.dep_labels[k] + "," + first.input.dep_labels[h] in gc2num:
                        gc2num[first.input.dep_labels[k] + "," + first.input.dep_labels[h]] += 1
                    else:
                        gc2num[first.input.dep_labels[k] + "," + first.input.dep_labels[h]] = 1

        total_entity +=1

print(num, total_entity)
print(type2num)
print(length2num)


print(dep_label2num)
total_amount = sum([dep_label2num[key] for key in dep_label2num])
print(total_amount)

counts = [(key, dep_label2num[key]) for key in dep_label2num]
counts = sorted(counts, key=lambda vals: vals[1], reverse=True)
print(counts)


print(gc2num)
counts = [(key, gc2num[key]) for key in gc2num]
counts = sorted(counts, key=lambda vals: vals[1], reverse=True)
print(counts)
