
import numpy as np
from typing import List, Tuple, Dict

from collections import defaultdict, Counter

class Span:

    def __init__(self, left, right, type):
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))

    def to_str(self, sent):
        return str(sent[self.left: (self.right+1)]) + ","+self.type

## the input to the evaluation should already have
## have the predictions which is the label.
## iobest tagging scheme
def evaluate(insts):

    p = 0
    total_entity = 0
    total_predict = 0

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

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))

    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return [precision, recall, fscore]


def evaluate_num(batch_insts, batch_pred_ids, batch_gold_ids, word_seq_lens, idx2label) -> Tuple[Dict, Dict, Dict]:

    # p = 0
    # total_entity = 0
    # total_predict = 0

    batch_p_dict = defaultdict(int)
    batch_total_entity_dict = defaultdict(int)
    batch_total_predict_dict = defaultdict(int)

    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        #convert to span
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
                batch_total_entity_dict[output[i][2:]] += 1
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))
                batch_total_entity_dict[output[i][2:]] += 1
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
                batch_total_predict_dict[prediction[i][2:]] += 1
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))
                batch_total_predict_dict[prediction[i][2:]] += 1
        # total_entity += len(output_spans)
        # total_predict += len(predict_spans)
        # p += len(predict_spans.intersection(output_spans))
        correct_spans = predict_spans.intersection(output_spans)
        for span in correct_spans:
            batch_p_dict[span.type] += 1
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    # return np.asarray([p, total_predict, total_entity], dtype=int)
    return Counter(batch_p_dict), Counter(batch_total_predict_dict), Counter(batch_total_entity_dict)