

from config.reader import Reader
from config.eval import Span


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


file = "data/ontonotes/train.sd.conllx"
digit2zero = False
reader = Reader(digit2zero)

insts = reader.read_conll(file, -1, True)
use_iobes(insts)

for i in range(len(insts)):

    inst = insts[i]
    gold_spans = get_spans(inst.output)


    for span in gold_spans:
        ent_words = ' '.join(inst.input.words[span.left:span.right+1])
        # if ent_words.islower() and span.type != "DATE" and span.type != "ORDINAL" and span.type != "PERCENT"\
        #         and span.type != "CARDINAL" and span.type != "MONEY" and span.type != "QUANTITY" and span.type != "TIME" \
        #         and span.type != "NORP":
        #     print(ent_words + " " + span.type)
        if span.right - span.left >= 6:
            print(ent_words + " " + span.type)
    ## book of the dead.