# 
# @author: Allan
#
from typing import List

type2num = {}

def extract(words: List[str], labels:List[str], heads:List[str], deps:List[str]):
    entity_pool = [] ## type, left, right
    completed_pool = []
    print(words)
    # print(labels)
    for i, label in enumerate(labels):
        if label == "_":
            continue
        if "|" in label:
            vals = label.split("|")
            for val in vals:
                if val.startswith("(") and val.endswith(")"):
                    completed_pool.append((i, i, val[1:-1]))
                elif val.startswith("("):
                    entity_pool.append((i, -1, val[1:]))
                elif val.endswith(")"):
                    found = False
                    for tup in entity_pool[::-1]:
                        start, end, cur = tup
                        if cur == val[:-1]:
                            completed_pool.append((start, i, cur))
                            entity_pool.remove(tup)
                            found = True
                            break
                    if not found:
                        raise Exception("not found the entity:{}".format(val))
                else:
                    raise Exception("not val type".format(val))
        else:
            if label.startswith("(") and label.endswith(")"):
                completed_pool.append((i, i, label[1:-1]))
            elif label.startswith("("):
                entity_pool.append((i, -1, label[1:]))
            elif label.endswith(")"):
                found = False
                for tup in entity_pool[::-1]:
                    start, end, cur = tup
                    if cur == label[:-1]:
                        completed_pool.append((start, i, cur))
                        entity_pool.remove(tup)
                        found = True
                        break
                if not found:
                    raise Exception("not found the entity:{}".format(label))
            else:
                raise Exception("not val type {}".format(label))
    assert (len(entity_pool) == 0)


    for i in range(len(words)):
        curr_pos = []
        for span in completed_pool:
            start, end, label = span
            if i >= start and i <= end:
                curr_pos.append(span)
        curr_pos = sorted(curr_pos, key=lambda span: span[1] - span[0])
        for span in curr_pos[1:]:
            completed_pool.remove(span)

    labels = ["O"] * len(words)
    visited = [False] * len(words)
    for span in completed_pool:
        start, end, label = span

        for check in visited[start:(end+1)]:
            if check:
                raise Exception("this position is checked.")

        if label not in ('person', 'loc', 'org'):
            label = 'misc'

        labels[start] = "B-"+label
        labels[(start+1):end] = ["I-" + label] * (end - start)
        visited[start: (end+1)] = [True] * (end-start + 1)

        if label in type2num:
            type2num[label] += 1
        else:
            type2num[label] = 1

    # print(labels)
    return labels


def read_all_sents(filename:str, out:str):
    print(filename)
    fres = open(out, 'w', encoding='utf-8')
    sents = []
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        heads = []
        deps = []
        labels = []
        pos_tags = []
        for line in f.readlines():
            line = line.rstrip()
            # print(line)
            if line.startswith("#"):
                continue
            if line == "":
                idx = 1
                labels = extract(words, labels, heads, deps)
                idx = 1
                for w, h, dep, label, pos_tag in zip(words, heads, deps, labels, pos_tags):
                    if dep == "sentence":
                        dep = "root"
                    fres.write("{}\t{}\t_\t{}\t{}\t_\t{}\t{}\t_\t_\t{}\n".format(idx, w, pos_tag, pos_tag, h, dep, label))
                    idx += 1
                fres.write('\n')

                words = []
                heads = []
                deps = []
                labels = []
                continue
            # 1	West	_	NNP	NNP	_	5	compound	_	_	B-MISC
            vals = line.split()
            idx = vals[0]
            word = vals[1]
            pos_tag = vals[4]
            head = vals[8]
            dep_label = vals[10]
            label = vals[12]
            words.append(word)
            pos_tags.append(pos_tag)
            heads.append(head)
            labels.append(label)
            deps.append(dep_label)
    fres.close()

def process(filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        heads = []
        deps =[]
        labels = []
        prev_label = "O"
        prev_raw_label = ""
        for line in f.readlines():
            line = line.rstrip()
            # print(line)
            if line.startswith("#"):
                prev_label = "O"
                prev_raw_label = ""
                continue
            if line == "":
                idx = 1
                for w, h, dep, label in zip(words, heads, deps, labels):
                    if dep == "sentence":
                        dep = "root"
                    fres.write("{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_\t{}\n".format(idx, w, h, dep, label))
                    idx += 1
                fres.write('\n')
                words = []
                heads = []
                deps = []
                labels = []
                prev_label = "O"
                continue
            #1	West	_	NNP	NNP	_	5	compound	_	_	B-MISC
            vals = line.split()
            idx = vals[0]
            word = vals[1]
            head = vals[8]
            dep_label = vals[10]
            label = vals[12]

            if label.startswith("("):
                if label.endswith(")"):
                    label = "B-" + label[1:-1]
                else:
                    label = "B-" + label[1:]
            elif label.startswith(")"):
                label = "I-" + label[:-1]
            else:
                if prev_label == "O":
                    label = "O"
                else:
                    if prev_raw_label.endswith(")"):
                        label = "O"
                    else:
                        label = "I-" + prev_label[2:]

            words.append(word)
            heads.append(head)
            labels.append(label)
            deps.append(dep_label)
            prev_label = label
            prev_raw_label = vals[12]
    fres.close()




# process("data/semeval10t1/en.train.txt", "data/semeval10t1/train.sd.conllx")
# process("data/semeval10t1/en.devel.txt", "data/semeval10t1/dev.sd.conllx")
# process("data/semeval10t1/en.test.txt", "data/semeval10t1/test.sd.conllx")

lang = "ca"
folder="sem" + lang
read_all_sents("data/"+folder+"/"+lang+".train.txt", "data/"+folder+"/train.sd.conllx")
print(type2num)
type2num = {}
read_all_sents("data/"+folder+"/"+lang+".devel.txt", "data/"+folder+"/dev.sd.conllx")

print(type2num)
type2num = {}
read_all_sents("data/"+folder+"/"+lang+".test.txt", "data/"+folder+"/test.sd.conllx")

print(type2num)