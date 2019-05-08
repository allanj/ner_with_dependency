# 
# @author: Allan
#

from config.reader import Reader

file = "data/ontonotes/train.sd.conllx"
digit2zero = False
reader = Reader(digit2zero)

insts = reader.read_conll(file, -1, True)
# devs = reader.read_conll(conf.dev_file, conf.dev_num, False)
# tests = reader.read_conll(conf.test_file, conf.test_num, False)

out_dep_label2num = {}

out_doubledep2num = {}

out_word2num = {}

label2idx = {}

def not_entity(label:str):
    if label.startswith("B-") or label.startswith("I-"):
        return False
    return True

def is_entity(label:str):
    if label.startswith("B-") or label.startswith("I-"):
        return True
    return False

for inst in insts:
    output = inst.output
    sent = inst.input

    for idx, (word, head_idx, ent, dep) in enumerate(zip(sent.words, sent.heads, output, sent.dep_labels)):
        if dep not in label2idx:
            label2idx[dep] = len(label2idx)
        if is_entity(ent):
            if head_idx == -1 or not_entity(output[head_idx]):
                if dep in out_dep_label2num:
                    out_dep_label2num[dep] +=1
                else:
                    out_dep_label2num[dep] = 1
                head_word = "root" if head_idx == -1 else sent.words[head_idx]
                if head_word in out_word2num:
                    out_word2num[head_word] += 1
                else:
                    out_word2num[head_word] = 1

                if head_idx != -1:
                    head_dep = sent.dep_labels[head_idx]
                    if (head_dep, dep) in out_doubledep2num:
                        out_doubledep2num[(head_dep, dep)] += 1
                    else:
                        out_doubledep2num[(head_dep, dep)] = 1


counts = [(key, out_dep_label2num[key]) for key in out_dep_label2num]
counts = sorted(counts, key=lambda vals: vals[1], reverse=True)
total_ent_dep = sum([nums[1] for nums in counts])
print(counts)
print("total is {}".format(total_ent_dep))


# counts = [(key, out_word2num[key]) for key in out_word2num]
# counts = sorted(counts, key=lambda vals: vals[1], reverse=True)
# total_ent_dep = sum([nums[1] for nums in counts])
# print(counts)
# print("total is {}".format(total_ent_dep))


counts = [(key, out_doubledep2num[key]) for key in out_doubledep2num]
counts = sorted(counts, key=lambda vals: vals[1], reverse=True)
total_ent_dep = sum([nums[1] for nums in counts])
print(counts)
print("total is {}".format(total_ent_dep))
