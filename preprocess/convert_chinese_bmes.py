# 
# @author: Allan
#

### This file is used to convert the segmented Chinese into character-based represnetation
## In order to run the Yang and Zhang's code for comparision



def use_iobes(labels):
    for pos in range(len(labels)):
        curr_entity = labels[pos]
        if pos == len(labels) - 1:
            if curr_entity.startswith("B-"):
                labels[pos] = curr_entity.replace("B-", "S-")
            elif curr_entity.startswith("I-"):
                labels[pos] = curr_entity.replace("I-", "E-")
        else:
            next_entity = labels[pos + 1]
            if curr_entity.startswith("B-"):
                if next_entity.startswith("O") or next_entity.startswith("B-"):
                    labels[pos] = curr_entity.replace("B-", "S-")
            elif curr_entity.startswith("I-"):
                if next_entity.startswith("O") or next_entity.startswith("B-"):
                    labels[pos] = curr_entity.replace("I-", "E-")

def process_conllx(filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                use_iobes(labels)
                idx = 1
                for i in range(len(words)):
                    word = words[i]
                    label = labels[i]
                    if len(word) == 1:
                        fres.write(word +" " + label + "\n")
                    else:
                        if label == "O" or label.startswith("I-"):
                            for j in range(len(word)):
                                fres.write(str(word[j]) + " " + label+ "\n")
                        elif label.startswith("B-"):
                            fres.write(str(word[0]) + " " + label + "\n")
                            for j in range(1, len(word)):
                                fres.write(str(word[j]) + " I-" + label[2:] + "\n")
                        elif label.startswith("E-"):
                            for j in range(len(word) - 1):
                                fres.write(str(word[j]) + " I-" + label[2:] + "\n")
                            fres.write(str(word[len(word) - 1]) + " " + label + "\n")
                        elif label.startswith("S-"):
                            fres.write(str(word[0]) + " B-" + label[2:] + "\n")
                            for j in range(1, len(word) - 1):
                                fres.write(str(word[j]) + " I-" + label[2:] + "\n")
                            fres.write(str(word[len(word) - 1]) + " E-" + label[2:] + "\n")

                fres.write('\n')
                words = []
                tags = []
                labels = []
                continue
            #1	West	_	NNP	NNP	_	5	compound	_	_	B-MISC
            idx, word, _, pos , _, _, head, dep_label, _, _, label = line.split()
            words.append(word)
            tags.append(pos)
            labels.append(label)
    fres.close()

process_conllx("data/ontonotes_chinese/train.sd.conllx", "data/ontonotes_chinese/onto.train.char")
process_conllx("data/ontonotes_chinese/dev.sd.conllx", "data/ontonotes_chinese/onto.dev.char")
process_conllx("data/ontonotes_chinese/test.sd.conllx", "data/ontonotes_chinese/onto.test.char")