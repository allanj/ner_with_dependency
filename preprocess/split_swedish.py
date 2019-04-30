# 
# @author: Allan
#

from tqdm import tqdm
import random

random.seed(42)





def preprocess(file_name):
    sents = []
    sent = []
    prev_label = 'O'
    lnum = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if line == "":
                if len(sent) > 0:
                    sents.append(sent)
                    sent = []
            else:
                vals = line.split()
                # print(lnum)
                # print(vals)
                label = vals[1]
                if label == '0':
                    label = 'O'

                if label != 'O':
                    if prev_label == 'O':
                        label = "B-" + label
                    else:
                        if prev_label[2:] == label:
                            label = "I-" + label
                        else:
                            label = "B-" + label
                prev_label = label

                word = vals[0]
                sent.append((word, label))

                if "test" in file_name:
                    if word == "." and label == "O":
                        sents.append(sent)
                        sent = []
                        prev_label = label
            lnum += 1
    random.shuffle(sents)

    return sents

def write_to_file(sents, output_file, start, end):
    out = open(output_file, 'w', encoding='utf-8')
    for sent in sents[start:end]:
        for word, label in sent:
            out.write(word + ' ' + label + '\n')
        out.write('\n')
    out.close()

#D:\Downloads\swedish-ner-corpus
file = "D:/Downloads/swedish-ner-corpus/train_corpus.txt"
sents = preprocess(file)
# write_to_file(sents, "data/swedish/train.txt", 0, 5163)
# write_to_file(sents, "data/swedish/dev.txt", 5163, len(sents))

file = "D:/Downloads/swedish-ner-corpus/test_corpus.txt"
test_sents = preprocess(file)
# print("number of sents:{}".format(len(sents)))
# write_to_file(sents, "data/swedish/test.txt", 0, len(sents))

all_sents = sents + test_sents
random.shuffle(all_sents)
num_train = int(len(all_sents) * 0.7)
num_dev = (len(all_sents) - num_train) // 2
num_test = len(all_sents) - num_train - num_dev
print("train: {}, dev: {}, test:{}".format(num_train, num_dev,num_test))
write_to_file(all_sents, "data/swedish/train.txt", 0, num_train)
write_to_file(all_sents, "data/swedish/dev.txt", num_train, num_train + num_dev)
write_to_file(all_sents, "data/swedish/test.txt", num_train + num_dev, len(all_sents))

