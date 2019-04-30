# 
# @author: Allan
#

from tqdm import tqdm
import random

random.seed(42)





def preprocess(file_name):
    sents = []
    sent = []
    lnum = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if line == "":
                if len(sent) > 0:
                    sents.append(sent)
                    sent = []
            else:
                word, label = line.split()
                # print(lnum)
                # print(vals)
                if label == "OUT":
                    label = "O"
                sent.append((word, label))
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
file = "data/af/Dataset.NCHLT-II.AF.NER.Full.txt"
sents = preprocess(file)

all_sents = sents
random.shuffle(all_sents)
num_train = int(len(all_sents) * 0.6)
num_dev = (len(all_sents) - num_train) // 2
num_test = len(all_sents) - num_train - num_dev
print("train: {}, dev: {}, test:{}".format(num_train, num_dev,num_test))
write_to_file(all_sents, "data/af/train.txt", 0, num_train)
write_to_file(all_sents, "data/af/dev.txt", num_train, num_train + num_dev)
write_to_file(all_sents, "data/af/test.txt", num_train + num_dev, len(all_sents))

