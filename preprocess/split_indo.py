from tqdm import tqdm
import random

random.seed(42)

file = "/Users/allanjie/allan/data/ner-dataset-modified-dee/20k_mdee_gazz.conll.txt"



sents = []
sent = []
with open(file, 'r', encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        line = line.rstrip()
        if line == "":
            sents.append(sent)
            sent = []
        else:
            word, label = line.split()
            sent.append((word, label))

random.shuffle(sents)

print(len(sents))

train_ratio = 0.7

num_train = int(len(sents) * train_ratio)
num_dev = (len(sents) - num_train) // 2
num_test =  len(sents)  - num_train - num_dev


trainf = "data/indo/train.txt"
devf = "data/indo/dev.txt"
testf = "data/indo/test.txt"


out = open(trainf, 'w', encoding='utf-8')
for sent in sents[:num_train]:
    for word, label in sent:
        out.write(word + ' ' + label + '\n')
    out.write('\n')
out.close()


out = open(devf, 'w', encoding='utf-8')
for sent in sents[num_train:(num_train+num_dev)]:
    for word, label in sent:
        out.write(word + ' ' + label + '\n')
    out.write('\n')
out.close()

out = open(testf, 'w', encoding='utf-8')
for sent in sents[(num_train+num_dev):]:
    for word, label in sent:
        out.write(word + ' ' + label + '\n')
    out.write('\n')
out.close()