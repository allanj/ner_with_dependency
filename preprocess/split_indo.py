from tqdm import tqdm
import random

random.seed(42)

file = "data/galician/gold_nospace_IOB.txt"



sents = []
sent = []
with open(file, 'r', encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        line = line.rstrip()
        if line.startswith("#"):
            continue
        vals = line.split()
        # print(vals)
        label = vals[0]
        word = vals[1]
        sent.append((word, label))
        if word == ".":
            sents.append(sent)
            sent = []

random.shuffle(sents)

print(len(sents))

train_ratio = 0.7

num_train = int(len(sents) * train_ratio)
num_dev = (len(sents) - num_train) // 2
num_test =  len(sents)  - num_train - num_dev


trainf = "data/galician/train.txt"
devf = "data/galician/dev.txt"
testf = "data/galician/test.txt"


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