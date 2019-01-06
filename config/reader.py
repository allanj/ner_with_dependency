# 
# @author: Allan
#

from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
from typing import List
import re

class Reader:


    def __init__(self, digit2zero:bool=True):
        self.digit2zero = digit2zero
        self.train_vocab = {}
        self.test_vocab = {}

    def read_conll(self, file: str, number: int = -1, is_train: bool = True) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        # vocab = set() ## build the vocabulary
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            heads = []
            deps = []
            labels = []
            tags = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words, heads, deps, tags), labels))
                    words = []
                    heads = []
                    deps = []
                    labels = []
                    tags = []
                    if len(insts) == number:
                        break
                    continue
                if "conll2003" in file:
                    word, pos, label, head, dep_label = line.split()
                else:
                    vals = line.split()
                    word = vals[1]
                    head = int(vals[6])
                    dep_label = vals[7]
                    pos = vals[3]
                    label = vals[10]
                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                heads.append(head - 1) ## because of 0-indexed.
                deps.append(dep_label)
                tags.append(pos)
                if is_train:
                    self.train_vocab[word]=0
                else:
                    self.test_vocab[word]=0
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts


