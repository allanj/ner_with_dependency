# 
# @author: Allan
#

from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
from typing import List
import re
import pickle

class Reader:


    def __init__(self, digit2zero:bool=True):
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_conll(self, file: str, number: int = -1, is_train: bool = True) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        num_entity = 0
        # vocab = set() ## build the vocabulary
        find_root = False
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            heads = []
            deps = []
            labels = []
            tags = []
            ori_words = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words, heads, deps, tags, ori_words=ori_words), labels))
                    words = []
                    heads = []
                    deps = []
                    labels = []
                    tags = []
                    ori_words = []
                    find_root = False
                    if len(insts) == number:
                        break
                    continue
                # if "conll2003" in file:
                #     word, pos, head, dep_label, label = line.split()
                # else:
                vals = line.split()
                word = vals[1]
                head = int(vals[6])
                dep_label = vals[7]
                pos = vals[3]
                label = vals[10]
                ori_words.append(word)
                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                if head == 0 and find_root:
                    raise err("already have a root")
                heads.append(head - 1) ## because of 0-indexed.
                deps.append(dep_label)
                tags.append(pos)
                self.vocab.add(word)
                labels.append(label)
                if label.startswith("B-"):
                    num_entity +=1
        print("number of sentences: {}, number of entities: {}".format(len(insts), num_entity))
        return insts

    def read_txt(self, file: str, number: int = -1, is_train: bool = True) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        # vocab = set() ## build the vocabulary
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            tags = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words, None, None, tags), labels))
                    words = []
                    labels = []
                    tags = []
                    if len(insts) == number:
                        break
                    continue
                if "conll2003" in file:
                    word, pos, label = line.split()
                else:
                    vals = line.split()
                    word = vals[1]
                    pos = vals[3]
                    label = vals[10]
                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                tags.append(pos)
                self.vocab.add(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def load_elmo_vec(self, file, insts):
        f = open(file, 'rb')
        all_vecs = pickle.load(f)  # variables come out in the order you put them in
        f.close()
        size = 0
        for vec, inst in zip(all_vecs, insts):
            inst.elmo_vec = vec
            size = vec.shape[1]
            # print(str(vec.shape[0]) + ","+ str(len(inst.input.words)) + ", " + str(inst.input.words))
            assert(vec.shape[0] == len(inst.input.words))
        return size


