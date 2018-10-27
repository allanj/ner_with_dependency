# 
# @author: Allan
#

from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
import re

class Reader:


    def __init__(self, digit2zero):
        self.digit2zero = digit2zero
        self.all_vocab = {}


    def read_from_file(self, file, number=-1):
        print("Reading file: " + file)
        insts = []
        # vocab = set() ## build the vocabulary
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words), labels))
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                word, _, label = line.split()
                if self.digit2zero:
                    word = re.sub('\d', '0', word)
                words.append(word)
                self.all_vocab[word] = 0
                labels.append(label)
        return insts


