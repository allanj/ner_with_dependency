# 
# @author: Allan
#
from common.sentence import  Sentence
class Instance:

    def __init__(self, input: Sentence, output):
        self.input = input
        self.output = output
        self.elmo_vec = None

    def __len__(self):
        return len(self.input)
