# 
# @author: Allan
#

from typing import List

class Sentence:

    def __init__(self, words: List[str], heads: List[int]=None , dep_labels: List[str]=None, pos_tags:List[str] = None,
                 ori_words: List[str] = None):
        self.words = words
        self.heads = heads
        self.dep_labels = dep_labels
        self.pos_tags = pos_tags
        self.ori_words = ori_words

    def __len__(self):
        return len(self.words)








# if __name__ == "__main__":
#
#     words = ["a" ,"sdfsdf"]
#     sent = Sentence(words)
#
#     print(len(sent))