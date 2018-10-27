# 
# @author: Allan
#


class Sentence:


    def __init__(self, words):
        self.words = words


    def __len__(self):
        return len(self.words)




# if __name__ == "__main__":
#
#     words = ["a" ,"sdfsdf"]
#     sent = Sentence(words)
#
#     print(len(sent))