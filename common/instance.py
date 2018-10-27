# 
# @author: Allan
#

# from common.sentence import Sentence

class Instance:

    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.input)




# if __name__ == "__main__":
#
#     words = ["a" ,"sdfsdf"]
#     inst  = Instance(Sentence(words), None)
#
#     print(len(inst))