# 
# @author: Allan
#
class Instance:

    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.input)
