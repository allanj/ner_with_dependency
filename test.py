


import argparse
from enum import Enum


class DepMethod(Enum):
    none = 0
    feat_emb = 1
    tree_lstm = 2
    gcn = 3
    selfattn = 4

    # def __str__(self):
    #     return self.name

# for x in DepMethod:
#     print(x)
#
#
# q = DepMethod.none
#

print(DepMethod['10'])

# parser = argparse.ArgumentParser(description="LSTM CRF implementation")
#
#
#
#
# parser.add_argument('--dep_method', type=DepMethod, default=DepMethod.none)
#
#
#
#
# args = parser.parse_args()
#
#
# print(args.dep_method)

