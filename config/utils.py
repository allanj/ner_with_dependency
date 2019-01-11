import numpy as np
import dynet as dy


START = "<START>"
STOP = "<STOP>"

import torch

def log_sum_exp(scores, num_labels):
    max_score_expr = dy.max_dim(scores)
    max_score_expr_broadcast = dy.concatenate([max_score_expr] * num_labels)
    # return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))
    '''
    sum_cols(x) has been deprecated.
    Please use sum_dim(x, [1]) instead.
    '''
    return max_score_expr + dy.log(dy.sum_dim(dy.exp(scores - max_score_expr_broadcast), [0]))




def log_sum_exp_pytorch(vec):
    """

    :param vec: [batchSize * from_label * to_label]
    :return: [batchSize * tagSize]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[1]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))
