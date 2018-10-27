import numpy as np
import dynet as dy

def log_sum_exp(scores, num_labels):
    # return dy.logsumexp(scores.npvalue().tolist())
    npval = scores.npvalue()
    argmax_score = np.argmax(npval)
    max_score_expr = dy.pick(scores, argmax_score)
    max_score_expr_broadcast = dy.concatenate([max_score_expr] * num_labels)
    # return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))
    '''
    sum_cols(x) has been deprecated.
    Please use sum_dim(x, [1]) instead.
    '''
    return max_score_expr + dy.log(dy.sum_dim(dy.exp(scores - max_score_expr_broadcast), [0]))