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



def batchify(config, all_inputs):
    """
    Batchify the all the instances ids.
    :param all_inputs:  [[words,chars, labels],[words,chars,labels],...]
    :return:
        zero paddding for word, char, and batch length
        word_seq: batch_size x max_sent_len
        word_seq_lens batch_size ,
        char_seq: (batch_size * max_sent_len, max_word_len)
        char_seq_lens: (batch_size * max_sent_len, 1)
        char_seq_rec: batch_size * max_sent_len, 1
        label_seq_tensor: batch_size, max_sent_len
        mask: batch_size, max_sent_len
    """
    batch_size = len(all_inputs)
    words = [sent[0] for sent in all_inputs] ## word ids.
    chars = [sent[1] for sent in all_inputs]
    labels = [sent[2] for sent in all_inputs]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros(batch_size, max_seq_len).long().to(config.device)
    label_seq_tensor = torch.zeros(batch_size, max_seq_len).long().to(config.device)
    mask = torch.zeros(batch_size, max_seq_len).byte().to(config.device)
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask
