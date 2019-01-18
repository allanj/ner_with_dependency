import numpy as np
# import dynet as dy
import torch
from typing import List
from common.instance import Instance

START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"


# def log_sum_exp(scores, num_labels):
#     max_score_expr = dy.max_dim(scores)
#     max_score_expr_broadcast = dy.concatenate([max_score_expr] * num_labels)
#     # return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))
#     '''
#     sum_cols(x) has been deprecated.
#     Please use sum_dim(x, [1]) instead.
#     '''
#     return max_score_expr + dy.log(dy.sum_dim(dy.exp(scores - max_score_expr_broadcast), [0]))


def log_sum_exp_pytorch(vec):
    """

    :param vec: [batchSize * from_label * to_label]
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def logSumExp(vec):
    """

    :param vec: [batchSize * tagSize * tagSize]
    :return: [batchSize * tagSize]
    """
    maxScores, idx = torch.max(vec, 2)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], vec.shape[1], 1).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 2))



def batchify(config, inputs):
    """
    Batchify the all the instances ids.
    :param inputs:  [[words,chars, labels],[words,chars,labels],...]
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
    batch_size = len(inputs)
    words = [sent[0] for sent in inputs] ## word ids.
    chars = [sent[1] for sent in inputs]
    labels = [sent[2] for sent in inputs]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros(batch_size, max_seq_len).long()
    label_seq_tensor = torch.zeros(batch_size, max_seq_len).long()
    mask = torch.zeros(batch_size, max_seq_len).byte()
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
    char_seq_tensor = torch.zeros(batch_size, max_seq_len, max_word_len).long()
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

    word_seq_tensor = word_seq_tensor.to(config.device)
    word_seq_lengths = word_seq_lengths.to(config.device)
    word_seq_recover = word_seq_recover.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    char_seq_tensor = char_seq_tensor.to(config.device)
    char_seq_recover = char_seq_recover.to(config.device)
    mask = mask.to(config.device)
    # print("word, ", word_seq_tensor[0])

    # print("char", char_seq_tensor[0])
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask



def simple_batching(config, insts: List[Instance]):
    """

    :param config:
    :param insts:
    :return:
        word_seq_tensor,
        word_seq_len,
        char_seq_tensor,
        char_seq_len,
        label_seq_tensor
    """
    batch_size = len(insts)
    batch_data = sorted(insts, key=lambda inst: len(inst.input.words), reverse=True)
    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()
    ### TODO: the 1 here might be used later?? We will make this as padding, because later we have to do a deduction.
    #### Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.words)) + [1] * (int(max_seq_len) - len(inst.input.words)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)
    for idx in range(batch_size):
        word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
        for word_idx in range(word_seq_len[idx]):
            char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
        for wordIdx in range(word_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, wordIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])   ###because line 119 makes it 1, every single character should have a id. but actually 0 is enough

    word_seq_tensor = word_seq_tensor.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    char_seq_tensor = char_seq_tensor.to(config.device)
    word_seq_len = word_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)

    return word_seq_tensor, word_seq_len, char_seq_tensor, char_seq_len, label_seq_tensor




def lr_decay(config, optimizer, epoch):
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer
