import numpy as np
import torch
from typing import List
from common.instance import Instance
from config.eval import Span

START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"
ROOT = "<ROOT>"
ROOT_DEP_LABEL = "root"
SELF_DEP_LABEL = "self"


def log_sum_exp_pytorch(vec):
    """

    :param vec: [batchSize * from_label * to_label]
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))



def simple_batching(config, insts: List[Instance]):
    from config.config import DepModelType,ContextEmb
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
    batch_data = sorted(insts, key=lambda inst: len(inst.input.words), reverse=True) ##object-based not direct copy
    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()
    ### NOTE: the 1 here might be used later?? We will make this as padding, because later we have to do a deduction.
    #### Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.words)) + [1] * (int(max_seq_len) - len(inst.input.words)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    word_emb_tensor = None
    if config.context_emb != ContextEmb.none:
        emb_size = insts[0].elmo_vec.shape[1]
        word_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size))

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)
    adjs = None
    adjs_in = None
    adjs_out = None
    dep_label_adj = None
    dep_label_tensor = None
    batch_dep_heads = None
    trees = None
    graphs = None
    if config.dep_model != DepModelType.none:
        if  config.dep_model == DepModelType.dggcn:
            adjs = [ head_to_adj(max_seq_len, inst, config) for inst in batch_data]
            adjs = np.stack(adjs, axis=0)
            adjs = torch.from_numpy(adjs)
            dep_label_adj = [head_to_adj_label(max_seq_len, inst, config) for inst in batch_data]
            dep_label_adj = torch.from_numpy(np.stack(dep_label_adj, axis=0)).long()

        if config.dep_model == DepModelType.dglstm:
            batch_dep_heads = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            dep_label_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        # trees = [inst.tree for inst in batch_data]
    for idx in range(batch_size):
        word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
        if config.context_emb != ContextEmb.none:
            word_emb_tensor[idx, :word_seq_len[idx], :] = torch.from_numpy(batch_data[idx].elmo_vec)

        if config.dep_model == DepModelType.dglstm:
            batch_dep_heads[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].dep_head_ids)
            dep_label_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].dep_label_ids)
        for word_idx in range(word_seq_len[idx]):
            char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
        for wordIdx in range(word_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, wordIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])   ###because line 119 makes it 1, every single character should have a id. but actually 0 is enough

    word_seq_tensor = word_seq_tensor.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    char_seq_tensor = char_seq_tensor.to(config.device)
    word_seq_len = word_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)
    # if config.use_elmo:
    #     word_emb_tensor = word_emb_tensor.to(config.device)
    if config.dep_model != DepModelType.none:
        if config.dep_model == DepModelType.dglstm:
            batch_dep_heads = batch_dep_heads.to(config.device)
            dep_label_tensor = dep_label_tensor.to(config.device)

    return word_seq_tensor, word_seq_len, word_emb_tensor, char_seq_tensor, char_seq_len, adjs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, trees, label_seq_tensor, dep_label_tensor



def lr_decay(config, optimizer, epoch):
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer



def head_to_adj(max_len, inst, config):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    directed = config.adj_directed
    self_loop = False #config.adj_self_loop
    ret = np.zeros((max_len, max_len), dtype=np.float32)

    for i, head in enumerate(inst.input.heads):
        if head == -1:
            continue
        ret[head, i] = 1

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in range(len(inst.input.words)):
            ret[i, i] = 1

    return ret


def head_to_adj_directed(max_len, inst, direction):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((max_len, max_len), dtype=np.float32)

    for i, head in enumerate(inst.input.heads):
        if head == -1:
            continue
        if direction == "in":
            ret[i, head] = 1
        else:
            ret[head, i] = 1

    return ret


def head_to_adj_label(max_len, inst, config):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    directed = config.adj_directed
    self_loop = config.adj_self_loop

    dep_label_ret = np.zeros((max_len, max_len), dtype=np.long)

    for i, head in enumerate(inst.input.heads):
        if head == -1:
            continue
        dep_label_ret[head, i] = inst.dep_label_ids[i]

        if config.double_dep_label:
            dep_label_ret[i, head] = config.deplabel2idx[inst.input.dep_labels[i] + "_prime"]

    if not directed and not config.double_dep_label:
        dep_label_ret = dep_label_ret + dep_label_ret.T

    if self_loop:
        for i in range(len(inst.input.words)):
            dep_label_ret[i, i] = config.root_dep_label_id

    return dep_label_ret


def get_spans(output):
    output_spans = set()
    start = -1
    for i in range(len(output)):
        if output[i].startswith("B-"):
            start = i
        if output[i].startswith("E-"):
            end = i
            output_spans.add(Span(start, end, output[i][2:]))
        if output[i].startswith("S-"):
            output_spans.add(Span(i, i, output[i][2:]))
    return output_spans

def preprocess(conf, insts, file_type:str):
    print("[Preprocess Info]Doing preprocessing for the CoNLL-2003 dataset: {}.".format(file_type))
    for inst in insts:
        output = inst.output
        spans = get_spans(output)
        for span in spans:
            if span.right - span.left + 1 < 2:
                continue
            count_dep = 0
            for i in range(span.left, span.right + 1):
                if inst.input.heads[i] >= span.left and inst.input.heads[i] <= span.right:
                    count_dep += 1
            if count_dep != (span.right - span.left):

                for i in range(span.left, span.right + 1):
                    if inst.input.heads[i] < span.left or inst.input.heads[i] > span.right:
                        if i != span.right:
                            inst.input.heads[i] = span.right
                            inst.input.dep_labels[i] = "nn" if "sd" in conf.affix else "compound"