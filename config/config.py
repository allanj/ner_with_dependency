# 
# @author: Allan
#

import numpy as np
from tqdm import tqdm
from typing import List
from common.instance import Instance
from config.utils import PAD, START, STOP, ROOT, ROOT_DEP_LABEL, SELF_DEP_LABEL
import torch
from enum import Enum
from termcolor import colored

class DepModelType(Enum):
    none = 0
    dglstm = 1
    dggcn = 2


class ContextEmb(Enum):
    none = 0
    elmo = 1
    bert = 2
    flair = 3


class InteractionFunction(Enum):
    concatenation = 0
    addition = 1
    mlp = 2



class Config:
    def __init__(self, args):

        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.ROOT = ROOT
        self.UNK = "<UNK>"
        self.unk_id = -1
        self.root_dep_label = ROOT_DEP_LABEL
        self.self_label = SELF_DEP_LABEL

        print(colored("[Info] remember to chec the root dependency label if changing the data. current: {}".format(self.root_dep_label), "red"  ))

        # self.device = torch.device("cuda" if args.gpu else "cpu")
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        self.context_emb = ContextEmb[args.context_emb]
        self.context_emb_size = 0
        self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero

        self.dataset = args.dataset

        self.affix = args.affix
        train_affix = self.affix.replace("pred", "") if "pred" in self.affix else self.affix
        self.train_file = "data/" + self.dataset + "/train."+train_affix+".conllx"
        self.dev_file = "data/" + self.dataset + "/dev."+train_affix+".conllx"
        self.test_file = "data/" + self.dataset + "/test."+self.affix+".conllx"
        self.label2idx = {}
        self.idx2labels = []
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0


        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        # self.lr_decay = 0.05
        self.use_dev = True
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)

        self.hidden_dim = args.hidden_dim
        self.num_lstm_layer = args.num_lstm_layer
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 30
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn
        # self.use_head = args.use_head
        self.dep_model = DepModelType[args.dep_model]

        self.dep_hidden_dim = args.dep_hidden_dim
        self.num_gcn_layers = args.num_gcn_layers
        self.gcn_mlp_layers = args.gcn_mlp_layers
        self.gcn_dropout = args.gcn_dropout
        self.adj_directed = args.gcn_adj_directed
        self.adj_self_loop = args.gcn_adj_selfloop
        self.edge_gate = args.gcn_gate

        self.dep_emb_size = args.dep_emb_size
        self.deplabel2idx = {}
        self.deplabels = []


        self.eval_epoch = args.eval_epoch


        self.interaction_func = InteractionFunction[args.inter_func] ## 0:concat, 1: addition, 2:gcn


        self.embedder_type = args.embedder_type if "embedder_type" in args.__dict__ else "normal"


    # def print(self):
    #     print("")
    #     print("\tuse gpu: " + )

    '''
      read all the  pretrain embeddings
    '''
    def read_pretrain_embedding(self):
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        embedding_dim = -1
        embedding = dict()
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if len(tokens) == 2:
                    continue
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    # print(tokens)
                    # print(embedding_dim)
                    # assert (embedding_dim + 1 == len(tokens))
                    if (embedding_dim + 1) != len(tokens):
                        continue
                    pass
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                first_col = tokens[0]
                embedding[first_col] = embedd
        return embedding, embedding_dim


    def build_word_idx(self, train_insts, dev_insts, test_insts):
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = 1
        self.unk_id = 1
        self.idx2word.append(self.UNK)

        self.word2idx[self.ROOT] = 2
        self.idx2word.append(self.ROOT)

        self.char2idx[self.PAD] = 0
        self.idx2char.append(self.PAD)
        self.char2idx[self.UNK] = 1
        self.idx2char.append(self.UNK)

        ##extract char on train, dev, test
        for inst in train_insts + dev_insts + test_insts:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        ##extract char only on train
        for inst in train_insts:
            for word in inst.input.words:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)
        # print(self.idx2word)
        # print(self.idx2char)
        # for idx, char in enumerate(self.idx2char):
        #     print(idx, ":", char)
        # print("separator")
        # for idx, word in enumerate(self.idx2word):
        #     print(idx, ":", word)
    '''
        build the embedding table
        obtain the word2idx and idx2word as well.
    '''
    def build_emb_table(self):
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            word_found_in_emb_vocab = 0
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                    word_found_in_emb_vocab += 1
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                    word_found_in_emb_vocab += 1
                else:
                    # self.word_embedding[self.word2idx[word], :] = self.embedding[self.UNK]
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            print(f"[Info] {word_found_in_emb_vocab} out of {len(self.word2idx)} found in the pretrained embedding.")
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

    def build_deplabel_idx(self, insts):
        if self.self_label not in self.deplabel2idx:
            self.deplabels.append(self.self_label)
            self.deplabel2idx[self.self_label] = len(self.deplabel2idx)
        for inst in insts:
            for label in inst.input.dep_labels:
                if label not in self.deplabels:
                    self.deplabels.append(label)
                    self.deplabel2idx[label] = len(self.deplabel2idx)
        self.root_dep_label_id = self.deplabel2idx[self.root_dep_label]

    def build_label_idx(self, insts):
        self.label2idx[self.PAD] = len(self.label2idx)
        self.idx2labels.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)

        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        self.label_size = len(self.label2idx)
        print("#labels: " + str(self.label_size))
        print("label 2idx: " + str(self.label2idx))

    def use_iobes(self, insts):
        for inst in insts:
            output = inst.output
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(self.B):
                        output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        output[pos] = curr_entity.replace(self.I, self.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(self.B):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.I, self.E)

    def map_insts_ids(self, insts: List[Instance]):
        insts_ids = []
        for inst in insts:
            words = inst.input.words
            inst.word_ids = []
            inst.char_ids = []
            inst.dep_label_ids = []
            inst.dep_head_ids = []
            inst.output_ids = []
            for word in words:
                if word in self.word2idx:
                    inst.word_ids.append(self.word2idx[word])
                else:
                    inst.word_ids.append(self.word2idx[self.UNK])
                char_id = []
                for c in word:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[self.UNK])
                inst.char_ids.append(char_id)
            for i, head in enumerate(inst.input.heads):
                if head == -1:
                    inst.dep_head_ids.append(i) ## appended it self.
                else:
                    inst.dep_head_ids.append(head)
            for label in inst.input.dep_labels:
                inst.dep_label_ids.append(self.deplabel2idx[label])
            for label in inst.output:
                inst.output_ids.append(self.label2idx[label])
            insts_ids.append([inst.word_ids, inst.char_ids, inst.output_ids])
        return insts_ids
