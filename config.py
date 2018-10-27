# 
# @author: Allan
#

import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim


class Config:
    def __init__(self, args):

        self.PAD = "<PAD>"
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

        # self.device = torch.device("cuda" if args.gpu else "cpu")
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.label2idx = {}
        self.idx2labels = []
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0


        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = 100
        self.lr_decay = 0.05
        self.batch_size = 10
        self.use_dev = True
        self.train_num = -1
        self.dev_num = -1
        self.test_num = -1

        self.hidden_dim = args.hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout_bilstm = 0.5
        self.char_emb_size = 25
        self.char_hidden_dim = 50
        self.use_char = False
        self.emb_dropout = 0.5

    # def print(self):
    #     print("")
    #     print("\tuse gpu: " + )

    '''
      read all the  pretrain embeddings
    '''
    def read_pretrain_embedding(self):
        print("reading the pretraing embedding")
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
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    assert (embedding_dim + 1 == len(tokens))
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                first_col = tokens[0]
                embedding[first_col] = embedd
        return embedding, embedding_dim


    '''
        build the embedding table
        obtain the word2idx and idx2word as well.
    '''
    def build_emb_table(self, vocab):
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)

        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        for word in vocab:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
        if self.embedding is not None:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])


    def build_label_idx(self, insts):
        self.label2idx[self.PAD] = 0
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

    def build_char_idx(self, insts):
        self.char2idx[self.PAD] = 0
        self.idx2char.append(self.PAD)
        for inst in insts:
            for word in inst.input.words:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.char2idx)
        # print("#curr char: " + str(self.num_char))

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


    def map_insts_ids(self, insts):
        insts_ids = []

        word_ids = []
        char_ids = []
        label_ids = []
        for inst in insts:
            words = inst.input.words
            output = inst.output
            for word in words:
                word_ids.append(self.word2idx[word])
                char_id = []
                for c in word:
                    char_id.append(self.char2idx[c])
                char_ids.append(char_id)
            for label in output:
                label_ids.append(self.label2idx[label])
            insts_ids.append([word_ids, char_ids, label_ids])
            word_ids = []
            char_ids = []
            label_ids = []
        return insts_ids
