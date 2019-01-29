# 
# @author: Allan
#

import numpy as np
from tqdm import tqdm


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
        self.unk = "unk"
        self.unk_id = -1

        # self.device = torch.device("cuda" if args.gpu else "cpu")
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero

        self.dataset = args.dataset

        self.affix = ""

        if self.dataset == "conll2003":
            self.affix = "." + args.affix

        self.train_file = "data/"+self.dataset+"/train"+self.affix+".conllx"
        self.dev_file = "data/"+self.dataset+"/dev"+self.affix+".conllx"
        ## following datasets do not have development set
        if self.dataset in ("abc", "cnn", "mnb", "nbc", "p25", "pri", "voa"):
            self.dev_file = "data/" + self.dataset + "/test.conllx"
        self.test_file = "data/"+self.dataset+"/test"+self.affix+".conllx"
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
        self.batch_size = 10
        self.use_dev = True
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.batch_size = args.batch_size
        self.eval_freq = args.eval_freq

        self.hidden_dim = args.hidden_dim
        # self.tanh_hidden_dim = args.tanh_hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn
        self.use_head = args.use_head

        self.dep_emb_size = args.dep_emb_size
        self.deplabel2idx = {}
        self.deplabels = []


        self.save_param = args.save_param
        self.eval_epoch = args.eval_epoch

        self.use_elmo = args.use_elmo
        # self.use2layerLSTM = args.use2layerLSTM
        self.second_hidden_size = args.second_hidden_size



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
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    # print(tokens)
                    # print(embedding_dim)
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
    def build_emb_table(self, train_vocab, test_vocab):
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)

        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        self.word2idx[self.unk] = 1
        self.unk_id = 1
        self.idx2word.append(self.unk)

        self.char2idx[self.unk] = 0
        self.idx2char.append(self.unk)

        for word in train_vocab:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            for c in word:
                if c not in self.char2idx:
                    self.char2idx[c] = len(self.idx2char)
                    self.idx2char.append(c)

        for word in test_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)
        # print(self.word2idx)
        #print(self.char2idx)

        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[self.unk]
                    # self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

    def build_deplabel_idx(self, insts):
        for inst in insts:
            for label in inst.input.dep_labels:
                if label not in self.deplabels:
                    self.deplabels.append(label)
                    self.deplabel2idx[label] = len(self.deplabel2idx)


    def build_label_idx(self, insts):
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

    def map_insts_ids(self, insts):
        for inst in insts:
            words = inst.input.words
            inst.input.word_ids = []
            inst.input.char_ids = []
            inst.input.dep_label_ids = []
            for word in words:
                if word in self.word2idx:
                    inst.input.word_ids.append(self.word2idx[word])
                else:
                    inst.input.word_ids.append(self.word2idx[self.unk])
                char_id = []
                for c in word:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[self.unk])
                inst.input.char_ids.append(char_id)
            for label in inst.input.dep_labels:
                inst.input.dep_label_ids.append(self.deplabel2idx[label])
        #     for label in output:
        #         label_ids.append(self.label2idx[label])
        #     insts_ids.append([word_ids, char_ids, label_ids])
        #     word_ids = []
        #     char_ids = []
        #     label_ids = []
        # return insts_ids


    # def map_word_to_ids_in_insts(self, insts):
    #     for inst in insts:
    #         words = inst.input.words
    #         inst.input_ids = []
    #         for word in words:
    #             inst.input_ids.append(self.word2idx[word])

    def find_singleton(self, train_insts):
        freq = {}
        self.singleton = set()
        for inst in train_insts:
            words = inst.input.words
            for w in words:
                if w in freq:
                    freq[w] += 1
                else:
                    freq[w] = 1
        for w in freq:
            if freq[w] == 1:
                self.singleton.add(self.word2idx[w])

    def insert_singletons(self, words, p=0.5):
        """
        Replace singletons by the unknown word with a probability p.
        """
        new_words = []
        for word in words:
            if word in self.singleton and np.random.uniform() < p:
                new_words.append(self.unk_id)
            else:
                new_words.append(word)
        return new_words