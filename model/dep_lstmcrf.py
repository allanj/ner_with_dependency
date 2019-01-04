import dynet as dy
from config.utils import log_sum_exp
import numpy as np
from model.char_rnn import CharRNN

START = "<START>"
STOP = "<STOP>"


class Dep_BiLSTM_CRF:

    def __init__(self, config, model):
        self.num_layers = 1
        self.input_dim = config.embedding_dim
        self.model = model
        self.use_char_rnn = config.use_char_rnn

        self.char_rnn = CharRNN(config, model) if self.use_char_rnn else None
        input_size = self.input_dim if not self.char_rnn else self.input_dim + config.charlstm_hidden_dim

        self.use_head = config.use_head

        hidden_size = config.hidden_dim
        if self.use_head:
            self.root_head = self.model.add_parameters((input_size))
            input_size *= 2
            input_size += config.dep_emb_size
        self.bilstm = dy.BiRNNBuilder(1, input_size, hidden_size, self.model,dy.LSTMBuilder)
        print("Input to word-level BiLSTM size: %d" % (input_size))
        print("BiLSTM hidden size: %d" % (config.hidden_dim))
        # self.bilstm.set_dropout(config.dropout_bilstm)
        self.num_labels = len(config.label2idx)
        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        # print(config.hidden_dim)


        self.linear_w = self.model.add_parameters((self.num_labels, hidden_size))
        self.linear_bias = self.model.add_parameters((self.num_labels,))

        dep_label_size = len(config.deplabels)
        self.transition = self.model.add_lookup_parameters((self.num_labels, self.num_labels))
        self.head_label_embeding = self.model.add_lookup_parameters((dep_label_size, config.dep_emb_size))
        vocab_size = len(config.word2idx)
        self.word2idx = config.word2idx
        print("Word Embedding size: %d x %d" % (vocab_size, self.input_dim))
        self.word_embedding = self.model.add_lookup_parameters((vocab_size, self.input_dim), init=config.word_embedding)

        self.dropout = config.dropout

    def save_shared_parameters(self):
        print("Saving the encoder parameter")
        # self.word_embedding.save("models/word_embedding.m")
        dy.save("basename", [self.char_rnn.char_emb, self.char_rnn.fw_lstm, self.char_rnn.bw_lstm, self.word_embedding, self.bilstm])

    def lstm_scoring(self, x, is_train, all_chars, heads, dep_labels):
        if self.use_char_rnn:
            embeddings = []
            for w, chars in zip(x, all_chars):
                word_emb = self.word_embedding[w]
                f, b = self.char_rnn.forward_char(chars)
                concat = dy.concatenate([word_emb, f, b])
                embeddings.append(dy.dropout(concat, self.dropout) if is_train else concat)
        else:
            embeddings = [dy.dropout(self.word_embedding[w], self.dropout) if is_train else self.word_embedding[w] for w in x ]

        if self.use_head:
            head_emb = [embeddings[head] if head != -1 else self.root_head for head in heads]
            head_label_emb = [self.head_label_embeding[label]  for label in dep_labels]
            word_reps =[dy.concatenate([emb, h_emb, hl_emb]) for emb, h_emb, hl_emb in zip(embeddings, head_emb, head_label_emb)]
        else:
            word_reps = embeddings
        lstm_out = self.bilstm.transduce(word_reps)
        features = [dy.affine_transform([self.linear_bias, self.linear_w, rep]) for rep in lstm_out]
        return features

    def forward_unlabeled(self, features):
        init_alphas = [-1e10] * self.num_labels
        init_alphas[self.label2idx[START]] = 0

        for_expr = dy.inputVector(init_alphas)
        for obs in features:
            alphas_t = []
            for next_tag in range(self.num_labels):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.num_labels)
                next_tag_expr = for_expr + self.transition[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr, self.num_labels))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transition[self.label2idx[STOP]]
        alpha = log_sum_exp(terminal_expr, self.num_labels)
        return alpha

    # Labeled network score
    def forward_labeled(self, features, tags):
        score = dy.scalarInput(0)
        tags = [self.label2idx[w] for w in tags]
        tags = [self.label2idx[START]] + tags
        for i, obs in enumerate(features):
            score = score + dy.pick(self.transition[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
        labeled_score = score + dy.pick(self.transition[self.label2idx[STOP]], tags[-1])

        return labeled_score

    def negative_log(self, x, y, x_chars=None, heads=None, deplabels = None):
        features = self.lstm_scoring(x, True, x_chars, heads, deplabels)
        # features = self.build_graph(x, True)
        unlabed_score = self.forward_unlabeled(features)
        labeled_score = self.forward_labeled(features, y)
        return unlabed_score - labeled_score

    def viterbi_decoding(self, features):
        backpointers = []
        init_vvars = [-1e10] * self.num_labels
        init_vvars[self.label2idx[START]] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transition[idx] for idx in range(self.num_labels)]
        for obs in features:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.num_labels):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs

            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[self.label2idx[STOP]]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == self.label2idx[START]
        # Return best path and best path's score
        return best_path, path_score

    def decode(self, x, x_chars=None, heads=None, deplabels=None):
        features = self.lstm_scoring(x, False, x_chars, heads, deplabels)
        # features = self.build_graph(x, False)
        best_path, path_score = self.viterbi_decoding(features)
        best_path = [self.labels[x] for x in best_path]
        # print(best_path)
        # print('path_score:', path_score.value())
        return best_path
