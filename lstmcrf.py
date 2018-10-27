import dynet as dy
from utils import log_sum_exp
import numpy as np

START = "<START>"
STOP = "<STOP>"

class BiLSTM_CRF:

    def __init__(self, config, model):
        self.num_layers = 1
        self.input_dim = config.embedding_dim
        self.model = model
        self.bilstm = dy.BiRNNBuilder(1, self.input_dim, config.hidden_dim, self.model,dy.LSTMBuilder)
        self.bilstm.set_dropout(config.dropout_bilstm)
        self.num_labels = len(config.label2idx)
        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        # print(config.hidden_dim)
        self.linear_w = self.model.add_parameters((self.num_labels, config.hidden_dim))
        self.linear_bias = self.model.add_parameters((self.num_labels))

        self.transition = self.model.add_lookup_parameters((self.num_labels, self.num_labels))
        vocab_size = len(config.word2idx)
        self.word2idx = config.word2idx
        self.word_embedding = self.model.add_lookup_parameters((vocab_size, self.input_dim))
        # self.emb_dropout = config.emb_dropout


    ## computing the negative loglikelihood
    def build_graph(self, x, is_train):
        dy.renew_cg()
        embeddings = [self.word_embedding[self.word2idx[w]] for w in x]
        if not is_train:
            self.bilstm.disable_dropout()
        lstm_out = self.bilstm.transduce(embeddings)
        ###check out the usage for transudce
        # print(len(lstm_out))
        # print(lstm_out[0].dim()) ## 100 x 1
        # print("w dim: " + str(self.linear_w.dim()))
        # print("bias dim: " + str(self.linear_bias.dim()))
        dy.affine_transform([self.linear_bias, self.linear_w, lstm_out[0]])
        features = [dy.affine_transform([self.linear_bias, self.linear_w, rep]) for rep in lstm_out]
        ##check out the usage of this part.
        return features

    def forward_unlabeled(self, features):
        ##unlabeled:
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

    def forward_labeled(self, features, tags):
        ##labeled network
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.label2idx[w] for w in tags]
        tags = [self.label2idx[START]] + tags
        for i, obs in enumerate(features):
            score = score + dy.pick(self.transition[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        labeled_score = score + dy.pick(self.transition[self.label2idx[STOP]], tags[-1])

        return labeled_score

    def negative_log(self, x, y):
        features = self.build_graph(x, True)
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

    def decode(self, x):
        features = self.build_graph(x, False)
        best_path, path_score = self.viterbi_decoding(features)
        best_path = [self.labels[x] for x in best_path]
        print(best_path)
        # print('path_score:', path_score.value())
        return best_path



