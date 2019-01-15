# 
# @author: Allan
#

import torch
import torch.nn as nn

from config.utils import START, STOP, log_sum_exp_pytorch
from pytorch_model.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence

class NNCRF(nn.Module):

    def __init__(self, config):
        super(NNCRF, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]

        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0

        self.transition = nn.Parameter(init_transition)

        self.input_size = config.embedding_dim
        if self.use_char:
            self.char_feature = CharBiLSTM(config)
            self.input_size += config.charlstm_hidden_dim

        vocab_size = len(config.word2idx)
        self.word_embedding = nn.Embedding(vocab_size, config.embedding_dim).to(self.device)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.word_embedding))
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
        self.hidden2tag = nn.Linear(config.hidden_dim, self.label_size).to(self.device)

    def neural_scoring(self, words, word_seq_lens, char_inputs, char_seq_lens, char_seq_recover):
        """
        :param words: (batch_size, sent_len)
        :param word_seq_lens: (batch_size, 1)
        :param chars: (batch_size * sent_len * word_length)
        :param char_seq_lens: numpy (batch_size * sent_len , 1)
        :param char_seq_recover: variable which records the char order information, used to recover char order
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        batch_size = words.size(0)
        sent_len = words.size(1)

        word_emb = self.word_embedding(words)
        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lens)
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            word_emb = torch.cat([word_emb, char_features], 2)
        word_rep = self.word_drop(word_emb)

        packed_words = pack_padded_sequence(word_rep, word_seq_lens, True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out, (seq_len, batch_size, hidden_size)
        lstm_out = lstm_out.transpose(1,0)
        outputs = self.hidden2tag(lstm_out)

        return outputs


    def calculate_all_scores(self, features):
        batch_size = features.size(0)
        seq_len = features.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    features.view(batch_size,seq_len, 1, self.label_size).expand(batch_size,seq_len,self.label_size, self.label_size)
        return scores

    def forward_unlabeled(self, all_scores, word_seq_lens, masks):
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        return torch.sum(last_alpha)

    def forward_labeled(self, all_scores, word_seq_lens, tags, masks):
        '''
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, label_size
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)

        return torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:])) + torch.sum(
            tagTransScoresEnd)

    def neg_log_obj(self, words, word_seq_lens, chars, char_seq_lens, char_seq_recover, masks, tags):
        features = self.neural_scoring(words, word_seq_lens, chars, char_seq_lens, char_seq_recover)
        all_scores = self.calculate_all_scores(features)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens, masks)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, masks)
        return unlabed_score - labeled_score

    def viterbiDecode(self, all_scores, word_seq_lens):
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        # sent_len =
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size]).to(self.device)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64).to(self.device)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64).to(self.device)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx

    def decode(self, batchInput):
        wordSeqTensor, wordSeqLengths, _, charSeqTensor, charSeqLengths, char_seq_recover, tagSeqTensor, _ = batchInput
        features = self.neural_scoring(wordSeqTensor, wordSeqLengths,charSeqTensor,charSeqLengths, char_seq_recover)
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbiDecode(all_scores, wordSeqLengths)
        # print(bestScores, decodeIdx)
        return bestScores, decodeIdx
