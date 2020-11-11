# 
# @author: Allan
#

import torch
import torch.nn as nn

from config.utils import START, STOP, PAD, log_sum_exp_pytorch
from model.charbilstm import CharBiLSTM
from model.deplabel_gcn import DepLabeledGCN
from model.transformers_embedder import TransformersEmbedder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config.config import DepModelType, ContextEmb, InteractionFunction
import torch.nn.functional as F

def to_device(tensors, device):
    res = ()
    for tensor in tensors:
        res += tensor.to(device)
    return res

class NNCRF(nn.Module):

    def __init__(self, config):
        super(NNCRF, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.dep_model = config.dep_model
        self.context_emb = config.context_emb
        self.interaction_func = config.interaction_func


        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]



        self.input_size = config.embedding_dim
        self.embedder_type = config.embedder_type
        if config.embedder_type == "normal":
            if self.use_char:
                self.char_feature = CharBiLSTM(config)
                self.input_size += config.charlstm_hidden_dim
            vocab_size = len(config.word2idx)
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
            self.word_drop = nn.Dropout(config.dropout).to(self.device)

            """
                Input size to LSTM description
            """
            self.charlstm_dim = config.charlstm_hidden_dim
            if self.dep_model == DepModelType.dglstm:
                self.input_size += config.embedding_dim + config.dep_emb_size
                if self.use_char:
                    self.input_size += config.charlstm_hidden_dim

            if self.context_emb != ContextEmb.none:
                self.input_size += config.context_emb_size
            self.embedding_dim = config.embedding_dim
        else:
            self.word_embedding = TransformersEmbedder(config)
            self.input_size = self.word_embedding.get_output_dim()
            if self.dep_model == DepModelType.dglstm:
                self.input_size += self.input_size + config.dep_emb_size
            self.embedding_dim = self.word_embedding.get_output_dim()

        if self.dep_model == DepModelType.dglstm and self.interaction_func == InteractionFunction.mlp:
            self.mlp_layers = nn.ModuleList()
            for i in range(config.num_lstm_layer - 1):
                self.mlp_layers.append(nn.Linear(config.hidden_dim, 2 * config.hidden_dim).to(self.device))
            self.mlp_head_linears = nn.ModuleList()
            for i in range(config.num_lstm_layer - 1):
                self.mlp_head_linears.append(nn.Linear(config.hidden_dim, 2 * config.hidden_dim).to(self.device))

        print("[Model Info] Input size to LSTM: {}".format(self.input_size))
        print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_dim))


        num_layers = 1
        if config.num_lstm_layer > 1 and self.dep_model != DepModelType.dglstm:
            num_layers = config.num_lstm_layer
        if config.num_lstm_layer > 0:
            self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True).to(self.device)

        self.num_lstm_layer = config.num_lstm_layer
        self.lstm_hidden_dim = config.hidden_dim

        if config.num_lstm_layer > 1 and self.dep_model == DepModelType.dglstm:
            self.add_lstms = nn.ModuleList()
            if self.interaction_func == InteractionFunction.concatenation or \
                    self.interaction_func == InteractionFunction.mlp:
                hidden_size = 2 * config.hidden_dim
            elif self.interaction_func == InteractionFunction.addition:
                hidden_size = config.hidden_dim

            print("[Model Info] Building {} more LSTMs, with size: {} x {} (without dep label highway connection)".format(config.num_lstm_layer-1, hidden_size, config.hidden_dim))
            for i in range(config.num_lstm_layer - 1):
                self.add_lstms.append(nn.LSTM(hidden_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device))

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)


        final_hidden_dim = config.hidden_dim if self.num_lstm_layer >0 else self.input_size
        """
        Model description
        """
        print("[Model Info] Dep Method: {}, hidden size: {}".format(self.dep_model.name, config.dep_hidden_dim))
        if self.dep_model != DepModelType.none:
            print("Initializing the dependency label embedding")
            self.dep_label_embedding = nn.Embedding(len(config.deplabel2idx), config.dep_emb_size).to(self.device)
            if self.dep_model == DepModelType.dggcn:
                self.gcn = DepLabeledGCN(config, config.hidden_dim)  ### lstm hidden size
                final_hidden_dim = config.dep_hidden_dim

        print("[Model Info] Final Hidden Size: {}".format(final_hidden_dim))
        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)

        init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        self.transition = nn.Parameter(init_transition)


    def neural_scoring(self, word_seq_tensor, word_seq_len, context_emb = None, chars = None,
                       char_seq_lens = None, batch_dep_heads = None, dep_label_tensor =None,
                       orig_to_tok_index: torch.Tensor = None,
                       input_mask: torch.Tensor = None, **kwargs):
        """
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :param chars: (batch_size * sent_len * word_length)
        :param char_seq_lens: numpy (batch_size * sent_len , 1)
        :param dep_label_tensor: (batch_size, max_sent_len)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        batch_size = word_seq_tensor.size(0)
        sent_len = max(word_seq_len)

        if self.embedder_type == "normal":
            word_emb = self.word_embedding(word_seq_tensor)
            if self.use_char:
                if self.dep_model == DepModelType.dglstm:
                    char_features = self.char_feature.get_last_hiddens(chars, char_seq_lens)
                    word_emb = torch.cat((word_emb, char_features), 2)
        else:
            word_emb = self.word_embedding(word_seq_tensor, orig_to_tok_index, input_mask)
        if self.dep_model == DepModelType.dglstm:
            size = self.embedding_dim if (not self.use_char or self.embedder_type!= "normal") else (self.embedding_dim + self.charlstm_dim)
            dep_head_emb = torch.gather(word_emb, 1, batch_dep_heads.view(batch_size, sent_len, 1).expand(batch_size, sent_len, size))

        if self.context_emb != ContextEmb.none:
            word_emb = torch.cat((word_emb, context_emb.to(self.device)), 2)

        if self.use_char and self.embedder_type == "normal":
            if self.dep_model != DepModelType.dglstm:
                char_features = self.char_feature.get_last_hiddens(chars, char_seq_lens)
                word_emb = torch.cat((word_emb, char_features), 2)

        """
          Word Representation
        """
        if self.dep_model == DepModelType.dglstm:
            dep_emb = self.dep_label_embedding(dep_label_tensor)
            word_emb = torch.cat((word_emb, dep_head_emb, dep_emb), 2)

        if self.embedder_type == "normal":
            word_rep = self.word_drop(word_emb)
        else:
            word_rep = word_emb

        sorted_seq_len, permIdx = word_seq_len.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]


        if self.num_lstm_layer > 0:
            packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
            lstm_out, _ = self.lstm(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
            feature_out = self.drop_lstm(lstm_out)
        else:
            feature_out = sorted_seq_tensor

        """
        Higher order interactions
        """
        if self.num_lstm_layer > 1 and (self.dep_model == DepModelType.dglstm):
            for l in range(self.num_lstm_layer-1):
                dep_head_emb = torch.gather(feature_out, 1, batch_dep_heads[permIdx].view(batch_size, sent_len, 1).expand(batch_size, sent_len, self.lstm_hidden_dim))
                if self.interaction_func == InteractionFunction.concatenation:
                    feature_out = torch.cat((feature_out, dep_head_emb), 2)
                elif self.interaction_func == InteractionFunction.addition:
                    feature_out = feature_out + dep_head_emb
                elif self.interaction_func == InteractionFunction.mlp:
                    feature_out = F.relu(self.mlp_layers[l](feature_out) + self.mlp_head_linears[l](dep_head_emb))

                packed_words = pack_padded_sequence(feature_out, sorted_seq_len, True)
                lstm_out, _ = self.add_lstms[l](packed_words, None)
                lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
                feature_out = self.drop_lstm(lstm_out)


        outputs = self.hidden2tag(feature_out)

        return outputs[recover_idx]

    def calculate_all_scores(self, features):
        batch_size = features.size(0)
        seq_len = features.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    features.view(batch_size, seq_len, 1, self.label_size).expand(batch_size,seq_len,self.label_size, self.label_size)
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

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def neg_log_obj(self, word_seq_tensor, word_seq_len,
                    context_emb = None,
                    chars = None,
                    char_seq_lens = None, batch_dep_heads = None, labels = None, dep_label_tensor= None,
                    orig_to_tok_index: torch.Tensor = None,
                    input_mask: torch.Tensor = None):
        word_seq_tensor = word_seq_tensor.to(self.device)
        # word_seq_len = word_seq_len.to(self.device)
        context_emb = context_emb.to(self.device) if context_emb is not None else None
        chars = chars.to(self.device) if chars is not None else None
        char_seq_lens = char_seq_lens.to(self.device) if char_seq_lens is not None else None
        batch_dep_heads = batch_dep_heads.to(self.device) if batch_dep_heads is not None else None
        labels = labels.to(self.device) if labels is not None else None
        dep_label_tensor = dep_label_tensor.to(self.device) if dep_label_tensor is not None else None
        orig_to_tok_index = orig_to_tok_index.to(self.device) if orig_to_tok_index is not None else None
        input_mask = input_mask.to(self.device) if input_mask is not None else None
        features = self.neural_scoring(word_seq_tensor= word_seq_tensor,
                                       word_seq_len = word_seq_len,
                                       context_emb= context_emb,
                                       chars= chars,
                                       char_seq_lens = char_seq_lens, batch_dep_heads=batch_dep_heads,
                                       dep_label_tensor= dep_label_tensor,
                                       orig_to_tok_index= orig_to_tok_index, input_mask=input_mask)

        all_scores = self.calculate_all_scores(features)

        batch_size = word_seq_tensor.size(0)
        sent_len = all_scores.size(1)
        seq_len = word_seq_len.to(self.device)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, seq_len.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)

        unlabed_score = self.forward_unlabeled(all_scores, seq_len, mask)
        labeled_score = self.forward_labeled(all_scores, seq_len, labels, mask)
        return unlabed_score - labeled_score


    def viterbiDecode(self, all_scores, word_seq_lens):
        word_seq_lens = word_seq_lens.to(self.device)
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

    def decode(self, batch):
        for key, value in batch.items():
            if key == "word_seq_len":
                continue
            batch[key] = batch[key].to(self.device) if batch[key] is not None else None
        features = self.neural_scoring(**batch)
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbiDecode(all_scores, batch["word_seq_len"])
        return bestScores, decodeIdx
