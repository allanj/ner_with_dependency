# 
# @author: Allan
#
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharBiLSTM(nn.Module):

    # def __init__(self, config):
    #     super(CharBiLSTM, self).__init__()
    #     print("[Info] Building character-level LSTM")
    #     self.char_emb_size = config.char_emb_size
    #     self.char2idx = config.char2idx
    #     self.chars = config.idx2char
    #     self.char_size = len(self.chars)
    #     self.device = config.device
    #     self.hidden = config.charlstm_hidden_dim
    #
    #     self.char_embeddings = nn.Embedding(self.char_size, self.char_emb_size)
    #     self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(self.char_size, self.char_emb_size)))
    #     self.char_embeddings = self.char_embeddings.to(self.device)
    #
    #     self.char_lstm = nn.LSTM(self.char_emb_size, self.hidden//2,num_layers=1, batch_first=True, bidirectional=True).to(self.device)

    def __init__(self):
        super(CharBiLSTM, self).__init__()
        print("[Info] Building character-level LSTM")
        self.char_emb_size = 25
        self.char_size = 47
        self.device = torch.device("cpu")
        self.hidden = 50

        self.char_embeddings = nn.Embedding(self.char_size, self.char_emb_size)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(self.char_size, self.char_emb_size)))
        self.char_embeddings = self.char_embeddings.to(self.device)

        self.char_lstm = nn.LSTM(self.char_emb_size, self.hidden//2,num_layers=1, batch_first=True, bidirectional=True).to(self.device)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: (batch_size, word_length)
                seq_lengths: (batch_size, ) 1 dimensional vector.
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_embeddings(input)
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, batch_first=True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        ## char_hidden = (h_t, c_t)
        #  char_hidden[0] = h_t = (2, batch_size, lstm_dimension)
        # char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)

    def forward(self, char_input, seq_lengths):
        return self.get_last_hiddens(char_input, seq_lengths)


# '''
# unit testing code
# '''
# if __name__ == "__main__":
#     print("test char bilstm")
#     cbilstm = CharBiLSTM()
#     input = torch.randint(0, 47, (10, 5),dtype=torch.long)
#     seq_lengths, _ = torch.randint(2, 5, (10,),dtype=torch.long).sort(0, descending=True)
#     res = cbilstm(input, seq_lengths)
#     print(res.size())

