
import dynet as dy

class CharCNN:

    def __init__(self, config, model):
        self.char_emb_size = config.char_emb_size
        self.char2idx = config.char2idx
        self.chars = config.idx2char
        self.char_size = len(self.chars)

        self.char_emb = model.add_lookup_parameters((self.char_size, 1, 1, self.char_emb_size))

        self.bilstm = dy.BiRNNBuilder(1, self.input_dim, config.hidden_dim, self.model, dy.LSTMBuilder)

        self.cnn_w = model.add_parameters((1, config))