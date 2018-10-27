
import argparse
import random
import numpy as np
import dynet as dy
from config import  Config
from reader import Reader
from lstmcrf import BiLSTM_CRF
import eval
from tqdm import tqdm

def setSeed(seed, useGpu=False):
    random.seed(seed)
    np.random.seed(seed)

def parse_arguments(parser):
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--digit2zero', action="store_true", default=True)
    parser.add_argument('--train_file', type=str, default="data/conll2003/train.txt")
    parser.add_argument('--dev_file', type=str, default="data/conll2003/dev.txt")
    parser.add_argument('--test_file', type=str, default="data/conll2003/test.txt")
    # parser.add_argument('--embedding_file', type=str, default="data/emb/glove.6B.100d.txt")
    parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.015)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)

    ##model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=100)

    return parser.parse_args()

def train(epoch, insts, dev_insts, test_insts):

    model = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(model, learning_rate=config.learning_rate)

    bicrf = BiLSTM_CRF(config, model)

    for i in range(epoch):
        epoch_loss = 0
        for inst in tqdm(insts):
            loss = bicrf.negative_log(inst.input.words, inst.output)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            epoch_loss += loss_value
        print(epoch_loss)
        ## evaluation
        for dev_inst in dev_insts:
            dev_inst.prediction =  bicrf.decode(dev_inst.input.words)
        metrics = eval.evaluate(dev_insts)
        print("precision "+str(metrics[0]) + " recall:" +str(metrics[1])+" f score : " + str(metrics[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    config = Config(opt)

    reader = Reader(config.digit2zero)
    setSeed(config.seed, opt.gpu)

    train_insts = reader.read_from_file(config.train_file, config.train_num)
    dev_insts = reader.read_from_file(config.dev_file, config.dev_num)
    test_insts = reader.read_from_file(config.test_file, config.test_num)
    print(reader.all_vocab)

    config.build_emb_table(reader.all_vocab)

    config.use_iobes(train_insts)
    config.use_iobes(dev_insts)
    config.use_iobes(test_insts)
    config.build_label_idx(train_insts)
    config.build_char_idx(train_insts)
    config.build_char_idx(dev_insts)
    config.build_char_idx(test_insts)

    print("num chars: " + str(config.num_char))
    print(str(config.char2idx))

    print("num words: " + str(len(config.word2idx)))
    print(config.word2idx)

    train(config.num_epochs, train_insts, dev_insts, test_insts)

    print(opt.mode)