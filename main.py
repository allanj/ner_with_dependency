
import argparse
import random
import numpy as np
import dynet as dy
from config import  Config
from reader import Reader
from lstmcrf import BiLSTM_CRF
import eval
from tqdm import tqdm
import math

def setSeed(seed, dy_param):
    random.seed(seed)
    np.random.seed(seed)
    dy_param.set_random_seed(seed)

def parse_arguments(parser):
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--digit2zero', action="store_true", default=True)
    parser.add_argument('--train_file', type=str, default="data/conll2003/train.txt")
    parser.add_argument('--dev_file', type=str, default="data/conll2003/dev.txt")
    parser.add_argument('--test_file', type=str, default="data/conll2003/test.txt")
    # parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt")
    parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.015)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)

    ##model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=100)

    parser.add_argument('--train_num', type=int, default=100)
    parser.add_argument('--dev_num', type=int, default=100)
    parser.add_argument('--test_num', type=int, default=100)

    return parser.parse_args()

def batching(insts, batch_size):
    num_batchs = math.ceil(len(insts) / batch_size) ## Up round to the num batches
    batch_insts = []
    print("batching for %d batches. " % (num_batchs))
    for i in range(num_batchs):
        one_batch = []

        end = (i + 1) * batch_size if i != num_batchs - 1 else len(insts)
        # print("start %d, end  %d" % (i * batch_size, end))
        for k in range(i * batch_size, end):
            one_batch.append(insts[k])
        batch_insts.append(one_batch)
    return batch_insts

def train(epoch, insts, dev_insts, test_insts, batch_size = 1):

    model = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(model, learning_rate=config.learning_rate)

    bicrf = BiLSTM_CRF(config, model)
    trainer.set_clip_threshold(5)
    print("number of instances: %d" % (len(insts)))

    best_dev = [-1, 0]
    best_test = [-1, 0]
    if batch_size != 1:
        batch_insts = batching(insts, batch_size)
    for i in range(epoch):
        epoch_loss = 0
        if batch_size != 1:
            for minibatch in tqdm(batch_insts):
                dy.renew_cg()
                losses = []
                for inst in minibatch:
                    loss = bicrf.negative_log(inst.input.words, inst.output)
                    loss_value = loss.value()
                    losses.append(loss)
                    epoch_loss += loss_value
                loss = dy.esum(losses)
                loss.forward()
                loss.backward()
                trainer.update()
        else:
            # for inst in insts:
            for inst in tqdm(insts):
                dy.renew_cg()
                loss = bicrf.negative_log(inst.input.words, inst.output)
                loss_value = loss.value()
                loss.backward()
                trainer.update()
                epoch_loss += loss_value
        print("Epoch %d: %.5f" % (i + 1, epoch_loss))
        ## evaluation
        for dev_inst in dev_insts:
            dy.renew_cg()
            dev_inst.prediction =  bicrf.decode(dev_inst.input.words)
        metrics = eval.evaluate(dev_insts)
        # print("precision "+str(metrics[0]) + " recall:" +str(metrics[1])+" f score : " + str(metrics[2]))

        print("[Dev set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (metrics[0], metrics[1], metrics[2]))
        if metrics[2] > best_dev[0]:
            best_dev[0] = metrics[2]
            best_dev[1] = i
        ## evaluation
        for test_inst in test_insts:
            dy.renew_cg()
            test_inst.prediction = bicrf.decode(test_inst.input.words)
        metrics = eval.evaluate(test_insts)
        print("[Test set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (metrics[0], metrics[1], metrics[2]))
        if metrics[2] > best_test[0]:
            best_test[0] = metrics[2]
            best_test[1] = i

    print("The best dev: %.2f" % (best_dev[0]))
    print("The best test: %.2f" % (best_test[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    config = Config(opt)

    dyparams = dy.DynetParams()
    dyparams.set_autobatch(config.batch_size != 1)
    reader = Reader(config.digit2zero)
    setSeed(config.seed, dyparams)

    dyparams.init()

    train_insts = reader.read_from_file(config.train_file, config.train_num)
    dev_insts = reader.read_from_file(config.dev_file, config.dev_num)
    test_insts = reader.read_from_file(config.test_file, config.test_num)
    # print("All vocabulary")
    # print(reader.all_vocab)

    config.build_emb_table(reader.all_vocab)

    config.use_iobes(train_insts)
    config.use_iobes(dev_insts)
    config.use_iobes(test_insts)
    config.build_label_idx(train_insts)
    config.build_char_idx(train_insts)
    config.build_char_idx(dev_insts)
    config.build_char_idx(test_insts)

    print("num chars: " + str(config.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(config.word2idx)))
    # print(config.word2idx)

    train(config.num_epochs, train_insts, dev_insts, test_insts, config.batch_size)

    print(opt.mode)