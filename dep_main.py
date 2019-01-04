
import argparse
import random
import numpy as np
from config.config import Config
from config.reader import Reader
from model.dep_lstmcrf import Dep_BiLSTM_CRF
from config import eval
# from tqdm import tqdm
# import math
import time
import dynet as dy


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)

def parse_arguments(parser):
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]
    for arg in dynet_args:
        parser.add_argument(arg)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--digit2zero', action="store_true", default=True)
    parser.add_argument('--dataset', type=str, default="all")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt")
    # parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--learning_rate', type=float, default=0.05) ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)

    ##model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--dep_emb_size', type=int, default=50, help="embedding size of dependency")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    # parser.add_argument('--tanh_hidden_dim', type=int, default=100)
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--use_head', type=int, default=1, choices=[0, 1], help="not use dependency")

    parser.add_argument('--train_num', type=int, default=-1)
    parser.add_argument('--dev_num', type=int, default=-1)
    parser.add_argument('--test_num', type=int, default=-1)
    parser.add_argument('--eval_freq', type=int, default=2000,help="evaluate frequency (iteration)")
    parser.add_argument('--eval_epoch',type=int, default=0, help="evaluate the dev set after this number of epoch")

    parser.add_argument("--save_param",type=int, choices=[0,1] ,default=0)

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def get_optimizer(model):

    if config.optimizer == "sgd":
        return dy.SimpleSGDTrainer(model, learning_rate=config.learning_rate)
    elif config.optimizer == "adam":
        return dy.AdamTrainer(model)

def train(epoch, insts, dev_insts, test_insts, batch_size = 1):

    model = dy.ParameterCollection()
    trainer = get_optimizer(model)

    bicrf = Dep_BiLSTM_CRF(config, model)
    trainer.set_clip_threshold(5)
    print("number of instances: %d" % (len(insts)))

    best_dev = [-1, 0]
    best_test = [-1, 0]
    # if batch_size != 1:
    #     batch_insts = batching(insts, batch_size)

    model_name= "models/lstm_crf_{}_{}_head_{}.m".format(config.dataset, config.train_num, config.use_head)
    print("[Info] The model will be saved to: %s, please ensure models folder exist" % (model_name))
    for i in range(epoch):
        epoch_loss = 0
        start_time = time.time()
        k = 0
        for index in np.random.permutation(len(insts)):
            inst = insts[index]
            dy.renew_cg()
            input = inst.input.word_ids
            # input = config.insert_singletons(inst.input.word_ids)
            loss = bicrf.negative_log(input, inst.output, x_chars=inst.input.char_ids, heads=inst.input.heads, deplabels=inst.input.dep_label_ids)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            epoch_loss += loss_value
            k = k + 1

            if i + 1 >= config.eval_epoch and (k % config.eval_freq == 0 or k == len(insts)):
                dev_metrics, test_metrics = evaluate(bicrf, dev_insts, test_insts)
                if dev_metrics[2] > best_dev[0]:
                    best_dev[0] = dev_metrics[2]
                    best_dev[1] = i
                    model.save(model_name)
                    if config.save_param:
                        bicrf.save_shared_parameters()  ##Optional step
                if test_metrics[2] > best_test[0]:
                    best_test[0] = test_metrics[2]
                    best_test[1] = i
                k = 0
        end_time = time.time()

        print("Epoch %d: %.5f, Time is %.2fs" % (i + 1, epoch_loss, end_time-start_time), flush=True)
    print("The best dev: %.2f" % (best_dev[0]))
    print("The best test: %.2f" % (best_test[0]))
    # model.populate(model_name)
    # evaluate(bicrf, dev_insts, test_insts)
    # if config.save_param:
    #     bicrf.save_shared_parameters()

def evaluate(model, dev_insts, test_insts):
    ## evaluation
    for dev_inst in dev_insts:
        dy.renew_cg()
        dev_inst.prediction = model.decode(dev_inst.input.word_ids, dev_inst.input.char_ids, dev_inst.input.heads, deplabels=dev_inst.input.dep_label_ids)
    dev_metrics = eval.evaluate(dev_insts)
    # print("precision "+str(metrics[0]) + " recall:" +str(metrics[1])+" f score : " + str(metrics[2]))
    print("[Dev set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (dev_metrics[0], dev_metrics[1], dev_metrics[2]))
    ## evaluation
    for test_inst in test_insts:
        dy.renew_cg()
        test_inst.prediction = model.decode(test_inst.input.word_ids, test_inst.input.char_ids, test_inst.input.heads, deplabels=test_inst.input.dep_label_ids)
    test_metrics = eval.evaluate(test_insts)
    print("[Test set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (test_metrics[0], test_metrics[1], test_metrics[2]))
    return dev_metrics, test_metrics

if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    config = Config(opt)

    reader = Reader(config.digit2zero)
    setSeed(config.seed)

    train_insts = reader.read_conll(config.train_file, config.train_num, True)
    dev_insts = reader.read_conll(config.dev_file, config.dev_num, False)
    test_insts = reader.read_conll(config.test_file, config.test_num, False)

    config.use_iobes(train_insts)
    config.use_iobes(dev_insts)
    config.use_iobes(test_insts)
    config.build_label_idx(train_insts)

    config.build_deplabel_idx(train_insts)
    config.build_deplabel_idx(dev_insts)
    config.build_deplabel_idx(test_insts)
    print("# deplabels: ", config.deplabels)
    print("dep label 2idx: ", config.deplabel2idx)



    config.build_emb_table(reader.train_vocab, reader.test_vocab)

    config.find_singleton(train_insts)
    config.map_insts_ids(train_insts)
    config.map_insts_ids(dev_insts)
    config.map_insts_ids(test_insts)



    print("num chars: " + str(config.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(config.word2idx)))
    # print(config.word2idx)

    train(config.num_epochs, train_insts, dev_insts, test_insts, config.batch_size)

    print(opt.mode)
