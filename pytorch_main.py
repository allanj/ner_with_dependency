
import argparse
import random
import numpy as np
from config.config import Config
from config.reader import Reader
from model.dep_lstmcrf import Dep_BiLSTM_CRF
from config import eval
from config.config import Config
# from tqdm import tqdm
# import math
import time
from pytorch_model.pytorch_lstmcrf import NNCRF
import torch
import torch.optim as optim
import torch.nn as nn
from config.utils import lr_decay, batchify
from typing import List
from common.instance import Instance


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
    parser.add_argument('--dataset', type=str, default="conll2003")
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
    parser.add_argument('--use_head', type=int, default=0, choices=[0, 1], help="not use dependency")

    parser.add_argument('--use_elmo', type=int, default=0, choices=[0, 1], help="use Elmo embedding or not")

    # parser.add_argument('--use2layerLSTM', type=int, default=0, choices=[0, 1], help="use 2 layer bilstm")
    parser.add_argument('--second_hidden_size', type=int, default=0, help="hidden size for 2nd bilstm layer")

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


def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        return optim.SGD(params, lr=config.learning_rate)
    elif config.optimizer.lower() == "adam":
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)

def learn_and_eval(config:Config, epoch: int, train_insts_ids, batch_size):
    # train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance], batch_size: int = 1
    model = NNCRF(config)
    optimizer = get_optimizer(config, model)
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    total_batch = train_num // batch_size + 1
    print("Shuffle the training instance")
    random.shuffle(train_insts_ids)
    batched_data = []
    for batch_id in range(total_batch):
        insts = train_insts[batch_id * batch_size:batch_id * (batch_size + 1)]
        batched_data.append(batchify(config, insts))
    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_name = "models/lstm_{}_{}_crf_{}_{}_head_{}_elmo_{}.m".format(config.hidden_dim, config.second_hidden_size, config.dataset, config.train_num, config.use_head, config.use_elmo)
    res_name = "results/lstm_{}_{}_crf_{}_{}_head_{}_elmo_{}.results".format(config.hidden_dim, config.second_hidden_size, config.dataset, config.train_num, config.use_head, config.use_elmo)
    print("[Info] The model will be saved to: %s, please ensure models folder exist" % (model_name))

    for i in range(epoch):
        epoch_loss = 0
        start_time = time.time()
        k = 0
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, epoch)
        for index in np.random.permutation(len(train_insts)):
            model.train()
            inst = train_insts[index]
            input = inst.input.word_ids
            # input = config.insert_singletons(inst.input.word_ids)
            loss = model.neg_log_obj(input, inst.output, x_chars=inst.input.char_ids, heads=inst.input.heads, deplabels=inst.input.dep_label_ids, elmo_vec=inst.elmo_vec)
            loss_value = loss.value()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip) ##clipping the gradient
            optimizer.update()
            epoch_loss += loss_value
            k = k + 1

            if i + 1 >= config.eval_epoch and (k % config.eval_freq == 0 or k == len(train_insts)):
                model.eval()
                dev_metrics = evaluate(model, dev_insts, "dev")
                test_metrics = evaluate(model, test_insts, "test")
                if dev_metrics[2] > best_dev[0]:
                    best_dev[0] = dev_metrics[2]
                    best_dev[1] = i
                    best_test[0] = test_metrics[2]
                    best_test[1] = i
                    model.save(model_name)
                k = 0
        end_time = time.time()

        print("Epoch %d: %.5f, Time is %.2fs" % (i + 1, epoch_loss, end_time-start_time), flush=True)
    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    model.populate(model_name)
    evaluate(model, test_insts, "test")
    write_results(res_name,test_insts)
    # if config.save_param:
    #     bicrf.save_shared_parameters()


def evaluate(model, insts, name:str):
    ## evaluation
    for inst in insts:
        dy.renew_cg()
        inst.prediction = model.decode(inst.input.word_ids, inst.input.char_ids, inst.input.heads, deplabels=inst.input.dep_label_ids, elmo_vec=inst.elmo_vec)
    metrics = eval.evaluate(insts)
    # print("precision "+str(metrics[0]) + " recall:" +str(metrics[1])+" f score : " + str(metrics[2]))
    print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, metrics[0], metrics[1], metrics[2]))
    return metrics


def test():
    model_name = "models/lstm_crf_{}_{}_head_{}.m".format(config.dataset, config.train_num, config.use_head)
    res_name = "results/lstm_crf_{}_{}_head_{}.results".format(config.dataset, config.train_num, config.use_head)
    model = dy.ParameterCollection()
    bicrf = Dep_BiLSTM_CRF(config, model)
    model.populate(model_name)
    evaluate(bicrf, test_insts, "test")
    write_results(res_name, test_insts)

def write_results(filename:str, insts):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.input)):
            words = inst.input.words
            tags = inst.input.pos_tags
            heads = inst.input.heads
            dep_labels = inst.input.dep_labels
            output = inst.output
            prediction = inst.prediction
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i, words[i], tags[i], heads[i], dep_labels[i], output[i], prediction[i]))
        f.write("\n")
    f.close()



if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    setSeed(conf.seed)

    trains = reader.read_conll(conf.train_file, conf.train_num, True)
    devs = reader.read_conll(conf.dev_file, conf.dev_num, False)
    tests = reader.read_conll(conf.test_file, conf.test_num, False)


    if conf.use_elmo:
        print('Loading the elmo vectors for all datasets.')
        reader.load_elmo_vec(conf.train_file + ".elmo.vec", trains)
        reader.load_elmo_vec(conf.dev_file + ".elmo.vec", devs)
        reader.load_elmo_vec(conf.test_file + ".elmo.vec", tests)

    conf.use_iobes(trains)
    conf.use_iobes(devs)
    conf.use_iobes(tests)
    conf.build_label_idx(trains)

    conf.build_deplabel_idx(trains)
    conf.build_deplabel_idx(devs)
    conf.build_deplabel_idx(tests)
    print("# deplabels: ", conf.deplabels)
    print("dep label 2idx: ", conf.deplabel2idx)

    conf.build_emb_table(reader.train_vocab, reader.test_vocab)

    conf.find_singleton(train)
    conf.map_insts_ids(trains)
    conf.map_insts_ids(devs)
    conf.map_insts_ids(test_insts)



    print("num chars: " + str(config.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(config.word2idx)))
    # print(config.word2idx)
    if opt.mode == "train":
        learn_and_eval(config.num_epochs, train_insts, dev_insts, test_insts, config.batch_size)
    else:
        ## Load the trained model.
        test()

    print(opt.mode)
