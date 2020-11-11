
import argparse
import random
import numpy as np
from config.reader import Reader
from config import eval, context_models
from config.config import Config, ContextEmb, DepModelType
import time
from model.lstmcrf import NNCRF
import torch
import torch.optim as optim
import torch.nn as nn
from config.utils import lr_decay, simple_batching, get_spans, preprocess, bert_batching
from config.transformers_util import tokenize_instance, get_huggingface_optimizer_and_scheduler
from typing import List
from common.instance import Instance
from termcolor import colored
import os
from tqdm import tqdm

def setSeed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--digit2zero', action="store_true", default=True)
    parser.add_argument('--dataset', type=str, default="ontonotes")
    parser.add_argument('--affix', type=str, default="sd")
    # parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt")
    parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--learning_rate', type=float, default=2e-5) ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_num', type=int, default=-1)
    parser.add_argument('--dev_num', type=int, default=-1)
    parser.add_argument('--test_num', type=int, default=-1)
    parser.add_argument('--eval_freq', type=int, default=4000, help="evaluate frequency (iteration)")
    parser.add_argument('--eval_epoch', type=int, default=0, help="evaluate the dev set after this number of epoch")

    ## model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--num_lstm_layer', type=int, default=1, help="number of lstm layers")
    parser.add_argument('--dep_emb_size', type=int, default=50, help="embedding size of dependency")
    parser.add_argument('--dep_hidden_dim', type=int, default=200, help="hidden size of gcn, tree lstm")

    ### NOTE: GCN parameters, useless if we are not using GCN
    parser.add_argument('--num_gcn_layers', type=int, default=1, help="number of gcn layers")
    parser.add_argument('--gcn_mlp_layers', type=int, default=1, help="number of mlp layers after gcn")
    parser.add_argument('--gcn_dropout', type=float, default=0.5, help="GCN dropout")
    parser.add_argument('--gcn_adj_directed', type=int, default=0, choices=[0, 1], help="GCN ajacent matrix directed")
    parser.add_argument('--gcn_adj_selfloop', type=int, default=0, choices=[0, 1], help="GCN selfloop in adjacent matrix, now always false as add it in the model")
    parser.add_argument('--gcn_gate', type=int, default=0, choices=[0, 1], help="add edge_wise gating")

    ##NOTE: this dropout applies to many places
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=0, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--dep_model', type=str, default="none", choices=["none", "dggcn", "dglstm"], help="dependency method")
    parser.add_argument('--inter_func', type=str, default="mlp", choices=["concatenation", "addition",  "mlp"], help="combination method, 0 concat, 1 additon, 2 gcn, 3 more parameter gcn")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "bert", "elmo", "flair"], help="contextual word embedding")

    parser.add_argument('--embedder_type', type=str, default="roberta-base",
                        choices=["normal"] + list(context_models.keys()),
                        help="normal means word embedding + char, otherwise you can use 'bert-base-cased' and so on")
    parser.add_argument('--parallel_embedder', type=int, default=0,
                        choices=[0, 1],
                        help="use parallel training for those (BERT) models in the transformers. Parallel on GPUs")



    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)

def batching_list_instances(config: Config, insts:List[Instance]):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        if config.embedder_type != "normal":
            batched_data.append(bert_batching(config, one_batch_insts))
        else:
            batched_data.append(simple_batching(config, one_batch_insts))

    return batched_data

def learn_from_insts(config:Config, epoch: int, train_insts, dev_insts, test_insts):
    # train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance], batch_size: int = 1
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)



    batched_data = batching_list_instances(config, train_insts)
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    model = NNCRF(config)
    if config.embedder_type == "normal":

        optimizer = get_optimizer(config, model)
    else:
        print(
            colored(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}",
                    'red'))
        print(
            colored(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.", 'red'))
        print(colored(
            f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications.",
            'red'))
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model,
                                                                       num_training_steps=len(batched_data) * epoch,
                                                                       weight_decay=0.0,
                                                                       eps=1e-8,
                                                                       warmup_step=0)
        print(colored(f"[Optimizer Info] Modify the optimizer info as you need.", 'red'))
        print(optimizer)
    model.to(config.device)
    best_dev = [-1, 0]
    best_test = [-1, 0]

    dep_model_name = config.dep_model.name
    if config.dep_model == DepModelType.dggcn:
        dep_model_name += '(' + str(config.num_gcn_layers) + "," + str(config.gcn_dropout) + "," + str(
            config.gcn_mlp_layers) + ")"
    model_name = "model_files/lstm_{}_{}_crf_{}_{}_{}_dep_{}_elmo_{}_{}_gate_{}_epoch_{}_lr_{}_comb_{}.m".format(config.num_lstm_layer, config.hidden_dim, config.dataset, config.affix, config.train_num, dep_model_name, config.context_emb.name, config.optimizer.lower(), config.edge_gate, epoch, config.learning_rate, config.interaction_func)
    res_name = "results/lstm_{}_{}_crf_{}_{}_{}_dep_{}_elmo_{}_{}_gate_{}_epoch_{}_lr_{}_comb_{}.results".format(config.num_lstm_layer, config.hidden_dim, config.dataset, config.affix, config.train_num, dep_model_name, config.context_emb.name, config.optimizer.lower(), config.edge_gate, epoch, config.learning_rate, config.interaction_func)
    print("[Info] The model will be saved to: %s, please ensure models folder exist" % (model_name))
    if not os.path.exists("model_files"):
        os.makedirs("model_files")
    if not os.path.exists("results"):
        os.makedirs("results")

    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in tqdm(np.random.permutation(len(batched_data)), desc="--training batch", total=len(batched_data)):
        # for index in range(len(batched_data)):
            model.train()
            loss = model.neg_log_obj(**batched_data[index])
            epoch_loss += loss.item()
            loss.backward()
            if config.dep_model == DepModelType.dggcn:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip) ##clipping the gradient
            optimizer.step()
            model.zero_grad()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        if i + 1 >= config.eval_epoch:
            model.eval()
            dev_metrics = evaluate(config, model, dev_batches, "dev", dev_insts)
            test_metrics = evaluate(config, model, test_batches, "test", test_insts)
            if dev_metrics[2] > best_dev[0]:
                print("saving the best model...")
                best_dev[0] = dev_metrics[2]
                best_dev[1] = i
                best_test[0] = test_metrics[2]
                best_test[1] = i
                torch.save(model.state_dict(), model_name)
                write_results(res_name, test_insts)
            model.zero_grad()

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    evaluate(config, model, test_batches, "test", test_insts)
    write_results(res_name, test_insts)



def evaluate(config:Config, model: NNCRF, batch_insts_ids, name:str, insts: List[Instance]):
    ## evaluation
    metrics = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    with torch.no_grad():
        for batch in tqdm(batch_insts_ids, desc="Evaluating instances", total=len(batch_insts_ids)):
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            # sorted_batch_insts = sorted(one_batch_insts, key=lambda inst: len(inst.input.words), reverse=True)
            batch_max_scores, batch_max_ids = model.decode(batch)
            metrics += eval.evaluate_num(one_batch_insts, batch_max_ids, batch["labels"], batch["word_seq_len"], config.idx2labels)
            batch_id += 1
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall,fscore), flush=True)
    return [precision, recall, fscore]


def test_model(config: Config, test_insts):
    dep_model_name = config.dep_model.name
    if config.dep_model == DepModelType.dggcn:
        dep_model_name += '(' + str(config.num_gcn_layers) + ","+str(config.gcn_dropout)+ ","+str(config.gcn_mlp_layers)+")"
    model_name = "model_files/lstm_{}_{}_crf_{}_{}_{}_dep_{}_elmo_{}_{}_gate_{}_epoch_{}_lr_{}_comb_{}.m".format(config.num_lstm_layer, config.hidden_dim,
                                                                                                                                      config.dataset, config.affix,
                                                                                                                                      config.train_num,
                                                                                                                                      dep_model_name,
                                                                                                                                      config.context_emb.name,
                                                                                                                                      config.optimizer.lower(),
                                                                                                                                      config.edge_gate,
                                                                                                                                      config.num_epochs,
                                                                                                                                      config.learning_rate, config.interaction_func)
    res_name = "results/lstm_{}_{}_crf_{}_{}_{}_dep_{}_elmo_{}_{}_gate_{}_epoch_{}_lr_{}_comb_{}.results".format(config.num_lstm_layer, config.hidden_dim,
                                                                                                                                      config.dataset, config.affix,
                                                                                                                                      config.train_num,
                                                                                                                                      dep_model_name,
                                                                                                                                      config.context_emb.name,
                                                                                                                                      config.optimizer.lower(),
                                                                                                                                      config.edge_gate,
                                                                                                                                      config.num_epochs,
                                                                                                                                      config.learning_rate, config.interaction_func)
    model = NNCRF(config)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    test_batches = batching_list_instances(config, test_insts)
    evaluate(config, model, test_batches, "test", test_insts)
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
            assert  len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i, words[i], tags[i], heads[i], dep_labels[i], output[i], prediction[i]))
        f.write("\n")
    f.close()






def main():
    parser = argparse.ArgumentParser(description="Dependency-Guided LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    setSeed(opt, conf.seed)

    trains = reader.read_conll(conf.train_file, -1, True)
    devs = reader.read_conll(conf.dev_file, conf.dev_num, False)
    tests = reader.read_conll(conf.test_file, conf.test_num, False)

    if conf.context_emb != ContextEmb.none:
        print('Loading the {} vectors for all datasets.'.format(conf.context_emb.name))
        conf.context_emb_size = reader.load_elmo_vec(conf.train_file.replace(".sd", "").replace(".ud", "").replace(".sud", "").replace(".predsd", "").replace(".predud", "").replace(".stud", "").replace(".ssd", "") + "."+conf.context_emb.name+".vec", trains)
        reader.load_elmo_vec(conf.dev_file.replace(".sd", "").replace(".ud", "").replace(".sud", "").replace(".predsd", "").replace(".predud", "").replace(".stud", "").replace(".ssd", "")  + "."+conf.context_emb.name+".vec", devs)
        reader.load_elmo_vec(conf.test_file.replace(".sd", "").replace(".ud", "").replace(".sud", "").replace(".predsd", "").replace(".predud", "").replace(".stud", "").replace(".ssd", "")  + "."+conf.context_emb.name+".vec", tests)

    conf.use_iobes(trains + devs + tests)
    conf.build_label_idx(trains)

    conf.build_deplabel_idx(trains + devs + tests)
    print("# deplabels: ", len(conf.deplabels))
    print("dep label 2idx: ", conf.deplabel2idx)

    if conf.embedder_type == "normal":
        conf.build_word_idx(trains, devs, tests)
        conf.build_emb_table()
        conf.map_insts_ids(trains + devs + tests)


        print("num chars: " + str(conf.num_char))
        # print(str(config.char2idx))

        print("num words: " + str(len(conf.word2idx)))
        # print(config.word2idx)
    else:
        """
        If we use the pretrained model from transformers
        we need to use the pretrained tokenizer
        """
        print(colored(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer", "red"))
        tokenize_instance(context_models[conf.embedder_type]["tokenizer"].from_pretrained(conf.embedder_type),
                          trains + devs + tests, conf.label2idx)
    if opt.mode == "train":
        if conf.train_num != -1:
            random.shuffle(trains)
            trains = trains[:conf.train_num]
        learn_from_insts(conf, conf.num_epochs, trains, devs, tests)
    else:
        ## Load the trained model.
        test_model(conf, tests)
        # pass

    print(opt.mode)

if __name__ == "__main__":
    main()