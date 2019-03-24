# 
# @author: Allan
#

from config.reader import  Reader
import numpy as np
import pickle
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)


def parse_sentence(tokenizer, model, words, mode:str="average"):
    model.eval()
    indexed_tokens = tokenizer.convert_tokens_to_ids(words)
    segments_ids = [0] * len(indexed_tokens)
    tokens_tensor = torch.LongTensor([indexed_tokens]).to(device)
    segments_tensors = torch.LongTensor([segments_ids]).to(device)
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    return encoded_layers

def read_parse_write(tokenizer, model, infile, outfile, mode):
    reader = Reader()
    insts = reader.read_conll(infile, -1, True)
    f = open(outfile, 'wb')
    all_vecs = []
    for inst in insts:
        vec = parse_sentence(tokenizer, model, inst.input.words, mode=mode)
        all_vecs.append(vec)
    pickle.dump(all_vecs, f)
    f.close()


def load_bert():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model.to(device)
    return tokenizer, model


device = torch.device('cuda:0')
tokenizer, bert_model = load_bert()
mode= "average"
dataset="conll2003"
dep = ""
file = "../data/"+dataset+"/train"+dep+".conllx"
outfile = file + ".bert."+mode+".vec"
read_parse_write(tokenizer, bert_model, file, outfile, mode)
file = "../data/"+dataset+"/dev"+dep+".conllx"
outfile = file + ".bert."+mode+".vec"
read_parse_write(tokenizer, bert_model, file, outfile, mode)
file = "../data/"+dataset+"/test"+dep+".conllx"
outfile = file + ".bert."+mode+".vec"
read_parse_write(tokenizer, bert_model, file, outfile, mode)