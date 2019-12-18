# 
# @author: Allan
#

from config.reader import  Reader
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import pickle


def parse_sentence(elmo, words, mode:str="average"):
    vectors = elmo.embed_sentence(words)
    if mode == "average":
        return np.average(vectors, 0)
    elif mode == 'weighted_average':
        return np.swapaxes(vectors, 0, 1)
    elif mode == 'last':
        return vectors[-1, :, :]
    elif mode == 'all':
        return vectors
    else:
        return vectors


def load_elmo():
    return ElmoEmbedder(cuda_device=0)



def read_parse_write(elmo, infile, outfile, mode):
    reader = Reader()
    insts = reader.read_conll(infile, -1, True)
    f = open(outfile, 'wb')
    all_vecs = []
    for inst in insts:
        vec = parse_sentence(elmo, inst.input.words, mode=mode)
        all_vecs.append(vec)
    pickle.dump(all_vecs, f)
    f.close()


elmo = load_elmo()
mode= "average"
dataset="ontonotes"
dep = ""
file = "../data/"+dataset+"/train"+dep+".conllx"
outfile = file + ".elmo."+mode+".vec"
read_parse_write(elmo, file, outfile, mode)
file = "../data/"+dataset+"/dev"+dep+".conllx"
outfile = file + ".elmo."+mode+".vec"
read_parse_write(elmo, file, outfile, mode)
file = "../data/"+dataset+"/test"+dep+".conllx"
outfile = file + ".elmo."+mode+".vec"
read_parse_write(elmo, file, outfile, mode)

