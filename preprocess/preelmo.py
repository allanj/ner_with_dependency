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
mode= "last"
file = "./data/conll2003/train.sd.conllx"
outfile = file + ".elmo."+mode+".vec"
read_parse_write(elmo, file, outfile, mode)
file = "./data/conll2003/dev.sd.conllx"
outfile = file + ".elmo."+mode+".vec"
read_parse_write(elmo, file, outfile, mode)
file = "./data/conll2003/test.sd.conllx"
outfile = file + ".elmo."+mode+".vec"
read_parse_write(elmo, file, outfile, mode)

