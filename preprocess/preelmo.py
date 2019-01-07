# 
# @author: Allan
#

from config.reader import  Reader
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import pickle


def parse_sentence(elmo, words, average:bool=True):
    vectors = elmo.embed_sentence(words)
    if average:
        return np.average(vectors, 0)
    else:
        return vectors


def load_elmo():
    return ElmoEmbedder(cuda_device=0)



def read_parse_write(elmo, infile, outfile):
    reader = Reader()
    insts = reader.read_conll(infile, -1, True)
    f = open(outfile, 'wb')
    all_vecs = []
    for inst in insts:
        vec = parse_sentence(elmo, inst.input.words)
        all_vecs.append(vec)
    pickle.dump(all_vecs, f)
    f.close()


elmo = load_elmo()
file = "../data/conll2003/train.conllx"
outfile = file + ".elmo.vec"
read_parse_write(elmo, file, outfile)
file = "../data/conll2003/dev.conllx"
outfile = file + ".elmo.vec"
read_parse_write(elmo, file, outfile)
file = "../data/conll2003/test.conllx"
outfile = file + ".elmo.vec"
read_parse_write(elmo, file, outfile)

