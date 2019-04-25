from elmoformanylangs import Embedder
import pickle

"""
This file should be deprecated since every time result is different.

"""


def read_conllx(filename:str):
    print(filename)
    sents = []
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                if len(words) == 0:
                    print("len is 0")
                sents.append(words)
                words = []
                continue
            vals = line.split()
            words.append(vals[1])
    return sents


def context_emb(emb, sents):
    ## 0, word encoder:
    ##1 for the first LSTM hidden layer
    ## 2 for the second LSTM hidden lyaer
    ## -1 for an average of 3 layers (default)
    ## -2 for all 3 layers
    return emb.sents2elmo(sents, -1)


def read_parse_write(elmo, in_file, out_file):
    sents = read_conllx(in_file)
    print("number of sentences: {} in {}".format(len(sents), in_file))
    f = open(out_file, 'wb')
    batch_size = 1
    all_vecs = []
    for idx in range(0, len(sents), batch_size):
        start = idx*batch_size
        end = (idx+1)*batch_size if (idx+1)*batch_size < len(sents) else len(sents)
        batch_sents = sents[start: end]
        #print(batch_sents)
        embs = context_emb(elmo, batch_sents)
        for emb in embs:
            all_vecs.append(emb)
    pickle.dump(all_vecs, f)
    f.close()



elmo = Embedder('/data/allan/embeddings/Indonesian_ELMo', batch_size=1)
# /data/allan/embeddings/Spanish_ELMo/


read_parse_write(elmo, "data/indo/train.sd.conllx", "data/indo/train.conllx.elmo.vec")
read_parse_write(elmo, "data/indo/dev.sd.conllx", "data/indo/dev.conllx.elmo.vec")
read_parse_write(elmo, "data/indo/test.sd.conllx", "data/indo/test.conllx.elmo.vec")

# sents = read_conll2002("../data/conll2002/train.txt")
# sents = [['我', '在', '中国']]
# result = context_emb(elmo, sents)
# print("something")


# sents = read_conll2002("../data/conll2002/dev.txt")
# # sents = [['multinacional', 'española', 'Telefónica']]
# context_emb(emb, sents)
#
# sents = read_conll2002("../data/conll2002/test.txt")
# # sents = [['multinacional', 'española', 'Telefónica']]
# context_emb(emb, sents)
