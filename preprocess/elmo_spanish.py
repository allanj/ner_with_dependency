from elmoformanylangs import Embedder
import pickle

"""
This file should be deprecated since every time result is different.

"""


def read_conll2002(filename:str):
    print(filename)
    sents = []
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                sents.append(words)
                words = []
                continue
            word, _ = line.split()
            words.append(word)
    return sents


def context_emb(emb, sents):
    return emb.sents2elmo(sents)


def read_parse_write(elmo, in_file, out_file):
    sents = read_conll2002(in_file)
    f = open(out_file, 'wb')
    batch_size = 1000
    all_vecs = []
    for idx in range(0, len(sents), batch_size):
        start = idx*batch_size
        end = (idx+1)*batch_size if (idx+1)*batch_size < len(sents) else len(sents)
        batch_sents = sents[idx*batch_size: (idx+1)*batch_size]
        embs = context_emb(elmo, batch_sents)
        for emb in embs:
            all_vecs.append(emb)
    pickle.dump(all_vecs, f)
    f.close()


elmo = Embedder('/Users/allanjie/Downloads/Spanish_ELMo/')
# /data/allan/embeddings/Spanish_ELMo/


# read_parse_write(elmo, "../data/conll2002/train.txt", "../data/conll2002/train.conllx.elmo.average.vec")
# read_parse_write(elmo, "../data/conll2002/dev.txt", "../data/conll2002/dev.conllx.elmo.average.vec")
# read_parse_write(elmo, "../data/conll2002/test.txt", "../data/conll2002/test.conllx.elmo.average.vec")

# sents = read_conll2002("../data/conll2002/train.txt")
sents = [['multinacional', 'española', 'Telefónica', 'Telefónica']]
result = context_emb(elmo, sents)
print("something")


# sents = read_conll2002("../data/conll2002/dev.txt")
# # sents = [['multinacional', 'española', 'Telefónica']]
# context_emb(emb, sents)
#
# sents = read_conll2002("../data/conll2002/test.txt")
# # sents = [['multinacional', 'española', 'Telefónica']]
# context_emb(emb, sents)
