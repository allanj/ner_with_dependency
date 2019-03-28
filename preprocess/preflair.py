from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, PooledFlairEmbeddings
import pickle
from config.reader import  Reader
import numpy as np
from flair.data import Sentence

def load_flair(mode = 'flair'):
    if mode == 'flair':
        stacked_embeddings = StackedEmbeddings([
            WordEmbeddings('glove'),
            PooledFlairEmbeddings('news-forward', pooling='min'),
            PooledFlairEmbeddings('news-backward', pooling='min')
        ])
    else:##bert
        stacked_embeddings = BertEmbeddings('bert-base-uncased')  ##concat last 4 layers give the best
    return stacked_embeddings

def embed_sent(embeder, sent):
    sent = Sentence(' '.join(sent))
    embeder.embed(sent)
    return sent


def read_parse_write(elmo, infile, outfile,):
    reader = Reader()
    insts = reader.read_conll(infile, -1, True)
    f = open(outfile, 'wb')
    all_vecs = []
    for inst in insts:
        sent = embed_sent(elmo, inst.input.words)
        # np.empty((len(sent)),dtype=np.float32)
        arr = []
        for token in sent:
            # print(token)
            # print(token.embedding)
            arr.append(np.expand_dims(token.embedding.numpy(), axis=0))
        # all_vecs.append(vec)
        all_vecs.append(np.concatenate(arr))
    pickle.dump(all_vecs, f)
    f.close()


mode = 'flair'
model = load_flair(mode=mode)
# mode= "average"
dataset="conll2003"
dep = ".sd"
file = "./data/"+dataset+"/train"+dep+".conllx"
outfile = file.replace(".sd", "") + "."+mode+".vec"
read_parse_write(model, file, outfile)
file = "./data/"+dataset+"/dev"+dep+".conllx"
outfile = file.replace(".sd", "") + "."+mode+".vec"
read_parse_write(model, file, outfile)
file = "./data/"+dataset+"/test"+dep+".conllx"
outfile = file.replace(".sd", "") + "."+mode+".vec"
read_parse_write(model, file, outfile)
