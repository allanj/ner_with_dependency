




import spacy
from spacy.pipeline import DependencyParser
from spacy.tokens import Doc
import tqdm




def process_conll2003(filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                heads, deps = spacy_process(words, tags)
                for w, tag, h, dep, label in zip(words, tags, heads, deps, labels):
                    fres.write("{}\t{}\t{}\t{}\t{}\n".format(w, tag, h, dep, label))
                fres.write('\n')
                words = []
                tags = []
                labels = []
                continue
            word, pos, label = line.split()
            words.append(word)
            tags.append(pos)
            labels.append(label)
    fres.close()


def spacy_process(words, tags):
    spaces = [False] * len(words)
    spaces[0] = True
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    for i in range(len(words)):
        doc[i].tag_ = tags[i]
    for name, proc in nlp.pipeline:
        # iterate over components in order
        doc = proc(doc)
    heads = []
    deps = []
    for tok in doc:
        if tok.i == tok.head.i:
            heads.append(-1)
        else:
            heads.append(tok.head.i)
        deps.append(tok.dep_)
    return heads, deps

nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])

process_conll2003("../data/conll2003/train.txt", "../data/conll2003/train.conllx")
process_conll2003("../data/conll2003/dev.txt", "../data/conll2003/dev.conllx")
process_conll2003("../data/conll2003/test.txt", "../data/conll2003/test.conllx")
