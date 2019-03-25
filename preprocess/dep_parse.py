




import spacy
from spacy.pipeline import DependencyParser
from spacy.tokens import Doc
import tqdm


def process_conll2002(filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                heads, deps, tags = spacy_process(words)
                idx = 1
                for w, tag, h, dep, label in zip(words, tags, heads, deps, labels):
                    if dep == "ROOT":
                        dep = "root"
                    fres.write("{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_\t{}\n".format(idx, w, h, dep, label))
                    idx += 1
                fres.write('\n')
                words = []
                tags = []
                labels = []
                continue
            word, label = line.split()
            words.append(word)
            labels.append(label)
    fres.close()


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
                idx = 1
                for w, tag, h, dep, label in zip(words, tags, heads, deps, labels):
                    if dep == "ROOT":
                        dep = "root"
                    fres.write("{}\t{}\t_\t{}\t{}\t_\t{}\t{}\t_\t_\t{}\n".format(idx, w, tag, tag, h, dep, label))
                    idx += 1
                fres.write('\n')
                words = []
                tags = []
                labels = []
                continue
            #1	West	_	NNP	NNP	_	5	compound	_	_	B-MISC
            idx, word, _, pos , _, _, head, dep_label, _, _, label = line.split()
            words.append(word)
            tags.append(pos)
            labels.append(label)
    fres.close()


def spacy_process(words, tags=None):
    spaces = [False] * len(words)
    spaces[0] = True
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    if tags is not None:
        for i in range(len(words)):
            doc[i].tag_ = tags[i]
    for name, proc in nlp.pipeline:
        # iterate over components in order
        doc = proc(doc)
    heads = []
    deps = []
    pred_tags = []
    for tok in doc:
        if tok.i == tok.head.i:
            heads.append(0)
        else:
            heads.append(tok.head.i+1)
        deps.append(tok.dep_)
        pred_tags.append(tok.tag_)
    if tags is not None:
        return heads, deps
    else:
        return heads, deps, pred_tags

# nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])
#
# process_conll2003("../data/conll2003/train.sd.conllx", "../data/conll2003/train.sud.conllx")
# process_conll2003("../data/conll2003/dev.sd.conllx", "../data/conll2003/dev.sud.conllx")
# process_conll2003("../data/conll2003/test.sd.conllx", "../data/conll2003/test.sud.conllx")


nlp = spacy.load('es_core_news_md', disable=['ner'])

process_conll2002("../data/conll2002/train.txt", "../data/conll2002/train.ud.conllx")
process_conll2002("../data/conll2002/dev.txt", "../data/conll2002/dev.ud.conllx")
process_conll2002("../data/conll2002/test.txt", "../data/conll2002/test.ud.conllx")
