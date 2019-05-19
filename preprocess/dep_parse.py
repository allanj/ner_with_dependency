




import spacy
from spacy.pipeline import DependencyParser
from spacy.tokens import Doc
import tqdm
import nltk
import benepar

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
                labels = []
                continue
            word, label = line.split()
            words.append(word)
            labels.append(label)
    fres.close()


def process_german(filename:str, out:str):
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
                labels = []
                continue
            word, _, label = line.split()
            words.append(word)
            labels.append(label)
    fres.close()

def process_dutch(filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        tags= []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                heads, deps = spacy_process(words, tags)
                idx = 1
                for w, tag, h, dep, label in zip(words, tags, heads, deps, labels):
                    if dep == "ROOT":
                        dep = "root"
                    fres.write("{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_\t{}\n".format(idx, w, h, dep, label))
                    idx += 1
                fres.write('\n')
                words = []
                tags= []
                labels = []
                continue
            word, tag, label = line.split()
            words.append(word)
            labels.append(label)
            tags.append(tag)
    fres.close()


def process_conllx(model, filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                heads, deps = spacy_process(model, words, tags)
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

def sa_parse(model, words):
    tree = model.parse(words)
    tree_str = str(tree).replace("\n", '')
    return tree_str

def parse_conllx(model, filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                tree_str = sa_parse(model, words)
                fres.write(tree_str + '\n')
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


def process_wnut(model, filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    line_idx = 1
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                heads, deps, tags = spacy_process(model, words)
                idx = 1
                for w, tag, h, dep, label in zip(words, tags, heads, deps, labels):
                    if dep == "ROOT":
                        dep = "root"
                    fres.write("{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_\t{}\n".format(idx, w, h, dep, label))
                    idx += 1
                fres.write('\n')
                words = []
                labels = []
                line_idx += 1
                continue
            #1	West	_	NNP	NNP	_	5	compound	_	_	B-MISC
            try:
                word, label = line.split()
            except:
                print(line_idx)
            line_idx += 1
            words.append(word)
            labels.append(label)
    fres.close()


def spacy_process(nlp, words, tags=None):
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
# process_conllx(nlp, "../data/ontonotes/train.sd.conllx", "../data/ontonotes/train.sud.conllx")
# process_conllx(nlp, "../data/ontonotes/dev.sd.conllx", "../data/ontonotes/dev.sud.conllx")
# process_conllx(nlp, "../data/ontonotes/test.sd.conllx", "../data/ontonotes/test.sud.conllx")

parser = benepar.Parser("benepar_en2_large")

parse_conllx(parser, "../data/conll2003/train.sd.conllx", "../data/conll2003/train.parse")
parse_conllx(parser, "../data/conll2003/dev.sd.conllx", "../data/conll2003/dev.parse")
parse_conllx(parser, "../data/conll2003/test.sd.conllx", "../data/conll2003/test.parse")

# nlp = spacy.load('es_core_news_md', disable=['ner'])
#
# process_conll2002("../data/conll2002/train.txt", "../data/conll2002/train.ud.conllx")
# process_conll2002("../data/conll2002/dev.txt", "../data/conll2002/dev.ud.conllx")
# process_conll2002("../data/conll2002/test.txt", "../data/conll2002/test.ud.conllx")
#
# nlp = spacy.load('nl_core_news_sm', disable=['tagger','ner'])
#
# process_dutch("../data/dutch/train.txt", "../data/dutch/train.ud.conllx")
# process_dutch("../data/dutch/dev.txt", "../data/dutch/dev.ud.conllx")
# process_dutch("../data/dutch/test.txt", "../data/dutch/test.ud.conllx")
#
#
# nlp = spacy.load('de_core_news_md', disable=['ner'])
#
# process_german("../data/german/train.txt", "../data/german/train.ud.conllx")
# process_german("../data/german/dev.txt", "../data/german/dev.ud.conllx")
# process_german("../data/german/test.txt", "../data/german/test.ud.conllx")


# model = spacy.load('en_core_web_lg', disable=['ner'])
# process_wnut(model, "../data/wnut17/wnut17train.conll", "../data/wnut17/train.ud.conllx")
# process_wnut(model, "../data/wnut17/emerging.dev.conll", "../data/wnut17/dev.ud.conllx")
# process_wnut(model, "../data/wnut17/emerging.test.annotated", "../data/wnut17/test.ud.conllx")
