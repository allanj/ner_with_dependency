




import spacy
from spacy.pipeline import DependencyParser
from spacy.tokens import Doc
import tqdm


import stanfordnlp

def init():
    prefix = 'D:/Users/Allan/stanfordnlp_resources/id_gsd_models/'
    config = {
        'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use
        'lang': 'id', # Language code for the language to build the Pipeline in
        'tokenize_model_path': prefix + 'id_gsd_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        # 'mwt_model_path': prefix + 'id_gsd_mwt_expander.pt',
        'pos_model_path': prefix + 'id_gsd_tagger.pt',
        'pos_pretrain_path': prefix + 'id_gsd.pretrain.pt',
        'lemma_model_path': prefix + 'id_gsd_lemmatizer.pt',
        'depparse_model_path': prefix + 'id_gsd_parser.pt',
        'depparse_pretrain_path': prefix + 'id_gsd.pretrain.pt',
        'use_gpu': False,
        'tokenize_pretokenized': True
    }
    nlp = stanfordnlp.Pipeline(**config) # Initialize the pipeline using a configuration dict
    return nlp
    # doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie.") # Run the pipeline on input text
    # doc.sentences[0].print_tokens() # Look at the result


def stanford_process(model, words):
    doc = model(' '.join(words))
    heads = [word.governor for word in doc.sentences[0].words]
    rels = [word.dependency_relation for word in doc.sentences[0].words]

    return heads, rels

def process_txt(model, filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    line_idx = 1
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                heads, deps = stanford_process(model, words)
                idx = 1
                for w, h, dep, label in zip(words, heads, deps, labels):
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



nlp = init()
process_txt(nlp, "data/indo/train.txt", "data/indo/train.conllx")
process_txt(nlp, "data/indo/dev.txt", "data/indo/dev.conllx")
process_txt(nlp, "data/indo/test.txt", "data/indo/test.conllx")
# model = spacy.load('en_core_web_lg', disable=['ner'])
# process_wnut(model, "../data/wnut17/wnut17train.conll", "../data/wnut17/train.ud.conllx")
# process_wnut(model, "../data/wnut17/emerging.dev.conll", "../data/wnut17/dev.ud.conllx")
# process_wnut(model, "../data/wnut17/emerging.test.annotated", "../data/wnut17/test.ud.conllx")
