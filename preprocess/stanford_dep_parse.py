





import stanfordnlp

def init(lang:str):
    prefix = 'D:/Users/Allan/stanfordnlp_resources/'+lang+'_models/'
    config = {
        'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use, pos
        'lang': lang[:2], # Language code for the language to build the Pipeline in
        'tokenize_model_path': prefix + lang + '_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        # 'mwt_model_path': prefix + lang + '_mwt_expander.pt',
        'pos_model_path': prefix + lang + '_tagger.pt',
        'pos_pretrain_path': prefix + lang + '.pretrain.pt',
        'lemma_model_path': prefix + lang + '_lemmatizer.pt',
        'depparse_model_path': prefix + lang + '_parser.pt',
        'depparse_pretrain_path': prefix + lang + '.pretrain.pt',
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
                heads, deps = stanford_process(model, words)
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


lang = 'en_ewt' ##id: id_gsd, gl_ctg, en, af_afribooms
nlp = init(lang)
# process_txt(nlp, "data/indo/train.txt", "data/indo/train.conllx")
# process_txt(nlp, "data/indo/dev.txt", "data/indo/dev.conllx")
# process_txt(nlp, "data/indo/test.txt", "data/indo/test.conllx")


# process_txt(nlp, "data/galician/train.txt", "data/galician/train.conllx")
# process_txt(nlp, "data/galician/dev.txt", "data/galician/dev.conllx")
# process_txt(nlp, "data/galician/test.txt", "data/galician/test.conllx")

# process_txt(nlp, "data/af/train.txt", "data/af/train.sd.conllx")
# process_txt(nlp, "data/af/dev.txt", "data/af/dev.sd.conllx")
# process_txt(nlp, "data/af/test.txt", "data/af/test.sd.conllx")

process_conllx(nlp, "data/ontonotes/train.sd.conllx", "data/ontonotes/train.stud.conllx")
process_conllx(nlp, "data/ontonotes/dev.sd.conllx", "data/ontonotes/dev.stud.conllx")
process_conllx(nlp, "data/ontonotes/test.sd.conllx", "data/ontonotes/test.stud.conllx")
