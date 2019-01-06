# 
# @author: Allan
#

import spacy
from spacy.pipeline import DependencyParser
from spacy.tokens import Doc
nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])
# nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])
# print(type(nlp.vocab))
#
#
# doc = nlp(u"This is a sentence.")
# processed = parser.predict([doc])
# print(processed)
# doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers....")

doc = Doc(nlp.vocab, words=[u'he', u'is', u'a', u'man'],
                           spaces=[True, False, False, False])

doc[0].tag_ = "NN"
doc[1].tag_ = "VBD"
doc[2].tag_ = "DT"
doc[3].tag_ = "NNP"
# for tok in doc:
#     tok.tag_ = "NN"
# parser = DependencyParser(nlp.vocab)
for name, proc in nlp.pipeline:
    print("name", name)
    # iterate over components in order
    doc = proc(doc)

# processed = nlp(doc)
print("length of the sentence:", len(doc))
for tok in doc:
    print(tok, tok.tag_, tok.head , tok.head.i)
# for chunk in doc.noun_chunks:
#     print(chunk.text, chunk.root.text, chunk.root.dep_,
#           chunk.root.head.text)