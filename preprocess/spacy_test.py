# 
# @author: Allan
#

import spacy
from spacy.pipeline import DependencyParser
from spacy.tokens import Doc
# nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_sm')
# print(type(nlp.vocab))
#
#
# doc = nlp(u"This is a sentence.")
# processed = parser.predict([doc])
# print(processed)
# doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers....")

doc = Doc(nlp.vocab, words=[u'hello', u'world', u'manufacturers.'],
                           spaces=[True, False, False])
# parser = DependencyParser(nlp.vocab)
# processed = parser(doc)
print("length of the sentence:", len(doc))
for tok in doc:
    print(tok, tok.head , tok.head.i)
# for chunk in doc.noun_chunks:
#     print(chunk.text, chunk.root.text, chunk.root.dep_,
#           chunk.root.head.text)