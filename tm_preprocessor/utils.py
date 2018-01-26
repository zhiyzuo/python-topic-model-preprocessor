import nltk
import numpy as np
from nltk.corpus import wordnet

def _to_unicode(str_, py3=True):
    if py3:
        return str_
    else:
        return unicode(str_)

def parse_pos_tag(tag):
    # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN

def _post_tag(corpus):
    corpus_w_tag = list()
    for doc in corpus:
        word_tag_tuple = nltk.pos_tag(doc)
        corpus_w_tag.append([[word, parse_pos_tag(tag)] for word, tag in word_tag_tuple])
    return corpus_w_tag
