import nltk
import numpy as np
import pandas as pd
from gensim import corpora
import warnings, sys, pkg_resources
from collections import defaultdict

from tm_preprocessor import utils

MAJOR_VERSION = sys.version_info[0]
if MAJOR_VERSION < 3:
    from string import digits, maketrans
    escape_encoding = 'string-escape'
else:
    from string import digits
    maketrans = str.maketrans
    escape_encoding = 'unicode_escape'

class Preprocessor(object):
    """
        Preprocessor of text corpus before feeding into topic modeling algorithms

        Attributes
        ----------
        corpus : np.array
            Processed corpus.
        documents : iteratble object (list/tuple/numpy array...)
            A list of documents
        punctuations : str
            String sequence of punctuations to be removed. By default: "!@#$%^*(),.:;&=+-_?'`\\
        stopwords : np.array
            An array of stopwords.
        vocabulary : corpora.Dictionary
            Dictionary of self.corpus

        Methods
        -------
        add_stopwords(additional_stopwords)
            Add additional stop words (`additional_stopwords`) to `self.stopwords`.
        tokenize(normalizer, min_freq, max_freq, min_length)
            Tokenize the corpus into bag of words using the specified `stemmer` or `lemmatizer` and
            minimum/maximum frequency (`min_freq` and `max_freq`) and length (`min_length`) of words/tokens.
        serialize(path, format_)
            Serialize corpus in `format_` and build vocabulary. Save dump files to `path`.
        get_word_ranking()
            Obtain frequency rankings of each word/token. Return `None` if none of the preprocessing has been done.

    """

    def __init__(self, documents, \
                 punctuations=r'''"!@#$%^*(),.:;&=+-_?\'`[]''', \
                 stopword_file=None):
        ''' Init function for `Preprocessor` class

            Parameters
            ----------
            corpus : iteratble object (list/tuple/numpy array...; init to None)
                Processed collection of documents.
            documents : iteratble object (list/tuple/numpy array...)
                A list of documents. This should not be altered throughout the processing
            punctuations : str
                String sequence of punctuations to be removed. By default: "[!@#$%^*(),.:;&=+-_?'`\\
            stopword_file : str
                File path for stop word list. By default use pre-defined stopwords.
        '''

        self.corpus = None
        self.vocabulary = None
        self.punctuations = punctuations

        ## escape backslahes
        self.documents = [doc.encode(escape_encoding).decode() for doc in documents]

        if stopword_file is None:
            self.stopwords = np.loadtxt(pkg_resources.resource_stream('tm_preprocessor', \
                                                                      'data/stopwords.csv'), \
                                        dtype=str)
        else:
            self.stopwords = np.loadtxt(stopword_file, dtype=str)

        #self.corpus = np.array([nltk.word_tokenize(doc) for doc in self.documents])
        #self.corpus = np.array([[token.lower() for token in doc] for doc in self.corpus])


    def add_stopwords(self, additional_stopwords):
        '''
            Add additional stopwords to the current `Preprocessor` object.

            Parameters
            ----------
            additional_stopwords : iteratble object (list/tuple/numpy array...; init to None)
                Additional stopwords.
        '''

        self.stopwords = np.insert(self.stopwords, 0, \
                                   additional_stopwords)

    def normalize(self, normalizer,
                  min_freq=.05, max_freq= .95, min_length=1):
        '''
            Normalize corpus by either lemmatization or stemming. Also remove rare/common and short words

            Parameters
            ----------
            stemmer : nltk.stem stemmers or lemmatizers
                Stemmer/lemmatizer to use. See http://www.nltk.org/api/nltk.stem.html. If `None`, do not stem.
            min_freq : float
                The minimum frequency (in ratio) of a token to be kept
            max_freq : float
                The maximum frequency (in ratio) of a token to be kept
            min_length : int
                The minimum length of a token to be kept
        '''
        ## remove stop words, stem, and tokenize them (lower case)
        if normalizer is None:
            self.corpus = np.array([[utils._to_unicode(word) for word in doc]\
                                     for doc in self.corpus])
        elif 'lemma' in str(type(normalizer)).lower():
            # if lemmatize, give the tags as well
            self.corpus = utils._post_tag(self.corpus)
            #print(self.corpus)
            self.corpus = [[utils._to_unicode(normalizer.lemmatize(word, tag)) for word, tag in doc]\
                            for doc in self.corpus]
        else:
            self.corpus = [[utils._to_unicode(normalizer.stem(word)) for word in doc]\
                            for doc in self.corpus]

        ## keep words occur more than once and more than one letter
        frequency_dict = defaultdict(int)
        total_count = 0
        for doc in self.corpus:
            for token in doc:
                frequency_dict[token] += 1
                total_count += 1

        self.corpus = [[token for token in doc \
                        if frequency_dict[token]/total_count >= min_freq and\
                           frequency_dict[token]/total_count <= max_freq and\
                           len(token) >= min_length] \
                        for doc in self.corpus]

    def serialize(self, path='.', format_='MmCorpus'):
        '''
            Serialize corpus and build vocabulary.

            Parameters
            ----------
            path : str
                The path to save corpus and vocabulary (current directory by default).
            format_ : str
                The format of the serialized corpus. See https://radimrehurek.com/gensim/tut1.html#corpus-formats
        '''

        if self.corpus is None:
            warnings.warn("Please at leaset select one of the preprocessing method first", \
                          UserWarning)

        else:
            ## build vocabulary
            self.dictionary = corpora.Dictionary(self.corpus)
            self.dictionary.save('%s/vocab.dict'%path)

            ## serialize corpus
            corpus_vector = [self.dictionary.doc2bow(doc) for doc in self.corpus]
            exec("corpora.%s.serialize('%s/corpus_%s.dump', \
                  corpus_vector, id2word=self.dictionary)"%(format_, path, format_))

    def get_word_ranking(self):
        '''
            Get the ranking of words (tokens). Note that this should be done a

            Returns
            -------
            pd.DataFrame
                Sorted dataframe with columns `word` and corresponding `frequency`.
        '''

        if self.corpus is None:
            warnings.warn("Please at leaset select one of the preprocessing method first", \
                          UserWarning)
            return

        frequency = defaultdict(int)
        for doc in self.corpus:
            for word in doc:
                frequency[word] += 1
        count_df = pd.DataFrame(list(frequency.items()), \
                                columns=['word', 'frequency'])
        return count_df.sort_values('frequency', ascending=False).reset_index(drop=True)

    def remove_digits_punctuactions(self):

        replace_punctuation = maketrans(self.punctuations, ' '*len(self.punctuations))
        replace_digits = maketrans(digits, ' '*len(digits))

        if MAJOR_VERSION < 3:
            self.corpus = [doc.translate(replace_punctuation, self.punctuations)\
                           for doc in self.documents]
            self.corpus = [doc.translate(replace_digits, digits).lower().split()\
                           for doc in self.corpus]
        else:
            self.corpus = [doc.translate(replace_punctuation)\
                           for doc in self.documents]
            self.corpus = [doc.translate(replace_digits).lower().split()\
                           for doc in self.corpus]

    def remove_stopwords(self):
        self.corpus = [[word for word in doc if word not in self.stopwords]\
                        for doc in self.corpus]
