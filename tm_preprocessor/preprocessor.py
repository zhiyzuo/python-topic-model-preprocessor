import warnings
import numpy as np
import pandas as pd
import pkg_resources
from nltk.stem import *
from gensim import corpora
from collections import defaultdict
from string import digits, maketrans

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
        remove_digits_punctuactions()
            Remove both digits and punctuations in the corpus.
        add_stopwords(additional_stopwords)
            Add additional stop words (`additional_stopwords`) to `self.stopwords`.
        tokenize(stemmer, min_freq, min_length)
            Tokenize the corpus into bag of words using the specified `stemmer` and 
            minimum frequency (`min_freq`) and length (`min_length`) of words/tokens.
        serialize(path, format_)
            Serialize corpus in `format_` and build vocabulary. Save dump files to `path`.
        get_word_ranking()
            Obtain frequency rankings of each word/token. Return `None` if none of the preprocessing has been done.

    """

    def __init__(self, documents, \
                 punctuations='''"!@#$%^*(),.:;&=+-_?\\\'`''', \
                 stopword_file=None):
        ''' Init function for `Preprocessor` class

            Parameters
            ----------
            corpus : iteratble object (list/tuple/numpy array...; init to None)
                Processed collection of documents.
            documents : iteratble object (list/tuple/numpy array...)
                A list of documents. This should not be altered throughout the processing
            punctuations : str
                String sequence of punctuations to be removed. By default: "!@#$%^*(),.:;&=+-_?'`\\
            stopword_file : str
                File path for stop word list. By default use pre-defined stopwords.
        '''

        self.corpus = None
        self.vocabulary = None
        self.documents = documents
        self.punctuations = punctuations
        if stopword_file is None:
            self.stopwords = np.loadtxt(pkg_resources.resource_stream('tm_preprocessor', \
                                                                      'data/stopwords.csv'), \
                                        dtype=str)
        else:
            self.stopwords = np.loadtxt(stopword_file, dtype=str)

    def remove_digits_punctuactions(self):
        ''' Remove digits and punctuations
        '''

        replace_punctuation = maketrans(self.punctuations, ' '*len(self.punctuations))
        replace_digits = maketrans(digits, ' '*len(digits))
        corpus = np.asarray([doc.encode('ascii', 'ignore').translate(replace_digits, digits) \
                                  for doc in self.documents])
        self.corpus = np.asarray([doc.translate(replace_punctuation, self.punctuations) \
                                  for doc in corpus])

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

    def tokenize(self, stemmer=PorterStemmer(), min_freq=1, min_length=1):
        '''
            Tokenize the corpus into bag of words

            Parameters
            ----------
            stemmer : nltk.stem stemmers
                Stemmer to use (Porter by default). See http://www.nltk.org/api/nltk.stem.html. If `None`, do not stem.
            min_freq : int
                The minimum frequency of a token to be kept
            min_length : int
                The minimum length of a token to be kept
        '''

        ## remove stop words, stem, and tokenize them (lower case)
        if stemmer is None:
            corpus = np.array([[word for word in doc.lower().split() \
                                if word not in self.stopwords] for doc in self.corpus])
        else:
            corpus = np.array([[stemmer.stem(word) for word in doc.lower().split() \
                                  if word not in self.stopwords] for doc in self.corpus])
        ## keep words occur more than once and more than one letter
        frequency = defaultdict(int)
        for doc in corpus:
            for token in doc:
                frequency[token] += 1

        self.corpus = np.asarray([[token for token in doc \
                                   if frequency[token] > min_freq and len(token) > 1] \
                                   for doc in corpus])

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
            exec("corpora.%s.serialize('%s/corpus_%s.dump', corpus_vector)"%(format_, path, format_))

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
        count_df = pd.DataFrame(frequency.items(), \
                                columns=['word', 'frequency'])
        return count_df.sort_values('frequency', ascending=False)

