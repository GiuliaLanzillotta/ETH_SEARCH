'''Baseline gridsearch file for ETH-Search DSL'''

#Loading Libraries
import pandas as pd
import os
import numpy as np
import re
import random
import nltk
from nltk import word_tokenize, RegexpTokenizer,PunktSentenceTokenizer, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import sys
import tomotopy as tp

#Loading the data
abstracts = pd.read_csv("abstracts_eng.csv")

#Pre-processing
'''Tokenisation'''
tokenised = []
count = 0
for abstract in abs_list:
    raw = abstract
    tokens = gensim.utils.simple_preprocess(str(raw), deacc=True)
    tokenised.append(tokens)
    count += len(tokens)
print(str(count)+" tokens created")

'''Stop-words'''
stop_words = stopwords.words('english')
cleaned = [[word for word in doc if word not in stop_words] for doc in tokenised]

'''Stemming and Lemmatisation'''
word_stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
lemmatized = [[lemmatiser.lemmatize(word_stemmer.stem(word)) for word in doc] for doc in cleaned]

'''Building n-grams'''
# n-grams creation hyperparameters
# min_count (float, optional) – Ignore all words and bigrams with total collected count lower than this value.
b_min_c = 5
t_min_c = 5
# threshold (float, optional) – Represent a score threshold for forming the phrases (higher means fewer phrases)
b_thre = 50
t_thre = 5
'''Bigrams'''
bigram = gensim.models.Phrases(lemmatized, min_count=b_min_c, threshold=b_thre)
bigram_mod = gensim.models.phrases.Phraser(bigram)#
'''Trigrams'''
trigram = gensim.models.Phrases(bigram[lemmatized], min_count=t_min_c, threshold=t_thre)
trigram_mod = gensim.models.phrases.Phraser(trigram)
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

bigrammed = make_bigrams(lemmatized)
trigrammed = make_trigrams(bigrammed)

cleaned = trigrammed

#Grid Search
random.seed = 11
# split data in train, test and validation set
train_size = int(0.7*len(cleaned)) #70% for training
test_size = int(0.3*len(cleaned)) #30% for testing

random.shuffle(cleaned)

train_docs = cleaned[0:train_size]
test_docs = cleaned[train_size:]

# Defining the grid

ks = [50, 100, 150, 200, 300, 350, 450]
# alpha = [1/k, 10/k, 0.1/k, None]
# eta = [1/w, 10/w, 0.1/w, None]

#Defining perplexity

def compute_test_pp(ll, docs):
    """ pp = exp(-ll/ct)"""
    ct = sum([len(docs) for doc in docs])
    pp = np.exp(-1*ll/ct)
    return pp


# Grid search of best topic number (this needs to run for a while)
# We collect LL, perplexity and coherence scores, saving them in variables

import time

pps = []
best_models = []
# number of words in our vocabulary
c = []
for doc in train_docs:
    c += doc
w = len(set(c))

# define training parameters
model_burn_in = 250
train_updates = 1000
train_iter = 10

for k in ks:

    start = time.time()

    for alpha in [1 / k, 10 / k, 0.1 / k, None]:
        for eta in [1 / w, 10 / w, 0.1 / w, None]:
            print("K= " + str(k) + ", alpha = " + str(alpha) + ", eta=" + str(eta) + " -----------------------------")
            model, LLs, _ = train_LDA(train_docs, k,
                                      alpha=alpha,
                                      eta=eta,
                                      model_burn_in=model_burn_in,
                                      train_updates=train_updates,
                                      train_iter=train_iter)

            # LL
            ll = get_test_LL(test_docs, model)
            ## PP
            # TODO: obtain perplexity on the test set
            pp = compute_test_pp(ll, test_docs)
            pps += [pp]
            print("Test perplexity = " + str(pp))

            end = time.time()

            print("Time elapsed: " + str(round(end - start, 1)) + " s")