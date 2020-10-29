'''Baseline gridsearch file for ETH-Search DSL'''
#Loading Libraries
import os
import sys
import numpy as np
import pandas as pd
import time
import random
import pickle

import nltk
from nltk import word_tokenize, RegexpTokenizer,PunktSentenceTokenizer, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import tomotopy as tp

## -------------------------------------
## Constants / hyperparameters  
## collecting here all the constant so that they're easier to spot and work on for us
## -------------------------------------
TRAIN_PERC = 0.7
MODEL_BURN_IN = 250
TRAIN_UPDATES = 1000
TRAIN_ITERS = 10
SAVE_TOP = 5

print("## -------------------------------------")
print("##\t GRID SEARCH ")
print("## -------------------------------------")

## -------------------------------------
## Parsing the command line arguments 
## -------------------------------------
import argparse 
# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-I","--input",help="Insert the input file name")
parser.add_argument("-L","--location",help="Insert the working directory")
parser.add_argument("-T","--type",help="Insert the type of model to grid search over. Supported values for now are: LDA, CTM, PA. Default is LDA.")
args  = parser.parse_args()

if not args.type: model_type="LDA"
else: model_type = args.type
if args.type not in ["LDA","CTM","PA"]: 
    print("Wrong usage. See help command for more info.")
    exit(1)



## -------------------------------------
## Loading the preprocessed file  
## -------------------------------------
standard_location = "/mnt/ds3lab-scratch/dslab2020/ethsearch/"
if args.location: location = args.location
else: location=standard_location
if args.input: input_file = args.input
else: input_file = "abstracts_processed.csv"
input_path = os.path.join(location,input_file)

print("Loading the file at "+input_path+" ...")
with open(input_path, "rb") as fp:   # Unpickling
    documents = pickle.load(fp)
print("Loading successful ")



## -------------------------------------
## Train/test split 
## -------------------------------------

# split data in train and test set
print("Splitting the data into train and testing ...")
random.seed = 11
train_size = int(TRAIN_PERC*len(documents)) 
test_size = int((1-TRAIN_PERC)*len(documents)) 
random.shuffle(documents)
train_docs = documents[0:train_size]
test_docs = documents[train_size:]



## -------------------------------------
## Helper functions 
## -------------------------------------

# Defining perplexity
def compute_test_pp(ll, docs):
    """ 
    Get test-perplexity as a function of test-loglikelihood 
    and number of tokens in the collection.
    --------------------------------
        pp = exp(-ll/ct)

    """
    ct = sum([len(docs) for doc in docs])
    pp = np.exp(-1*ll/ct)
    return pp

# Computing test log-likelihood
def get_test_LL(test_docs, model):
    
    # make a list of documents of type required by tp
    test_set = []
    for doc in test_docs:
        test_set.append(model.make_doc(doc))
    
    # return topic distribution and log-likelihood of new documents
    topic_dist, likelihood = model.infer(test_set)
    
    # use mean log-likelihood as performance measure
    return np.mean(likelihood)

# Compute number of distinct tokens in a collection
def get_num_tokens(collection):
    c = []
    for doc in collection:
        c += doc
    w = len(set(c))
    return w

## -------------------------------------
## LDA grid search
## -------------------------------------

def train_LDA(documents, k, min_cf=0, min_df=0, rm_top=0, alpha=0.1, eta=0.01, model_burn_in=100, 
              train_updates = 1000, train_iter = 10):
    # instantiate
    model = tp.LDAModel(tw=tp.TermWeight.ONE, min_df=min_df, min_cf=min_cf, rm_top=rm_top, k=k, alpha = alpha, 
                        eta = eta)
    # add documents to model
    for doc in documents: model.add_doc(doc)
    # training**
    model.burn_in = model_burn_in
    # initialising 
    model.train(iter=0)
    print('Num docs:', len(model.docs), ', Vocab size:', len(model.used_vocabs), ', Num words:', model.num_words)
    print('Removed top words:', model.removed_top_words)
    print('Training...')
    # actual training 
    for i in range(0, train_updates, train_iter):
        model.train(train_iter)
        if i%100==0:print('Iteration: {}'.format(i))

    return model

def LDA_search(train_docs, test_docs):
    """ Wrapper function for LDA grid search """

    print("Starting LDA grid search ...")

    # THE GRID
    ks = [50, 100, 150, 200, 300, 350, 450]

    pps = [] # perplexities 

    w = get_num_tokens(train_docs)

    start = time.time()

    for k in ks:
        startk = time.time()
        for alpha in [1 / k, 10 / k, 0.1 / k, None]:
            for eta in [1 / w, 10 / w, 0.1 / w, None]:

                print(" ------------------------------------------- ")
                print("| K = " + str(k) + "|\t alpha = " + str(alpha) + "\t| eta=" + str(eta))
                
                model = train_LDA(train_docs, k,
                                    alpha=alpha,
                                    eta=eta,
                                    model_burn_in=MODEL_BURN_IN,
                                    train_updates=TRAIN_UPDATES,
                                    train_iter=TRAIN_ITERS)

                # LL
                ll = get_test_LL(test_docs, model)
                ## PP
                pp = compute_test_pp(ll, test_docs)
                pps += [pp]
                print("Test perplexity = " + str(pp))
            
                

        endk = time.time()
        print("Time elapsed for k="+str(k)+" : " + str(round(endk - startk, 1)) + " s")

    end = time.time()
    print("Time elapsed: " + str(round(end - start, 1)) + " s")



## -------------------------------------
## CTM grid search
## -------------------------------------

def train_CTM(documents, k, min_cf=0, rm_top=0, smoothing_alpha=0.1, eta=0.01, model_burn_in=100, 
              train_updates = 1000, train_iter = 10):
    
    # instantiate
    model = tp.CTModel(tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top, k=k, smoothing_alpha = smoothing_alpha,
                      eta = eta)
    
    # add documents to model
    for doc in documents: model.add_doc(doc)
    
    # training**
    model.burn_in = model_burn_in
    # initialising 
    model.train(iter=0)
    print('Num docs:', len(model.docs), ', Vocab size:', len(model.used_vocabs), ', Num words:', model.num_words)
    print('Removed top words:', model.removed_top_words)
    print('Training...')
    # actual training 
    for i in range(0, train_updates, train_iter):
        model.train(train_iter)
        if i%100==0:print('Iteration: {}'.format(i))
    return model

def CTM_search(train_docs, test_docs):
    """ Wrapper function for CTM grid search """
    print("Starting CTM grid search ...")

    # perplexities
    pps = []

    w = get_num_tokens(train_docs)

    # The GRID
    ks = [50, 100, 150, 200, 300, 350, 450]
    etas = [1/w, 10/w, 0.1/w, None]
    
    start = time.time()

    for k in ks:
        startk = time.time()
        for eta in etas:
            print(" ------------------------------------------- ")
            print("| K = " + str(k) + "\t| eta=" + str(eta))
            model= train_CTM(train_docs, k,
                                eta=eta, 
                                model_burn_in=MODEL_BURN_IN,
                                train_updates = TRAIN_UPDATES, 
                                train_iter = TRAIN_ITERS)

            ll = get_test_LL(test_docs, model)
            pp = compute_test_pp(ll, test_docs)
            pps += [pp]
            print("Test perplexity = "+str(pp))

        endk = time.time()
        print("Time elapsed for k="+str(k)+" : " + str(round(endk - startk, 1)) + " s")

    end = time.time()
    print("Time elapsed: " + str(round(end - start, 1)) + " s")

## -------------------------------------
## Pachinko grid search
## -------------------------------------

def train_PA(documents, k1, k2, min_cf=0, rm_top=0, alpha=0.1, eta=0.01, model_burn_in=100, 
              train_updates = 1000, train_iter = 10):
    
    # instantiate
    model = tp.PAModel(tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top, k1=k1, k2=k2, alpha = alpha, eta = eta)
    
    # add documents to model
    for doc in documents: model.add_doc(doc)
    
    # training**
    model.burn_in = model_burn_in
    # initialising 
    model.train(iter=0)
    print('Num docs:', len(model.docs), ', Vocab size:', len(model.used_vocabs), ', Num words:', model.num_words)
    print('Removed top words:', model.removed_top_words)
    print('Training...')
    # actual training 
    for i in range(0, train_updates, train_iter):
        model.train(train_iter)
        if i%100==0:print('Iteration: {}'.format(i))
    return model

def PA_search(train_docs, test_docs):
    """ Wrapper function for Pachinko grid search """
    
    print("Starting Pachinko grid search ...")

    w = get_num_tokens(train_docs)

    # The GRID
    k2s = [50, 100, 150, 200, 300, 350, 450]
    etas = [1/w, 10/w, 0.1/w, None]

    # perplexities
    pps = []

    start = time.time()

    for k2 in k2s:
        startk2 = time.time()
        for k1 in [int(k2/5), int(k2/10), int(k2/20)]:
            startk1 = time.time()
            for alpha in [1/k1, 0.1/k1, 0.01/k1]:  
                
                for eta in etas:
                    print(" ------------------------------------------- ")
                    print("| K1= "+str(k1)+ "\t| K2= " + str(k2) + "\t| alpha = "+str(alpha)+"\t| eta="+str(eta))
                    model= train_PA(train_docs, k1 = k1, k2 = k2, 
                                        alpha=alpha,
                                        eta=eta, 
                                        model_burn_in=MODEL_BURN_IN,
                                        train_updates = TRAIN_UPDATES, 
                                        train_iter = TRAIN_ITERS)

    
                    ll = get_test_LL(test_docs, model)
                    pp = compute_test_pp(ll, test_docs)
                    pps += [pp]
                    print("Test perplexity = "+str(pp))

            endk1 = time.time()
            print("Time elapsed for k1="+str(k1)+" : " + str(round(endk1 - startk1, 1)) + " s")

        endk2 = time.time()
        print("Time elapsed for k2="+str(k2)+" : " + str(round(endk2 - startk2, 1)) + " s")

    end = time.time()
    print("Time elapsed: " + str(round(end - start, 1)) + " s")



# finally running the grid search

grid_search_fun = {
    "LDA": LDA_search,
    "CTM": CTM_search,
    "PA": PA_search
}

grid_search_fun[model_type](train_docs, test_docs)