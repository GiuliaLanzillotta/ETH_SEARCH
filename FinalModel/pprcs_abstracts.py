""" Preprocessing script for abstracts text""" 


#Loading Libraries
import os
import sys
import numpy as np
import pandas as pd
import time
import random
import pickle

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize, RegexpTokenizer,PunktSentenceTokenizer, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


## -------------------------------------
## Constants / hyperparameters  
## collecting here all the constant so that they're easier to spot and work on for us
## -------------------------------------

'''n-grams model hyperparameters'''
# min_count (float, optional) 
# – Ignore all words and bigrams with total collected count lower than this value.
BIGRAM_MIN_C = 5
TRIGRAM_MIN_C = 5
# threshold (float, optional) 
# – Represent a score threshold for forming the phrases (higher means fewer phrases)
BIGRAM_THRS = 50
TRIGRAM_THRS = 5



print("## -------------------------------------")
print("##\t ABSTRACTS PRE-PROCESSING ")
print("## -------------------------------------")

## -------------------------------------
## Parsing the command line arguments 
## -------------------------------------
import argparse 
# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-I","--input",help="Insert the input file name")
parser.add_argument("-L","--location",help="Insert the working directory")
parser.add_argument("-O","--output",help="Insert the output file name")
args  = parser.parse_args()


## -------------------------------------
## Loading the file 
## -------------------------------------
standard_location = "/mnt/ds3lab-scratch/dslab2020/ethsearch/"
if args.location: location = args.location
else: location=standard_location
if args.input: input_file = args.input
else: input_file = "abstracts_eng.csv"

input_path = os.path.join(location,input_file)
print("Reading the file at "+input_path+" ...")
abstracts = pd.read_csv(input_path)
abs_list = list(abstracts['abstract'])

## -------------------------------------
## Preprocessing 
## -------------------------------------
'''Tokenisation'''
print("Tokenising the text ...")
t1 = time.time()
tokenised = []
count = 0
for abstract in abs_list:
    raw = abstract
    tokens = gensim.utils.simple_preprocess(str(raw), deacc=True)
    tokenised.append(tokens)
    count += len(tokens)
print(str(count)+" tokens created")
t2 = time.time()
print("Time: " + str(round(t2 - t1, 1)) + " s")

'''Stop-words'''
print("Removing stopwords ...")
t1 = time.time()
stop_words = stopwords.words('english')
cleaned = [[word for word in doc if word not in stop_words] for doc in tokenised]
t2 = time.time()
print("Time: " + str(round(t2 - t1, 1)) + " s")

'''Stemming and Lemmatisation'''
print("Stemming and lemmatising ...")
t1 = time.time()
word_stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
lemmatized = [[lemmatiser.lemmatize(word_stemmer.stem(word)) for word in doc] for doc in cleaned]
t2 = time.time()
print("Time: " + str(round(t2 - t1, 1)) + " s")

'''Building n-grams'''
print("Modeling bigrams ...")
t1 = time.time()
bigram = gensim.models.Phrases(lemmatized, min_count=BIGRAM_MIN_C, threshold=BIGRAM_THRS)
bigram_mod = gensim.models.phrases.Phraser(bigram)
bigrammed = [bigram_mod[doc] for doc in lemmatized]
t2 = time.time()
print("Time: " + str(round(t2 - t1, 1)) + " s")

print("Modeling trigrams ...")
t1 = time.time()
trigram = gensim.models.Phrases(bigram[lemmatized], min_count=TRIGRAM_MIN_C, threshold=TRIGRAM_THRS)
trigram_mod = gensim.models.phrases.Phraser(trigram)
trigrammed = [trigram_mod[bigram_mod[doc]] for doc in bigrammed]
cleaned = trigrammed
t2 = time.time()
print("Time: " + str(round(t2 - t1, 1)) + " s")


## -------------------------------------
## Saving output 
## -------------------------------------
if args.output: output_file = args.output
else: output_file = "abstracts_processed.csv"

output_path = os.path.join(location,output_file)
print("Saving the file at "+output_file+" ...")

with open(output_path, "wb") as fp:   #Pickling
    pickle.dump(cleaned, fp)

print("Saving successful, exiting...")
