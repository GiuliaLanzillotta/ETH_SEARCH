""" Main processing module for ETM model """
#Loading Libraries
import warnings
warnings.filterwarnings("ignore")
import time 
import argparse
import pickle 
import numpy as np 
import nltk 
nltk.download('punkt')
import string
import os 
import math 
import random 
import sys
import pandas as pd
import scipy.io
from pathlib import Path
import torch
from torch import nn, optim
from torch.nn import functional as F

from etm import ETM
import utils
from transformers import DistilBertTokenizerFast, DistilBertModel

## -------------------------------------
## Constants / hyperparameters  
## collecting here all the constant so that they're easier to spot and work on for us
## -------------------------------------

SEED = 11 
BATCH_SIZE = 3000
SUBSET_SIZE = 100

### model-related arguments 

t_hidden_size = 800 # dimension of hidden space of q(theta)
theta_act = 'relu' # either tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu
train_embeddings = False

### optimization-related arguments
lr = 0.05
lr_factor = 5.0 #divide learning rate by this
epochs = 100 
enc_drop = 0.0 # dropout rate on encoder
clip = 0.0 # gradient clipping
nonmono = 10 # number of bad hits allowed ...?
weight_decay = 1.2e-6
anneal_lr = False # whether to anneal the learning rate or not
bow_norm = True # normalize the bows or not 
_optimizer = "adam"

### evaluation, visualization, and logging-related arguments
num_words = 10 #number of words for topic viz'
log_interval = 2 #when to log training
visualize_every = 10 #when to visualize results
tc = False # whether to compute topic coherence or not
td = False # whether to compute topic diversity or not

### data and file related arguments
training_docs = 0.8
testing_docs = 0.2
training_batch_size = 0.2 # 20% of the training set --> 5 batches training
eval_batch_size = 1.0  #input batch size for evaluation
load_from = "" #the name of the ckpt to run evaluation from


### final configurations 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setting the random seed 
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():torch.cuda.manual_seed(SEED)


print("## -------------------------------------")
print("##\t ETM TOPIC MODEL ")
print("## -------------------------------------")


## -------------------------------------
## Parsing the command line arguments 
## -------------------------------------
import argparse 
# Initiate the parser
parser = argparse.ArgumentParser()
# TODO: modify the standard location 
parser.add_argument("-I","--input",help="Insert the input file name", default="abstracts_eng.csv")
parser.add_argument("-L","--location",help="Insert the working directory", default="/cluster/home/glanzillo/etm/")
parser.add_argument("-B","--batch",default=0, type=int, help="Insert number of the batch to work on") # each job will work only on one batch
parser.add_argument("-V","--vocab",
    help="Insert the name of the file where to save the vocabulary", 
    default="vocab_etm") 
parser.add_argument("-EF","--embedding_file",
    help="Insert the name of the file where to save the embedding",       
    default="embedding_etm") 
parser.add_argument("-E","--embedding",
    help="Insert the name of the embedding to use",       
    default="glove") 
parser.add_argument("-T","--tokens",
    help="Insert the name of the file where to save the tokens produces", 
    default="new_tokens_etm")
parser.add_argument("-K","--topics", type=int, 
    help="Insert the number of topics per batch.", 
    default=25)
# note: the results directory will be created in the location 
# specified by the argument "location"
parser.add_argument("-R","--results",
    help="Insert the name of the directory where to save the results produced", 
    default="./results_etm/") 
parser.add_argument("-F","--from_file",
    help="Insert whether to load the processed input from file or run the processing.", 
    type= bool, 
    default=False) 
args  = parser.parse_args()


# creating the results directory and initialising important paths
Path(os.path.join(args.location,args.results)).mkdir(parents=True, exist_ok=True)
location = args.location
input_file = args.input
input_path = os.path.join(location,input_file)
results_path = os.path.join(location,args.results)
num_topics = args.topics
# global flags
global BERT
global GLOVE 
BERT = args.embedding.lower() == "bert"
GLOVE = args.embedding.lower() == "glove"
if BERT: 
    rho_size = 768 # dimension of rho 
    emb_size = 768 # dimension of embeddings 
if GLOVE: 
    rho_size = 300 
    emb_size = 300 
ckpt = os.path.join(results_path, 'etm_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
         num_topics, t_hidden_size, _optimizer, clip, theta_act, lr, training_batch_size, rho_size, train_embeddings))


## -------------------------------------
## Loading the preprocessed file  
## -------------------------------------
print("Loading the file at "+input_path+" ...")
abstracts = pd.read_csv(input_path) #Replace with latest version
collection = list(abstracts['abstract'])
print("Loading successful ")


## -------------------------------------
## Batch split 
## -------------------------------------

random.seed(SEED)
random.shuffle(collection)
start = BATCH_SIZE*args.batch
batch = collection[start:start+BATCH_SIZE]


## -------------------------------------
## Data preprocessing. 
## Computes: embedding matrix, vocabulary, tokenization
## -------------------------------------

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
# note: the words in this list are only lower 
#   case but distilbert tokenizer incorporates 
#   lower casing so we should be fine! read more 
#   here: https://huggingface.co/transformers/_modules/transformers/tokenization_distilbert_fast.html
stop_words = stopwords.words('english') 
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
def get_custom_stopwords():
    custom_stops = []
    with open("stops.txt", 'r', encoding="utf-8") as f:
        for line in enumerate(f):
            word = line[1].split()[0]
            custom_stops.append(word)
    return custom_stops
custom_stops = get_custom_stopwords()

# First defining utility functions 
# ---------------
def process_subset_cosine(doc_subset, tokenizer, model, set_of_embeddings, 
                          idx2word, new_token_ids, threshold=0.9, only_nouns=True):
    """ 
    Processing of a subset of the batch using cosine similarity clustering. 
    
    Parameters 
    ----- 
    doc_subset: list of documents (aka list of strings)
    tokenizer: instance of Bert tokenizer 
    model: instance of Bert model 
    set_of_embeddings: set containing the embedding vectors already in the vocabulary 
    idx2word: vocabulary mapping our token ids to the corresponding word. 
            Notice that each word can be mapped to multiple token ids.
    new_token_ids: representation of the collection with our token ids. 
    threshold: cosine similarity threshold. 
            Vectors with cosine similarity above the threshold are considered equal. 
            
    Returns 
    -----
    Updated versions of set_of_embeddings, idx2word and new_token_ids
    
    """
    
    tokenised_collection = tokenizer(doc_subset, return_tensors="pt", padding=True)
    model.resize_token_embeddings(len(tokenizer))
    print("tokenisation done")
    try: 
        with torch.no_grad(): 
            embedded_collection = model(**tokenised_collection)
    except Exception as e: 
        print("There was a problem with this subset, we'll skip it!")
        print(e)
        return set_of_embeddings, idx2word, new_token_ids
    embedded_collection.requires_grad = False
    # extract lower layers hidden states
    lower_hiddens = torch.sum(torch.stack(embedded_collection[1][0:4], dim=0), dim=0)
    #lower_hiddens = embedded_collection[1][6].cpu()

    print("embeddings done")
    
    
    ##  preparing the variables we need ---------
    if len(idx2word) == 0:idx = 0
    else:idx = len(set_of_embeddings)
        
    cos = torch.nn.CosineSimilarity(dim = 0)
    
    subset_size = len(doc_subset)
    start = time.time()
    
    ## processing the collection document by document ----------
    for i in range(subset_size):
        t1 = time.time()
        embedded_doc = lower_hiddens[i][tokenised_collection["attention_mask"][i].bool()] # removing padding using the attention mask 
        tokens_ids = tokenised_collection["input_ids"][i]
        new_token_ids_doc = []
        
        
        for j,emb_vector in enumerate(embedded_doc):
            
            token_id = tokens_ids[j].cpu().numpy() # bert current token 
            word = tokenizer.convert_ids_to_tokens([token_id])[0] # corresponding word
            # jump to the next token if the word is a stopword 
            if only_nouns and word not in nouns: continue
            if word in stop_words or word.startswith("##") or word in string.punctuation: continue 
            if word not in idx2word.values(): # we add the embedding anyway if we haven't encountered that word previously 
                # add new embedding to the set 
                set_of_embeddings.add(emb_vector)
                # increase the index and save the word in the dictionary 
                idx2word[idx] = word # save it in our vocabulary
                new_token_ids_doc += [idx]
                idx += 1
            else: # find the right id for the word and add it to our new tokenisation
                word_occurrences = [position for position, v in enumerate(list(idx2word.values())) if v == word]
                word_embeddings = [list(set_of_embeddings)[occ] for occ in word_occurrences]
                bool_list = [cos(emb_vector, other) >= threshold for other in word_embeddings] 
                if not any(bool_list): 
                    # add new embedding to the set 
                    set_of_embeddings.add(emb_vector)
                    # increase the index and save the word in the dictionary 
                    idx2word[idx] = word # save it in our vocabulary
                    new_token_ids_doc += [idx]
                    idx += 1
                else: 
                    word_id = list(idx2word.values()).index(word)
                    new_token_ids_doc += [word_id]
                
        new_token_ids += [new_token_ids_doc]
        t2 = time.time()
        try: 
            if i%(subset_size//3)==0:
                print("Document "+str(i)+" done. Time: "+str(round(t2-t1,2))+" s.")
        except Exception as _: pass # case subset_size//3 = 0 we get a division by 0 (subset_size must be < 3)
            
    end = time.time()
    print("Total time for the subset: "+str(round(end-start,2))+" s.")
    
    return set_of_embeddings, idx2word, new_token_ids

def process_batch(batch, subset_size, tokenizer, model):
    """ Processing of a batch of documents in the collection. """
    
    ## initialisation 
    idx2word = {} 
    set_of_embeddings = set()
    idx = 0 
    new_token_ids = []
    subset_size = 25 
    
    
    start = time.time()

    ## processing the batch one subset at a time
    for s in range(0,BATCH_SIZE,subset_size):
        print("Processing subset "+str(s//subset_size + 1))
        if s+subset_size < len(batch):batch_subset = batch[s:s+subset_size]
        else: batch_subset = batch[s:]
        set_of_embeddings, idx2word, new_token_ids = process_subset_cosine(batch_subset, tokenizer, model, 
                                                                           set_of_embeddings, idx2word, 
                                                                           new_token_ids, threshold=0.9)
        print("Number of word vectors so far: "+str(len(idx2word)))    
        print()
        
    end = time.time()
    print("Total time: "+str(round(end-start,2))+" s.")
    
    return set_of_embeddings, idx2word, new_token_ids

from nltk.tokenize import word_tokenize 

def tokenise_batch(batch, idx2word):
    """ 
        Tokenisation of a batch of documents in the collection given a pre-computed vocabulary.
    """
    
    ## initialisation 
    batch_size = len(batch)
    total_words = 0
    matched_words = 0
    new_token_ids = []
    
    start = time.time()
    
    for i,doc in enumerate(batch): 
        new_token_ids_doc = []
        
        for word in word_tokenize(doc.lower()): 
            if word in stop_words or word in custom_stops: continue # ignore the word
            total_words +=1
            
            if word in idx2word.values():
                matched_words+=1
                word_id = list(idx2word.values()).index(word)
                new_token_ids_doc += [word_id]
        
        new_token_ids += [new_token_ids_doc]

        if i%(batch_size//3)==0:
            print(str(i+1)+ " documents tokenised.")
            tmp = time.time()
            print("Time so far: "+str(round(tmp-start,2))+" s.")
        
    end = time.time()
    
    print("Total time: "+str(round(end-start,2))+" s.")
    print("Proportion of matched words: "+str(round(matched_words/total_words,2)))
    
    return new_token_ids

def get_batch(corpus, ind, vocab_size, device):
    """
    This function takes as input a list of tokenised documents (corpus)
    and the indices of the documents in the batch (ind)
    and returns as output the torch tensor to feed into the net. 
    The list of documents defines the batch to work on. 
    """
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    
    for i, doc_id in enumerate(ind):
        doc = corpus[doc_id]
        L = len(doc)
        if doc_id != -1:
            for word_id in doc:
                counts = doc.count(word_id)
                data_batch[i, word_id] = counts
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch

def do_processing(tokenizer, model, load_from_file):
    if load_from_file: 
        # Loading from binary 
        print("Loading from binary the processed input.")
        with open(os.path.join(results_path, args.vocab), "rb") as fp: idx2word = pickle.load(fp)
        with open(os.path.join(results_path, args.embedding), "rb") as fp: embedding = pickle.load(fp)
        with open(os.path.join(results_path, args.tokens), "rb") as fp: new_token_ids = pickle.load(fp)
    else: 
        set_of_embeddings, idx2word, new_token_ids = process_batch(batch, SUBSET_SIZE, tokenizer, model)
        embedding = torch.stack(list(set_of_embeddings))
        print("Saving to binary the results of the input processing.")
        # saving to binary intermediate result 
        with open(os.path.join(results_path, args.vocab), "wb") as fp: pickle.dump(idx2word, fp)
        with open(os.path.join(results_path, args.embedding), "wb") as fp: pickle.dump(embedding, fp)
        with open(os.path.join(results_path, args.tokens), "wb") as fp: pickle.dump(new_token_ids, fp)

    return embedding, idx2word, new_token_ids


print("## -------------------------------------")
print("##\t DATA PROCESSING ")
print("## -------------------------------------")

if BERT: 
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True, output_hidden_states=True)
    bert.eval()
    bert.to(device)
    embedding, idx2word, new_token_ids = do_processing(tokenizer, bert, args.from_file)

if GLOVE: 
    print("Loading glove vocabulary and embedding.")
    with open(os.path.join(results_path, args.vocab), "rb") as fp: idx2word = pickle.load(fp)
    with open(os.path.join(results_path, args.embedding_file), "rb") as fp: embedding = pickle.load(fp)
    if args.from_file:  
        with open(os.path.join(results_path, args.tokens), "rb") as fp: new_token_ids = pickle.load(fp)
    else: 
        print("Tokenising the batch.")
        new_token_ids = tokenise_batch(batch, idx2word)
        print("Saving to binary the results of the input processing.")
        with open(os.path.join(results_path, args.tokens), "wb") as fp: pickle.dump(new_token_ids, fp)


vocab_size = len(idx2word)


## -------------------------------------
## A few utility functions for the training/validation/visualization
## ------------------------------------

def get_optimizer(name, model):
    "Just a switch"
    if name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr_factor)
    elif name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=lr_factor)
    elif name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=lr_factor)
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=lr_factor)
    elif name == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=lr_factor)
    else:
        print('Defaulting to vanilla SGD')
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer

def train_test_split(collection):
    """ 
        Train-test splitting of our newly tokenised collection. 
        Note: no need to shuffle since the data has already been shuffled 
        during preprocessing.

        The collection is a list of list of tokens. 
    """
    num_docs_train = int(training_docs*BATCH_SIZE)
    train_corpus = collection[:num_docs_train]
    test_corpus = collection[num_docs_train:]
    return num_docs_train, train_corpus, test_corpus

num_docs_train, train_corpus, test_corpus = train_test_split(new_token_ids)
num_docs_test = len(new_token_ids) - num_docs_train
# going from relative to absolute length
_training_batch_size = int(len(train_corpus)*training_batch_size)
_eval_batch_size = int(len(test_corpus)*eval_batch_size) 

def train(model, epoch, corpus, num_docs_train=num_docs_train, 
            batch_size=_training_batch_size, vocab_size=vocab_size, 
            bow_norm=bow_norm, clip=clip, log_interval=log_interval):
    """ Just the training function ... """
    
    model.train() #setting the model in training mode
    # preparing all the data structures 
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0
    indices = torch.randperm(num_docs_train)
    indices = torch.split(indices, batch_size)
    
    for idx, ind in enumerate(indices): # all our batches 
        optimizer.zero_grad()
        data_batch = get_batch(corpus, ind, vocab_size, device)
        sums = data_batch.sum(1).unsqueeze(1) # what are we summing ?? 
        
        # maybe normalising the input 
        if bow_norm: normalized_data_batch = data_batch / sums
        else: normalized_data_batch = data_batch
        # loss on the batch 
        recon_loss, kld_theta = model(data_batch, normalized_data_batch)
        total_loss = recon_loss + kld_theta
        total_loss.backward(retain_graph=True) # compute backpropagation
        # maybe clip the gradient 
        if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # finally update the weights 
        # accumulate the total loss 
        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1
        
        # visualisation/print time! ('cur' stands for current ...)
        if idx % log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_real_loss = round(cur_loss + cur_kl_theta, 2)
            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    
    # Wrapping up the results of the epoch! 
    cur_loss = round(acc_loss / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('-'*50)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

def visualize(model, num_topics=num_topics, num_words=num_words, 
                vocab=idx2word, show_emb=True, **kwargs):
    """ 
    This is a cool visualisation function. 
    Takes as input the model so far and shows the discovered embeddings! 
    """
    emb_model = kwargs.get("emb_model")
    tokenizer = kwargs.get("tokenizer")
    model.eval() #set the net in evaluation mode 
    # set a few words to query 
    queries = ['insurance', 'weather', 'particles', 'religion', 'man', 'love', 
                'intelligence', 'money', 'politics', 'health', 'people', 'family']

    ## visualize topics using monte carlo (sampling from the posterior I guess)
    with torch.no_grad(): # no gradients computation - makes forward pass lighter 
        print('-'*20)
        print('Visualize topics...')
        topics_words = []
        gammas = model.get_beta() # topics distributions 
        for k in range(num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-num_words+1:][::-1])
            topic_words = [vocab[a] for a in top_words] 
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))

        if show_emb:
            ## visualize word embeddings by using V to get nearest neighbors
            print('-'*20)
            print('Visualize word embeddings by using output embedding matrix')
            
            # extract the embeddings from the model! 
            try:embeddings = model.rho.weight  # Vocab_size x E
            except:embeddings = model.rho         # Vocab_size x E
            
            
            for word in queries:
                if BERT:
                    # extracting Bert representation of the word
                    inputs = tokenizer(word, return_tensors="pt")
                    outputs = emb_model(**inputs).last_hidden_state[0]
                    outputs.requires_grad = False
                    if outputs.size()[0]>1: #aggregate
                        outputs = torch.sum(outputs, dim=0)
                    query=outputs
                if GLOVE:
                    word_id = list(vocab.values()).index(word)
                    query = embeddings[word_id]
                nns = utils.nearest_neighbors(q=query, 
                         embeddings=embeddings, vocab=list(vocab.values()))
                print('word: {} .. neighbors: {}'.format(word, nns)) # utility function 

def evaluate(model, train_corpus, test_coprus, vocab=idx2word,
                num_docs_test=num_docs_test, tc=tc, td=td, 
                eval_batch_size=_eval_batch_size, 
                vocab_size=vocab_size, 
                bow_norm=bow_norm):
    """
    Evaluating the trained model on the test set using either perplexity, 
    or coherence and diversity. 
    Compute perplexity on document completion.
    """
    
    model.eval() # set model in evaluation mode 
    with torch.no_grad():
        indices = torch.split(torch.tensor(range(num_docs_test)), eval_batch_size)
        
        ## get \beta here
        beta = model.get_beta()

        ### do dc and tc here
        acc_loss = 0
        cnt = 0
        
        for idx, ind in enumerate(indices):
            data_batch = get_batch(test_corpus, ind, vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            if bow_norm: normalized_data_batch = data_batch / sums
            else: normalized_data_batch = data_batch
                
            ## get theta
            theta, _ = model.get_theta(normalized_data_batch)
            ## get prediction loss
            res = torch.mm(theta, beta)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch).sum(1)
            loss = recon_loss / sums.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        
        # Calculate final loss 
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('Eval Doc Completion PPL: {}'.format(ppl_dc))
        
        
        if tc or td: # calculate topic coherence or topic diversity 
            beta = beta.data.cpu().numpy()
            if tc:
                print('Computing topic coherence...')
                utils.get_topic_coherence(beta, train_corpus, vocab)
            if td:
                print('Computing topic diversity...')
                utils.get_topic_diversity(beta, 25)
        return ppl_dc

## -------------------------------------
## Finally training 
## ------------------------------------

print("## -------------------------------------")
print("##\t TRAINING THE MODEL ")
print("## -------------------------------------")

# define model
etm_model = ETM(num_topics = num_topics, 
            vocab_size = vocab_size, 
            t_hidden_size = t_hidden_size, 
            rho_size = rho_size, 
            emsize = emb_size, 
            theta_act = theta_act, 
            embeddings = embedding,
            train_embeddings = train_embeddings, 
            enc_drop = enc_drop).to(device)

print('model: {}'.format(etm_model))

optimizer = get_optimizer(name=_optimizer, model=etm_model)

# Initialising the data structures 
best_epoch = 0 
best_val_ppl = 1e9
all_val_ppls = []

# Let's get a sense of how bad the model is before training 
print('\n')
print('Visualizing model quality before training...')
visualize(etm_model)
print('\n')

for epoch in range(1, epochs):
    
    train(etm_model, epoch, train_corpus) # train 
    num_docs_test = len(new_token_ids) - num_docs_train
    val_ppl = evaluate(etm_model, test_corpus, num_docs_test) # evaluate 
    
    # only saving the model if it's the best so far 
    if val_ppl < best_val_ppl: 
        with open(ckpt, 'wb') as f:
            torch.save(etm_model, f)
        best_epoch = epoch 
        best_val_ppl = val_ppl
        
    else:
        ## check whether to anneal lr (aka decreasing it by a constant factor )
        lr = optimizer.param_groups[0]['lr']
        if anneal_lr and (len(all_val_ppls) > nonmono and val_ppl > min(all_val_ppls[:-nonmono]) and lr > 1e-5):
            optimizer.param_groups[0]['lr'] /= lr_factor
            
    # maybe visualise 
    if epoch % visualize_every == 0:
        if GLOVE: visualize(etm_model)
        if BERT: visualize(etm_model, tokenizer = tokenizer, emb_model=bert)

        
    #save perplexities 
    all_val_ppls.append(val_ppl)


    print("Training finished.")


print("## -------------------------------------")
print("##\t TESTING THE MODEL ")
print("## -------------------------------------")

# load trained model and evaluate it  
with open(ckpt, 'rb') as f:
    etm_model = torch.load(f)
etm_model = etm_model.to(device)
etm_model.eval()

with torch.no_grad():
    ## ---------------
    ## Idea : get document completion perplexities
    test_ppl = evaluate(etm_model, test_corpus, num_docs_test)

    ## ----------------
    ## Idea : get most used topics
    indices = torch.tensor(range(num_docs_test)) # training documents indices 
    indices = torch.split(indices, _training_batch_size)
    #just initialising data structures 
    thetaAvg = torch.zeros(1, num_topics).to(device)
    thetaWeightedAvg = torch.zeros(1, num_topics).to(device)
    cnt = 0
    for idx, ind in enumerate(indices):
        data_batch = get_batch(test_corpus,ind, vocab_size, device) # TODO: fix here 
        sums = data_batch.sum(1).unsqueeze(1) 
        cnt += sums.sum(0).squeeze().cpu().numpy()
        # maybe normalise 
        if bow_norm:normalized_data_batch = data_batch / sums
        else: normalized_data_batch = data_batch
        # get the theta 
        theta, _ = etm_model.get_theta(normalized_data_batch)
        thetaAvg += theta.sum(0).unsqueeze(0) /num_docs_train
        weighed_theta = sums * theta
        thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
        # let's print the progress as we go 
        if idx % 100 == 0 and idx > 0:
            print('batch: {}/{}'.format(idx, len(indices)))
    # finally the results are in 
    thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
    print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

    # Now we show the topics
    # A nice visualisation is always welcome 
    beta = etm_model.get_beta()
    topic_indices = list(np.random.choice(num_topics, 10)) # 10 random topics
    print('\n')
    for k in range(num_topics):#topic_indices:
        gamma = beta[k]
        top_words = list(gamma.cpu().numpy().argsort()[-num_words+1:][::-1])
        topic_words = [idx2word[a] for a in top_words]
        print('Topic {}: {}'.format(k, topic_words))

    # Why not, also showing a few embeddings 
    if train_embeddings:
        # get embeddings from the model 
        try:rho_etm = etm_model.rho.weight.cpu()
        except:rho_etm = etm_model.rho.cpu()
        queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love', 
                        'intelligence', 'money', 'politics', 'health', 'people', 'family']
        print('\n')
        print('ETM embeddings...')
        for word in queries:
            print('word: {} .. etm neighbors: {}'.format(word, utils.nearest_neighbors(word, rho_etm, idx2word)))
        print('\n')
