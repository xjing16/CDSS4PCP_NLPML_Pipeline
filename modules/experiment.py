import argparse
import json 
import os
import sys

import io

import spacy
import scispacy
from scispacy.linking import EntityLinker
import pubmed_parser as pp
from tqdm import tqdm

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import unidecode
import contractions
import string
import re

import utils

import gensim
from gensim.models import KeyedVectors

import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Get the absolute path under the current working directory
WORKING_PATH = os.getcwd()

DATA_FOLDER = 'data'
PUBMED_PROCESSED_DATA = 'data/pubmed_data/processed_data/processed_full_data.xml'
PUBMED_PROCESSED_DATA_GS = 'data/pubmed_data/processed_data/processed_44_abstracts_test.xml'
HOLDOUT_DATA = 'data/HoldOut_Total.txt'
JSON_DATA = 'data/pubmed_data/json_data_synthetic_labels'
PRE_PROCESSED_DATA = 'data/pubmed_data/preprocess_data'
INTERIM_DATA = 'interim_data/'
VOCAB = INTERIM_DATA+'/vocab/'
MODEL_FOLDER = INTERIM_DATA+'/models/'
RESULT_FOLDER = INTERIM_DATA+'results/'

# json_data_folder = os.path.join(WORKING_PATH, JSON_DATA)
# if not os.path.exists(json_data_folder): os.mkdir(json_data_folder)
# interim_data_folder = os.path.join(WORKING_PATH, INTERIM_DATA)
# if not os.path.exists(interim_data_folder): os.mkdir(interim_data_folder)

stop_words = stopwords.words('english')

#deselect 'no' and 'not' from stop words
stop_words.remove('no')
stop_words.remove('not')

english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')',
                '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{','=','/','>','<','|','+','_','~']

english_punctuations_arr= ''.join(english_punctuations)

class LanguageModel:
    
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_lg")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

        self.raw_data = None
        self.processed_data = None

    def load_pubmed_data(self,path):
        '''
        Load PubMed XML parsed Data with PMIDS
        '''
        self.raw_data = pp.parse_medline_xml(path,
                                    year_info_only=False,
                                    nlm_category=False,
                                    author_list=False,
                                    reference_list=False)

        return self.raw_data

    def sent2words(self, sent):
        '''Pre-process sentence text to words'''
        
        ## Expand Words
        sent = contractions.fix(sent)
        
        ## Convert sentence to lowercase and strip whitespaces
        sent = sent.lower().strip()
        
        ## Convert unicode characters in sentence to ASCII
        sent = unidecode.unidecode(sent)
        sent = sent.encode("ascii", "ignore").decode()
        
        ## Remove URL's from Sentence
        sent = re.sub('http[s]?://\S+', '', sent)

        ## Remove words like 2.0-2
        # sent = re.sub('\d+\.*\d*\-+\d+\.*\d*', ' ', sent)
        sent = re.sub(' \d*\.*\d*\-+\d*\.*\d* ', ' ', sent) ## Updated regex to find 30-, -2, 3-
        
        # Remove punctuation characters
        sent = sent.translate(str.maketrans(english_punctuations_arr,' '*len(english_punctuations_arr)))
        
        pos, length = {}, {}
        words = word_tokenize(sent)
        for i, (w, p) in enumerate(pos_tag(words)):
            # Computational features
            pos[w] = p
            length[w] = len(w)
        
        # Remove Stop Words
        # words = [word for word in word_tokenize(sent) if not word in stop_words]
        
        # Remove punctuation only strings from sentence
        words = [word for word in words if not all(c in string.punctuation for c in word) and len(word)>1]
        
        # Remove Numericals from sentence
        words = [x for x in words if not all(c.isdigit() for c in x)] 
        
        ## Return words, POS tags, lengths of words, processed sentence
        return words, pos, length, ' '.join(words)

    def generate_synthetic_dataset(self, fullData, predictData=False, min_sentences=3, filter_kp = False):
        '''
        Create Train & Test Datasets as JSON with Synthetic Keywords
        '''

        if filter_kp:
            nlp2 = spacy.load('en_ner_bc5cdr_md')
            nlp2.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

        outlier_kws=[
        'http','www','university','department','antibiotic','antimicrobial','institute','ministry', 'pubmed',
        '.gov','.org','.com','.edu','.net',
        'city','disease','injury','trauma','syndrome','country','national','regimen','swelling','cholesterol','cerebrovascular','leukemia'
        'surgery', 'medication','infection','stroke','diabetes','bleeding','comorbid','java','python',
        "united states",'united kingdom','india','china','germany','france','ghana','australia','italy','england','japan',
        'english','spanish','french','british','spain'
        ]

        comb_arr=[]
        pmid_arr =[]
        dataset = []
        p_id=0
        total_len = len(fullData)
        with tqdm(enumerate(fullData),total=total_len) as pbar1:
            for index, paper in pbar1:
        # for paper in fullData:
                comb_arr.append(paper['title']+' '+paper['abstract'])
                pmid_arr.append(paper['pmid'])
                data = {}
                data["id"] = paper['pmid']
                data["title"] = paper['title']
                data["abstract"] = paper['abstract']
                data["methods"] = ""
                data["results"] = ""
                dataset.append(data)
                p_id += 1
                pbar1.set_description("JSON pre-processing is completed for %s-th document!"%(p_id))

        if predictData==True:
            with open(os.path.join(INTERIM_DATA, "testgs.json"), "w") as outfile:
                json.dump(dataset, outfile)
            return True
                
        idx=0
        remove_sent_no_outliers = []
        with tqdm(self.nlp.pipe(comb_arr),total=total_len) as pbar1:
            # for doc in nlp.pipe(comb_arr):
            for doc in pbar1:
                # text = [token.orth_ for token in doc if not token.is_punct | token.is_space] 
                sents = [" ".join([token.orth_.lower() for token in sent if not token.is_punct | token.is_space]) for sent in doc.sents]
                sents = [self.sent2words(sent)[0] for sent in sents]
                text = [token.orth_ for token in doc if not token.is_punct | token.is_space] 

                ## Skip documents without minimum sentences
                if len(sents)<min_sentences:
                    remove_sent_no_outliers.append(idx)
                    idx+=1
                    continue

                kws = [str(x) for x in doc.ents]

                ## Filter KWS for DISEASES/GPE/Institutes
                if filter_kp:
                    rule_out_ents = [str(_) for _ in nlp2(comb_arr[idx]).ents]
                    rule_out_ents = [_ for _ in rule_out_ents if not _.isupper()]
                    if rule_out_ents!=[]:
                        kws = [_ for _ in kws if _ not in rule_out_ents]
                    kws = [_ for _ in kws if all([True if n_ not in _.lower() else False for n_ in outlier_kws])]

                kws.sort(reverse=True, key=len)
                
                dataset[idx]['fullText'] = text
                dataset[idx]["keywords"]= kws
                dataset[idx]['sentences'] = sents
                idx+=1
                pbar1.set_description("Synthetic Keywrods generated for %s-th document!"%idx)

        ## Filter Docs with less than minimum senetences
        i=0
        j=0
        print("Total Articles removed: ",len(remove_sent_no_outliers))
        print([pmid_arr[x] for x in remove_sent_no_outliers])
        while i<idx:
            if i not in remove_sent_no_outliers:
                dataset[j]=dataset[i]
                j+=1
            i+=1
        dataset[j:]=[]

        return dataset

    # create sequences of length 5 tokens
    def create_seq(self, sent, seq_len = 5):
        
        sequences = []

        # if the number of tokens in 'text' is greater than 5
        if len(sent) > seq_len:
            for i in range(seq_len, len(sent)):
                # select sequence of tokens
                seq = sent[i-seq_len:i+1]
                # add to the list
                sequences.append(" ".join(seq))

            return sequences

        # if the number of tokens in 'text' is less than or equal to 5
        else:
            return sent + ['<UNK>']*(len(sent)-seq_len)

    def get_integer_seq(self, seq, token2int):
        return [token2int[w] for w in seq.split()]

class Trainer:

    def get_batches(self, arr_x, arr_y, batch_size):    
        # iterate through the arrays
        prv = 0
        for n in range(batch_size, arr_x.shape[0], batch_size):
            x = arr_x[prv:n,:]
            y = arr_y[prv:n,:]
            prv = n
            yield x, y  

    def train(self, net, epochs=10, batch_size=32, lr=0.001, clip=1):
        
        # optimizer
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        
        # loss
        criterion = nn.CrossEntropyLoss()
        
        # push model to GPU
        net.cuda()

        net.train()
        
        for e in range(epochs):
            
            losses = []

            # initialize hidden state
            h = net.init_hidden(batch_size)
            
            for x, y in self.get_batches(x_int_train, y_int_train, batch_size):
                
                # convert numpy arrays to PyTorch arrays
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                targets = targets.type(torch.LongTensor)
                
                # push tensors to GPU
                inputs, targets = inputs.cuda(), targets.cuda()

                # detach hidden states
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                net.zero_grad()
                
                # get the output from the model
                output, h = net(inputs, h)
                
                # calculate the loss and perform backprop
                loss = criterion(output, targets.view(-1))
                
                losses.append(loss.detach().item())

                # back-propagate error
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)

                # update weigths
                opt.step()            
                
            print("Epoch: {}/{}...".format(e+1, epochs),"mean_loss : %0.2f, Perplexity : %0.2f"%(np.mean(losses), np.exp(np.mean(losses))))

        net.eval()
        losses = []
        h = net.init_hidden(batch_size)
        for x, y in self.get_batches(x_int_test, y_int_test, batch_size):
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            targets = targets.type(torch.LongTensor)

            inputs, targets = inputs.cuda(), targets.cuda()
            h = tuple([each.data for each in h])
            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output, targets.view(-1))

            losses.append(loss.detach().item())

        print("Test Perplexity: %0.2f"%(np.exp(np.mean(losses))))

    # predict next token
    def predict(self, net, tkn, h=None, token2int={}):

        # tensor inputs
        x = np.array([[token2int[tkn]]])
        inputs = torch.from_numpy(x)

        # push to GPU
        inputs = inputs.cuda()

        # detach hidden state from history
        h = tuple([each.data for each in h])

        # get the output of the model
        out, h = net(inputs, h)

        # get the token probabilities
        p = F.softmax(out, dim=1).data

        p = p.cpu()

        p = p.numpy()
        p = p.reshape(p.shape[1],)

        # get indices of top 3 values
        top_n_idx = p.argsort()[-3:][::-1]

        # randomly select one of the three indices
        sampled_token_index = top_n_idx[random.sample([0,1,2],1)[0]]

        # return the encoded value of the predicted char and the hidden state
        return int2token[sampled_token_index], h

    # function to generate text
    def sample(self, net, size, token2int, prime='it is'):
            
        # push to GPU
        net.cuda()
        
        net.eval()

        # batch size is 1
        h = net.init_hidden(1)

        toks = prime.split()

        # predict next token
        for t in prime.split():
            token, h = self.predict(net, t, h, token2int)
        
        toks.append(token)

        # predict subsequent tokens
        for i in range(size-1):
            token, h = self.predict(net, toks[-1], h, token2int)
            toks.append(token)

        return ' '.join(toks)

class EmbeddingsReader:

    @staticmethod
    def from_text(filename, vocab, unif=0.25):
        
        def init_embeddings(vocab_size, embed_dim, uniform):
            return np.random.uniform(-uniform, uniform, (vocab_size, embed_dim))

        with io.open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n ")
                values = line.split(" ")

                if i == 0:
                    # fastText style
                    if len(values) == 2:
                        weight = init_embeddings(len(vocab), values[1], unif)
                        continue
                    # glove style
                    else:
                        weight = init_embeddings(len(vocab), len(values[1:]), unif)
                word = values[0]
                if word in vocab:
                    vec = np.asarray(values[1:], dtype=np.float32)
                    weight[vocab[word]] = vec
        if '<UNK>' in vocab:
            weight[vocab['<UNK>']] = 0.0
        
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, weight.shape[1]
    
    @staticmethod
    def from_binary(filename, vocab, unif=0.25):
        def init_embeddings(vocab_size, embed_dim, uniform):
            return np.random.uniform(-uniform, uniform, (vocab_size, embed_dim))

        def read_word(f):

            s = bytearray()
            ch = f.read(1)

            while ch != b' ':
                s.extend(ch)
                ch = f.read(1)
            s = s.decode('utf-8')
            # Only strip out normal space and \n not other spaces which are words.
            return s.strip(' \n')

        vocab_size = len(vocab)
        with io.open(filename, "rb") as f:
            header = f.readline()
            file_vocab_size, embed_dim = map(int, header.split())
            weight = init_embeddings(len(vocab), embed_dim, unif)
            if '<UNK>' in vocab:
                weight[vocab['<UNK>']] = 0.0
            width = 4 * embed_dim
            for i in range(file_vocab_size):
                word = read_word(f)
                raw = f.read(width)
                if word in vocab:
                    vec = np.fromstring(raw, dtype=np.float32)
                    weight[vocab[word]] = vec
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, embed_dim

class WordLSTM(nn.Module):
    
    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001, bidirection=False, pretrainedWE = None, vocab={}):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.lr = lr
        self.bidirection = bidirection 
        self.num_directional = 2 if bidirection else 1
        self.n_hidden = n_hidden
        self.vocab_size = len(vocab)

        #print(bidirection, self.bidirection, type(self.bidirection))
        
        if pretrainedWE==None:
            self.emb_layer = nn.Embedding(self.vocab_size, 200)

            ## define the LSTM
            self.lstm = nn.LSTM(200, self.n_hidden, n_layers, 
                                dropout=drop_prob, batch_first=True, bidirectional=self.bidirection)
        else:
            self.emb_layer, self.emb_dim = EmbeddingsReader.from_binary(pretrainedWE, vocab)
            self.lstm = nn.LSTM(self.emb_dim, self.n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True, bidirectional=self.bidirection)
        
        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden*self.num_directional, self.vocab_size)      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
        ## pass input through embedding layer
        embedded = self.emb_layer(x)     
        
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        ## pass through a dropout layer
        out = self.dropout(lstm_output)
        
        #out = out.contiguous().view(-1, self.n_hidden) 
        out = out.reshape(-1, self.n_hidden*self.num_directional)

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if GPU is available
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers*self.num_directional, batch_size, self.n_hidden).zero_().cuda(),
                    weight.new(self.n_layers*self.num_directional, batch_size, self.n_hidden).zero_().cuda())
        
        # if GPU is not available
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model Experiments")

    parser.add_argument('-bD','--bidirection', help="Uni/Bi-Directional Context. Options: [True/False]. Default(False)", default=False, type=bool)
    parser.add_argument('-wET','--wordEmbdType', type=str, help="Word Embedding Type. Options: [basic, fastText, word2vec]. Default(basic)", default="basic")
    parser.add_argument("-minSent","--minSentences", default=3, help="Minimum Sentences in Each Text. Deafult(3)", type=int)
    parser.add_argument("-fKP","--filterKP", default=False, help="Filter KP generation for DISEASE/GPE/HEALTH/MEDICATION. Deafult(False)", type=bool)
    parser.add_argument("-seqLen","--seqLength", type=int, default = 128, help="To create Train Dataset with sequence length sliding window over sentence. Deafult(5)")
    parser.add_argument("-bS","--batchSize", type=int, default= 64, help="BiLM Training Batch Size. Default(32)")
    parser.add_argument("-epochs","--epochs", type=int, default=20, help="BiLM Training Epochs. Default(20)")
    parser.add_argument("-preGenSynthDS", "--preGenSyntheticData", default=False, help="Pre-load the Created", type=bool)
    parser.add_argument('-ttSplit', "--traintestSplit", type=int, default=3, help="Train Test Split Parts. Deafult(3) = 1100 Train | 2200 Test")
    parser.add_argument('-nLayers','--noLayers', type=int, default=4, help="No.of LSTM Layers. default(4)")

    args = parser.parse_args()

    print('='*75)
    print("""
    Bi-Directional Context: {}
    Word Embedding: {}
    Minimum Sentences: {}
    Filter KeyPhrase: {}
    DataSet(X,Y) - Sliding Window Sequence Length: {}
    Batch Size: {}
    Epochs: {}
    Use Pre-Generated Synthetic Data: {}
    Train Test Split-Parts: {:.3f}:{:.3f}
    """.format(
        args.bidirection, args.wordEmbdType, args.minSentences,
        args.filterKP, args.seqLength, args.batchSize, args.epochs,
        args.preGenSyntheticData, 1/args.traintestSplit, 1-1/args.traintestSplit
    ))
    print('='*75)

    lm = LanguageModel()

    if not args.preGenSyntheticData:
        ## Read PubMed XML processed Data - For all CDSS
        pubmed_data_folder = os.path.join(WORKING_PATH, PUBMED_PROCESSED_DATA)
        fullData = lm.load_pubmed_data(pubmed_data_folder)

        ## Remove HoldOut Data from Fulltext
        pmids_HO = set(utils.read_text(HOLDOUT_DATA))
        print("\nArticles for Hold-Out: {}\n".format(len(pmids_HO)))
        
        N=len(fullData)
        i=0
        j=0
        while i<N:
            if fullData[i]['pmid'] not in pmids_HO:
                fullData[j]=fullData[i]
                j+=1
            i+=1
        fullData[j:]=[]
        
        ## Processed Data
        processed_data = lm.generate_synthetic_dataset(fullData, min_sentences=args.minSentences, filter_kp=args.filterKP)

        dataset_length = len(processed_data)

        train_data = processed_data[:dataset_length//args.traintestSplit]
        test_data = processed_data[dataset_length//args.traintestSplit:]

        with open(os.path.join(INTERIM_DATA, "train.json"), "w") as outfile:
            json.dump(train_data, outfile)

        with open(os.path.join(INTERIM_DATA, "test.json"), "w") as outfile:
            json.dump(test_data, outfile) 
    else:
        train_data = utils.read_json_array(os.path.join(INTERIM_DATA, "train.json"))
        test_data = utils.read_json_array(os.path.join(INTERIM_DATA, "test.json"))

    ## Make up some training data
    train_data = utils.read_json_array(os.path.join(INTERIM_DATA, "train.json"))
    test_data = utils.read_json_array(os.path.join(INTERIM_DATA, "test.json"))

    training_data = []
    for article in train_data:
        _, labels, sentences = mark_keyword_all_sentences(article['keywords'], article['sentences'])
        for x,y in zip(sentences, labels):
            if len(x)>3:
                training_data.append((x,y))

    testing_data = []
    for article in test_data:
        _, labels, sentences = mark_keyword_all_sentences(article['keywords'], article['sentences'])
        for x,y in zip(sentences, labels):
            if len(x)>3:
                testing_data.append((x,y))

    #print('Sample: ', train_corpus_sentences[0])

    train_seqs = [lm.create_seq(i, seq_len=args.seqLength) for i in train_corpus_sentences]
    test_seqs = [lm.create_seq(i, seq_len=args.seqLength) for i in test_corpus_sentences]

    print(train_seqs[0])
    sys.exit(0)

    # create inputs and targets (x and y)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for s in train_seqs:
        x_train.append(" ".join(s.split()[:-1]))
        y_train.append(" ".join(s.split()[1:]))

    for s in test_seqs:
        x_test.append(" ".join(s.split()[:-1]))
        y_test.append(" ".join(s.split()[1:]))

    if args.wordEmbdType == "basic":
        # create integer-to-token mapping
        int2token = {0:'<UNK>'}
        cnt = 1

        ## List of Vocabulary/words from all sentences in courpus
        data = []
        for sent in train_corpus_sentences+test_corpus_sentences:
            data.extend(sent)

        ## Create integer-to-word mapping
        for w in set(data):
            int2token[cnt] = w
            cnt+= 1

        # create token-to-integer mapping
        token2int = {t: i for i, t in int2token.items()}

        # set vocabulary size
        vocab_size = len(int2token)
        print("Vocabulary Size: {}".format(vocab_size))

        word_embedding_file = None
    elif args.wordEmbdType == "word2vec":

        ## List of Sentences
        data = train_corpus_sentences+test_corpus_sentences

        modelW2V = gensim.models.Word2Vec(
            sentences=data,
            size=300,window=5,min_count=0,workers=10,
            sg=1,seed=42,compute_loss=True
            )

        word_embedding_file = 'lM_word2vec.bin'
        modelW2V.wv.save_word2vec_format(word_embedding_file, binary=True)
        modelW2V = KeyedVectors.load_word2vec_format(word_embedding_file, binary=True)

        # create integer-to-token mapping
        int2token = {0:'<UNK>'}
        cnt = 1

        for word in modelW2V.vocab.keys():
            int2token[cnt] = word
            cnt+=1

        # create token-to-integer mapping
        token2int = {t: i for i, t in int2token.items()}

        # set vocabulary size
        vocab_size = len(int2token)
        print("Vocabulary Size: {}".format(vocab_size))

    with open("token2int.json","w") as f:
        json.dump(token2int, f)
    with open("int2token.json","w") as f:
        json.dump(int2token, f)

    # lm.token2int = token2int

    # convert text sequences to integer sequences
    x_int_train = [lm.get_integer_seq(i, token2int) for i in x_train]
    y_int_train = [lm.get_integer_seq(i, token2int) for i in y_train]
    x_int_test = [lm.get_integer_seq(i, token2int) for i in x_test]
    y_int_test = [lm.get_integer_seq(i, token2int) for i in y_test]

    x_int_train = [i for i in x_int_train if len(i)==args.seqLength]
    y_int_train = [i for i in y_int_train if len(i)==args.seqLength]
    x_int_test = [i for i in x_int_test if len(i)==args.seqLength]
    y_int_test = [i for i in y_int_test if len(i)==args.seqLength]

    # convert lists to numpy arrays
    x_int_train = np.array(x_int_train, dtype="int32")
    y_int_train = np.array(y_int_train, dtype="int32")
    x_int_test = np.array(x_int_train, dtype="int32")
    y_int_test = np.array(y_int_train, dtype="int32")

    # # instantiate the model
    # net = WordLSTM(n_layers=args.noLayers, bidirection=args.bidirection, pretrainedWE=word_embedding_file, vocab=token2int)

    # # push the model to GPU (avoid it if you are not using the GPU)
    # net.cuda()
    # print(net)
    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # # Train the model
    # trainer = Trainer()
    # trainer.train(net, batch_size = args.batchSize, epochs=args.epochs)
    # print("\n"*2)

    # torch.save(net.state_dict(),"BiLM.pt")

    ## Pre-trained CDSS Word2Vec
    # print(trainer.sample(net, 45, token2int, prime = "clinical decision support systems"),"\n")
    # print(trainer.sample(net, 45, token2int, prime = "prescription errors"),"\n")
    # print(trainer.sample(net, 45, token2int, prime = "one of the"),"\n")
    # print(trainer.sample(net, 45, token2int, prime = "asthma management programs for primary care providers increasing adherence to asthma guidelines"),"\n")