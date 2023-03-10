# Implementing "Improving Clinical Named Entity Recognition with Global Neural Attention"
# https://www.researchgate.net/profile/Chengyu-Wang-4/publication/326501010_Improving_Clinical_Named_Entity_Recognition_with_Global_Neural_Attention_Second_International_Joint_Conference_APWeb-WAIM_2018_Macau_China_July_23-25_2018_Proceedings_Part_II/links/5b9655f0299bf147393991cb/Improving-Clinical-Named-Entity-Recognition-with-Global-Neural-Attention-Second-International-Joint-Conference-APWeb-WAIM-2018-Macau-China-July-23-25-2018-Proceedings-Part-II.pdf

# CODE ADOPT:: https://github.com/littleflow3r/attention-bilstm-for-relation-classification/blob/master/model.py
# NEXT STEPS for Adoption: 
## 1. https://github.com/u784799i/biLSTM_attn/blob/master/model.py
## 2. https://github.com/anonymousDualFraud/DualFraud/blob/10ce4cf669e3dfb2baa3cca847f38e90c49778ed/model/DualFraud.ipynb

# Attention - Pytorch & Keras - https://www.kaggle.com/code/mlwhiz/attention-pytorch-and-keras/notebook
# PyTorch - Bi-LSTM + Attention - https://www.kaggle.com/code/robertke94/pytorch-bi-lstm-attention/notebook

## Pytorch Tutorials:
## https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html
## https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html 

## References:
## https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/
## https://blog.floydhub.com/attention-mechanism/
## https://theaisummer.com/attention/
## https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f [Code in TF]

## Read
## https://arxiv.org/pdf/1703.07713.pdf Hierarchical RNN with Static Sentence-Level Attention for Text-Based Speaker Change Detection
## https://arxiv.org/pdf/1508.04025.pdf - Effective Approaches to Attention-based Neural Machine Translation
## Code: https://sites.google.com/site/textscd/
## Seq2seq and attention - Pytorch Implementation - EN https://wikidocs.net/166653

## https://medium.com/@ashishkankal/generating-sentences-using-lstm-and-attention-neural-network-5aea4d45c751

## Attn-BiLSTM-CRF
# BiLSTM-Attention realizes relation extraction (based on pytorch) - https://programmersought.com/article/437310334572/
# BilSTM + Attention Pytorch Implementation - https://programmersought.com/article/68725413549/

## Char-Level BiLSTM-CRF
# https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import os
import sys
import utils

import unidecode
import contractions
from nltk import sent_tokenize, word_tokenize
import nltk
import logging
import re
import string

import numpy as np

from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from seqeval.scheme import IOB2

from languageModel import EmbeddingsReader

import createDataset as cData

import gensim
from gensim.models import KeyedVectors

import spacy
import scispacy
from scispacy.linking import EntityLinker

from tqdm import tqdm

logger = logging.getLogger(__name__)
stop_words = nltk.corpus.stopwords.words('english')

#deselect 'no' and 'not' from stop words
stop_words.remove('no')
stop_words.remove('not')

english_punctuations = [',', '.', ':', ';', '``','?', '???','???','(', ')',
                '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{','=','/','>','<','|','+','_','~']
english_punctuations_arr= ''.join(english_punctuations)

torch.manual_seed(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKING_PATH = os.getcwd()

PUBMED_PROCESSED_DATA_GS = 'data/pubmed_data/processed_data/processed_44_abstracts_test.xml'
INTERIM_DATA = 'interim_data/'
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
EPOCHS = 100
BATCH_SIZE = 64
# Max Sentence Length for padding
MAX_SENT_LEN = 128
IMPOSSIBLE = -1e4
## Use pre-trained word embeddings
wordEmbdType = "word2vec" # basic/word2vec
WORD_EMBED_FILE = 'lM_word2vec.bin'
print("Word Embedding Type: ", wordEmbdType)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()

def sent2words(sent):
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
    sent = re.sub(' \d*\.*\d*\-+\d*\.*\d* ', ' ', sent)
    
    # Remove punctuation characters
    sent = sent.translate(str.maketrans(english_punctuations_arr,' '*len(english_punctuations_arr)))
    
    pos, length = {}, {}
    words = word_tokenize(sent)
    for i, (w, p) in enumerate(nltk.pos_tag(words)):
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

# Mark Keyphrases with BIO Format Labels
def mark_keyword_all_sentences(keywords, sentences):

    sentence_lengths = [len(sent) for sent in sentences]
    logger.debug("Sentence length: %s"%sentence_lengths)

    complete_text = []
    for sent in sentences:
        complete_text.extend(sent)
    logger.debug("Complete Text: %s"%complete_text)

    complete_text_len = len(complete_text)
    mapper = ['O']*complete_text_len
    kws = [sent2words(x)[0] for x in keywords]

    for kw in kws:
        kw_len=len(kw)
        if kw_len == 0:
            continue
        i=0
        while i<complete_text_len-kw_len:
            if complete_text[i:i+kw_len]==kw and mapper[i:i+kw_len]==['O']*kw_len:
                mapper[i:i+kw_len]=['I-KP']*kw_len
                mapper[i]='B-KP'
            i+=1

    final_mapper = []
    final_tokens = []
    start=0

    for slen in sentence_lengths:
        final_mapper.append(mapper[start:start+slen])
        final_tokens.append(complete_text[start:start+slen])
        start+=slen
    
    return complete_text, final_mapper, final_tokens

def gen_gs_dataset_v2(gsData):
    '''
    Create Gold Standard dataset as JSON with Keywords
    '''
    
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    files = os.listdir("/home/rgoli/MetaMap-src/data/gold_standard/AnnotationResults_Abstracts/done")
    pmid_manual_kw={}
    for file in files:
        if file=='.ipynb_checkpoints': continue
        with open("/home/rgoli/MetaMap-src/data/gold_standard/AnnotationResults_Abstracts/done/"+file,'r') as fp:
            temp = fp.read().splitlines()
            pmid_manual_kw[temp[0].strip()]=temp[1:]

    comb_arr=[]
    pmid_arr =[]
    dataset = []
    p_id=0
    total_len = len(gsData)
    with tqdm(enumerate(gsData),total=total_len) as pbar1:
        for index, paper in pbar1:
            comb_arr.append(paper['title']+' '+paper['abstract'])
            pmid_arr.append(paper['pmid'])
            data = {}
            data["id"] = paper['pmid']
            data["title"] = paper['title']
            data["abstract"] = paper['abstract']
            data["methods"] = ""
            data["results"] = ""
            if paper['pmid'] in pmid_manual_kw:
                data["keywords"]=pmid_manual_kw[paper['pmid']]
            else:
                continue
            dataset.append(data)
            p_id += 1
            pbar1.set_description("JSON pre-processing is completed for %s-th document!"%(p_id))
            
    idx=0
    with tqdm(nlp.pipe(comb_arr),total=total_len) as pbar1:
        for doc in pbar1:
            sents = [" ".join([token.orth_.lower() for token in sent if not token.is_punct | token.is_space]) for sent in doc.sents]
            sents = [sent2words(sent)[0] for sent in sents]
            text = [token.orth_ for token in doc if not token.is_punct | token.is_space] 
            
            dataset[idx]['fullText'] = text
            dataset[idx]['sentences'] = sents
            idx+=1
            pbar1.set_description("GS Data for %s-th document!"%idx)

    return dataset

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1).to(DEVICE)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim).to(DEVICE))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        print(eij.shape)

        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, use_pretrained_embd=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        if use_pretrained_embd == None:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        else:
            print("Using pretrained word embeddings")
            self.word_embeds, self.embedding_dim = use_pretrained_embd
            self.word_embeds.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, HIDDEN_DIM // 2,
                            num_layers=1, bidirectional=True)

        self.lstm2 = nn.LSTM(HIDDEN_DIM*2, HIDDEN_DIM // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(HIDDEN_DIM, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size), requires_grad=True)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_ix[START_TAG], :] = IMPOSSIBLE
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = IMPOSSIBLE

        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=BATCH_SIZE, hidden_dim = HIDDEN_DIM // 2):
        ## NLayers, Batch Size, HiddenSize
        return (torch.randn(2, batch_size, hidden_dim).to(DEVICE),
                torch.randn(2, batch_size, hidden_dim).to(DEVICE))

    def _forward_alg(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.tag_to_ix[START_TAG]] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.tag_to_ix[STOP_TAG]])
        # print("_forward_alg: ", scores,scores.shape)
        return scores

    def attnetwork(self, encoder_out, final_hidden):
        #print("attnnetwork1: ",encoder_out.shape,final_hidden.shape)

        hidden = final_hidden.squeeze(0)
        #print("attnnetwork2: ", hidden.shape)

        attn_weights = torch.bmm(encoder_out, hidden).squeeze(2)
        #print("attnnetwork3: ", attn_weights.shape)

        soft_attn_weights = F.softmax(attn_weights, 1)
        #print("attnnetwork4: ", soft_attn_weights.shape)

        soft_attn_weights = soft_attn_weights.permute(0,2,1)
        #print("attnnetwork4-1: ", soft_attn_weights.shape)
        attn_applied = torch.bmm(final_hidden,soft_attn_weights)

        # print(encoder_out.transpose(1,2).shape)
        # attn_applied = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights).squeeze(2)
        #print("attnnetwork5: ", attn_applied.shape)

        return attn_applied.permute(0,2,1)

    def _get_lstm_features(self, sentences, masks=None):
        # print('_get_lstm_features : ',sentences.shape)
        B, L = sentences.shape

        self.hidden = self.init_hidden(B)
        self.hidden2 = self.init_hidden(B)

        self.attention_layer = Attention(L, B)

        # MAX_SENT_LEN=128 * No. of Sents(Batch Size) * EMBEDDING_DIM=300
        ## 128 * 5 * 300
        embeds = self.word_embeds(sentences).view(L, B, -1) 

        # LSTM_OUT1 = MAX_SENT_LEN=128 * BS * HIDDEN_DIM=256 ; FINAL HIDDEN STATE = 2 * BS * 128 ; FINAL CELL STATE = 2 * BS * 128
        lstm_out1, self.hidden = self.lstm(embeds, self.hidden)
        # 128 * 5 * 256
        # print(lstm_out1.shape) #torch.Size([128, 5, 256])
        # 2 * 5 * 128, 2 * 5 * 128
        # print(self.hidden[0].shape, self.hidden[1].shape)
        # # print(self.hidden[0][-2,:,:].unsqueeze(0).shape, self.hidden[0][-1,:,:].unsqueeze(0).shape) #torch.Size([2, 5, 128]) torch.Size([2, 5, 128])

        # # fbout = lstm_out1[:, :, :HIDDEN_DIM//2]+ lstm_out1[:, :, HIDDEN_DIM//2:] #sum bidir outputs F+B
        # # # print(fbout.shape) # torch.Size([128, 5, 128])
        # # fbout = fbout.permute(1,0,2)
        # # fbhn = (self.hidden[0][-2,:,:]+self.hidden[0][-1,:,:]).unsqueeze(0)
        # # fbcs = (self.hidden[0][-2,:,:]+self.hidden[0][-1,:,:]).unsqueeze(0)

        fbout = lstm_out1.detach().clone()
        fbout = fbout.permute(1,0,2)
        fbhn = torch.cat((self.hidden[0][-2,:,:].unsqueeze(0),self.hidden[0][-1,:,:].unsqueeze(0)),2).repeat(B,1,1).permute(1,2,0)
        fbcs = torch.cat((self.hidden[1][-2,:,:].unsqueeze(0),self.hidden[1][-1,:,:].unsqueeze(0)),2).repeat(B,1,1).permute(1,2,0)
        #print('fbout, fbhn, fbcs') 
        #print(fbout.shape, fbhn.shape, fbcs.shape) # torch.Size([5, 128, 128]) torch.Size([1, 5, 128]) torch.Size([1, 5, 128])

        global_attn = self.attnetwork(fbout, fbcs)
        #print(global_attn.shape) #torch.Size([5, 256])

        # h_lstm_atten = self.attention_layer(fbout, masks)
        # print(h_lstm_atten.shape)

        global_info = torch.cat((fbout, global_attn), 2).permute(1,0,2)
        #print(global_info.shape)

        ## Stack LSTMs
        # MAX_SENT_LEN=128 * BS * HIDDEN_DIM=256
        lstm_out2, self.hidden2 = self.lstm2(global_info, self.hidden2)

        # BS * MAX_SENT_LEN=128 * HIDDEN_DIM=256
        lstm_out = lstm_out2.view(B, L, HIDDEN_DIM)

        # BS * HIDDEN_DIM=256 * TAGSET_SIZE=5
        lstm_feats = self.hidden2tag(lstm_out)
        #sys.exit(0)
        return lstm_feats

    def _score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.tag_to_ix[START_TAG], dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.tag_to_ix[STOP_TAG], last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def _viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.tag_to_ix[START_TAG]] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.tag_to_ix[STOP_TAG]]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def neg_log_likelihood(self, sentences, tags, masks):
        feats = self._get_lstm_features(sentences)

        forward_score = self._forward_alg(feats, masks)
        gold_score = self._score_sentence(feats, tags, masks)
        return (forward_score - gold_score).mean()

    def forward(self, sentence, masks):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats, masks)
        return score, tag_seq

print('<'*30,'START Read Data','>'*30)
## Make up some training data
train_data = utils.read_json_array(os.path.join(INTERIM_DATA, "train.json"))
test_data = utils.read_json_array(os.path.join(INTERIM_DATA, "test.json"))

pubmed_data_folder = os.path.join(WORKING_PATH, PUBMED_PROCESSED_DATA_GS)
test44Abstracts = cData.load_pubmed_data(pubmed_data_folder)
testgs_data = gen_gs_dataset_v2(test44Abstracts)

training_data = []
training_data_doc_sents = [] 
p_id=0
print("Train Keyword Marking:")
with tqdm(enumerate(train_data),total=len(train_data)) as pbar1:
    for idx, article in pbar1:
        _, labels, sentences = mark_keyword_all_sentences(article['keywords'], article['sentences'])
        sent_count = 0
        for x,y in zip(sentences, labels):
            if len(x)>3:
                training_data.append((x,y))
                sent_count +=1
        training_data_doc_sents.append(sent_count)
    p_id += 1
    pbar1.set_description("Train Keyword Marking is completed for %s-th document!"%(p_id))    

testing_data = []
testing_data_doc_sents = []
p_id=0
print("Test Keyword Marking:")
with tqdm(enumerate(test_data),total=len(test_data)) as pbar1:
    for idx, article in pbar1:
        _, labels, sentences = mark_keyword_all_sentences(article['keywords'], article['sentences'])
        sent_count = 0
        for x,y in zip(sentences, labels):
            if len(x)>3:
                testing_data.append((x,y))
                sent_count +=1
        testing_data_doc_sents.append(sent_count)
    p_id += 1
    pbar1.set_description("Test Keyword Makring is completed for %s-th document!"%(p_id))   

gs_data = []
gs_data_doc_sents = []
p_id=0
print("GS Keyword Marking:")
with tqdm(enumerate(testgs_data),total=len(testgs_data)) as pbar1:
    for idx, article in pbar1:
        _, labels, sentences = mark_keyword_all_sentences(article['keywords'], article['sentences'])
        sent_count = 0
        for x,y in zip(sentences, labels):
            if len(x)>3:
                gs_data.append((x,y))
                sent_count +=1
        gs_data_doc_sents.append(sent_count)
    p_id += 1
    pbar1.set_description("Gold Standard Keyword Makring is completed for %s-th document!"%(p_id))         

## For Word Embeddings
train_corpus_sentences = []
for x in train_data:
    train_corpus_sentences.extend(x['sentences'])
print("Total Sentences in Train Corpus: ", len(train_corpus_sentences))

test_corpus_sentences = []
for x in test_data:
    test_corpus_sentences.extend(x['sentences'])
print("Total Sentences in Test Corpus: ", len(test_corpus_sentences))

if wordEmbdType == "basic":
    # create integer-to-token mapping
    ix_to_word = {0:'UNKNOWN'}
    cnt = 1

    ## List of Vocabulary/words from all sentences in courpus
    data = []
    for sent in train_corpus_sentences+test_corpus_sentences:
        data.extend(sent)

    ## Create integer-to-word mapping
    for w in set(data):
        ix_to_word[cnt] = w
        cnt+= 1

    # VOCAB >>> create token-to-integer mapping
    word_to_ix = {t: i for i, t in ix_to_word.items()}

    # set vocabulary size
    vocab_size = len(ix_to_word)
    print("Vocabulary Size: {}".format(vocab_size))

    word_embeddings = None
elif wordEmbdType == "word2vec":
    ## List of Sentences
    data = train_corpus_sentences+test_corpus_sentences
    print("Creating Word2Vec Embedding!!!")
    modelW2V = gensim.models.Word2Vec(
        sentences=data,
        size=300,window=5,min_count=0,workers=10,
        sg=1,seed=42,compute_loss=True
        )

    word_embedding_file = 'lM_word2vec.bin'
    modelW2V.wv.save_word2vec_format(word_embedding_file, binary=True)
    modelW2V = KeyedVectors.load_word2vec_format(word_embedding_file, binary=True)

    # create integer-to-token mapping
    ix_to_word = {0:'UNKNOWN'}
    cnt = 1

    for word in modelW2V.vocab.keys():
        ix_to_word[cnt] = word
        cnt+=1

    # VOCAB >>> create token-to-integer mapping
    word_to_ix = {t: i for i, t in ix_to_word.items()}

    # set vocabulary size
    vocab_size = len(ix_to_word)
    print("Vocabulary Size: {}".format(vocab_size))

    word_embeddings = EmbeddingsReader.from_binary(WORD_EMBED_FILE, word_to_ix)

# ## Load Vocab
# word_to_ix = utils.read_json_array("token2int.json")

tag_to_ix = {"O": 0, "B-KP": 1, "I-KP": 2, START_TAG: 3, STOP_TAG: 4}
ix_to_tag = {0:"O", 1: "B-KP", 2: "I-KP", 3: "O", 4: "O"}
# ix_to_tag = {0:"O", 1: "B-KP", 2: "I-KP", 3: START_TAG, 4: STOP_TAG}
print('<'*30,'END Read Data','>'*30)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, use_pretrained_embd=word_embeddings)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

def get_batches(data, batch_size):    
    # iterate through the arrays
    # prv = 0
    for n in range(0, len(data), batch_size):
    #for n in range(batch_size, len(data), batch_size):
        # print("Prev={}, N={}".format(prv,n))
        data_x = []
        data_masks = []
        data_y = []
        for x,y in data[n:n+batch_size]:
        #for x,y in data[prv:n]:
            for idx, w in enumerate(x[:MAX_SENT_LEN]):
                if w not in word_to_ix:
                    x[idx] = 'UNKNOWN'
            idxs = [word_to_ix[w] for w in x[:MAX_SENT_LEN]]
            data_masks.append(torch.tensor([1 if idx else 0 for idx in idxs], dtype=torch.long))
            data_x.append(torch.tensor(idxs, dtype=torch.long))
            data_y.append(torch.tensor([tag_to_ix[t] for t in y[:MAX_SENT_LEN]], dtype=torch.long))

        # pad first seq to desired length
        data_x[0] = nn.ConstantPad1d((0, MAX_SENT_LEN - data_x[0].shape[0]), 0)(data_x[0])
        data_y[0] = nn.ConstantPad1d((0, MAX_SENT_LEN - data_y[0].shape[0]), 0)(data_y[0])
        data_masks[0] = nn.ConstantPad1d((0, MAX_SENT_LEN - data_masks[0].shape[0]), 0)(data_masks[0])
        
        # pad all seqs to desired length
        data_x_pad = pad_sequence(data_x, batch_first= True)
        data_y_pad = pad_sequence(data_y, batch_first= True)
        data_masks_pad = pad_sequence(data_masks, batch_first= True)
        
        # Convert list of tensors to tuple
        data_x_tuple = tuple(data_x_pad)
        data_y_tuple = tuple(data_y_pad)
        data_masks_tuple = tuple(data_masks_pad)
        
        # Convert tuple of tensors to Stack of Tensors
        data_x = torch.stack(data_x_tuple)
        data_y = torch.stack(data_y_tuple)
        data_masks = torch.stack(data_masks_tuple)
        
        # prv +=1
        # print("Batch: ", prv)
        
        yield data_x, data_y, data_masks

def get_batches_by_doc(data, doc_sents):    
    # iterate through the arrays
    prv = 0
    for n in doc_sents:
        data_x = []
        data_masks = []
        data_y = []
        for x,y in data[prv:prv+n]:
        #for x,y in data[prv:n]:
            for idx, w in enumerate(x[:MAX_SENT_LEN]):
                if w not in word_to_ix:
                    x[idx] = 'UNKNOWN'
            idxs = [word_to_ix[w] for w in x[:MAX_SENT_LEN]]
            data_masks.append(torch.tensor([1 if idx else 0 for idx in idxs], dtype=torch.long))
            data_x.append(torch.tensor(idxs, dtype=torch.long))
            data_y.append(torch.tensor([tag_to_ix[t] for t in y[:MAX_SENT_LEN]], dtype=torch.long))

        # pad first seq to desired length
        try:
            data_x[0] = nn.ConstantPad1d((0, MAX_SENT_LEN - data_x[0].shape[0]), 0)(data_x[0])
            data_y[0] = nn.ConstantPad1d((0, MAX_SENT_LEN - data_y[0].shape[0]), 0)(data_y[0])
            data_masks[0] = nn.ConstantPad1d((0, MAX_SENT_LEN - data_masks[0].shape[0]), 0)(data_masks[0])
        except Exception as e:
            print("Prev={}, N={}".format(prv,n))
            print("Total Len:", len(data), "Access:", prv+n)
            print(data[prv:prv+n])
            print(e)
            print(data_x, data_y)
            # print(len(data_x), len(data_y), print(len(data_masks)))
        
        # pad all seqs to desired length
        data_x_pad = pad_sequence(data_x, batch_first= True)
        data_y_pad = pad_sequence(data_y, batch_first= True)
        data_masks_pad = pad_sequence(data_masks, batch_first= True)
        
        # Convert list of tensors to tuple
        data_x_tuple = tuple(data_x_pad)
        data_y_tuple = tuple(data_y_pad)
        data_masks_tuple = tuple(data_masks_pad)
        
        # Convert tuple of tensors to Stack of Tensors
        data_x = torch.stack(data_x_tuple)
        data_y = torch.stack(data_y_tuple)
        data_masks = torch.stack(data_masks_tuple)
        
        prv += n
        # print("Batch: ", prv)
        
        yield data_x, data_y, data_masks

model_stateDict = model.state_dict()
# print('<'*30,'Bi-LSTM-CRF Model state_dict','>'*30)
# for param_tensor in model_stateDict:
#     print(param_tensor, "\t", model_stateDict[param_tensor].size())

pretrainedLM = torch.load("BiLM.pt")
pretrainedLM_stateDict = pretrainedLM
# print('<'*30,'Pretrained LM state_dict','>'*30)
# for param_tensor in pretrainedLM_stateDict:
#     print(param_tensor, "\t", pretrainedLM_stateDict[param_tensor].size())

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrainedLM_stateDict.items() if k in model_stateDict}
# 2. overwrite entries in the existing state dict
model_stateDict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(model_stateDict)

print('<'*30,'START Train','>'*30)
model.train()
model.to(DEVICE)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
minloss = 10000
for epoch in range(EPOCHS):
    losses= []
    # for inputs, targets, masks in get_batches(training_data, BATCH_SIZE):
    for inputs, targets, masks, in get_batches_by_doc(training_data, training_data_doc_sents):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        inputs, targets, masks = inputs.to(DEVICE), targets.to(DEVICE), masks.to(DEVICE)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(inputs, targets, masks)
        # print("loss: ",loss, loss.shape)
        losses.append(loss.detach().item())

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    
    curr_model_loss = np.mean(losses).item()
    if (epoch+1)%4==0:
        print("Epoch: {}/{} Loss: {:.3f}".format(epoch+1,EPOCHS,curr_model_loss))
    if curr_model_loss<minloss:
        minloss = curr_model_loss
        best_model = model
print('<'*30,'END Train','>'*30)

torch.save(best_model.state_dict(), 'globNeuAttn-best-parameters.pt')
model.load_state_dict(torch.load('globNeuAttn-best-parameters.pt'))

print('<'*30,'START Test','>'*30)
# Check predictions after training
model.eval()
losses= []
with torch.no_grad():
    # for inputs, targets, masks in get_batches(testing_data, BATCH_SIZE):
    for inputs, targets, masks in get_batches_by_doc(testing_data, testing_data_doc_sents):
        inputs, targets, masks = inputs.to(DEVICE), targets.to(DEVICE), masks.to(DEVICE)
        loss = model.neg_log_likelihood(inputs, targets, masks)
        losses.append(loss.detach().item())
print('Test Loss: {:.3f}'.format(np.mean(losses)))
print('<'*30,'END Test','>'*30)

# torch.save(model.state_dict(),"BiLSTM-CRF.pt")

def calc_scores(TP,FN,FP,TN):
    
    accuracy = (TP+TN)/(TP+FN+FP+TN)
    misclassification = (FP+FN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    f1score=(2 * precision * sensitivity) / (precision + sensitivity)
    print("""    Accuracy:\t{:5f}
    Misclassification:\t{:5f}
    Precision:\t{:5f}
    Sensitivity/Recall:\t{:5f}
    Specificity:\t{:5f}
    F1 Score:\t{:5f}""".format(accuracy,misclassification,precision, sensitivity,specificity,f1score))
    return (accuracy,misclassification,precision, sensitivity,specificity,f1score)


def get_metrics(text_arr, kws_arr, gs_arr, mappers=False):
    
    if mappers==False:
        ## For confusion matrix
        test_mappers = []
        gs_mappers = []

        ## Iterate over PMID texts and get mappingf of keywords
        for idx,text in enumerate(text_arr):
            text_len = len(text)

            test_kws = [x.split() for x in kws_arr[idx]]
            gs_kws = [x.split() for x in gs_arr[idx]]

            result = []
            test_mapper = [0]*text_len
            gs_mapper = [0]*text_len

            for kw in test_kws:
                kw_len = len(kw)
                i=0
                while i<text_len-kw_len:
                    if text[i:i+kw_len]==kw and test_mapper[i:i+kw_len]==[0]*kw_len:
                        test_mapper[i:i+kw_len]=[1]*kw_len
                    i+=1

            for kw in gs_kws:
                kw_len=len(kw)
                i=0
                while i<text_len-kw_len:
                    if text[i:i+kw_len]==kw and gs_mapper[i:i+kw_len]==[0]*kw_len:
                        gs_mapper[i:i+kw_len]=[1]*kw_len
                    i+=1 
            test_mappers.append(test_mapper)
            gs_mappers.append(gs_mapper)
    else:
        test_mappers=kws_arr
        gs_mappers=gs_arr
        
    ## Calculate Metrics from Text Mappers
    TP, TN, FN, FP = 0, 0, 0, 0
    N = len(text_arr)
    i=0
    matrix = np.zeros(9).reshape(3,3)
    labels=['B-KP','I-KP','O']
    while i<N:
        matrix += confusion_matrix(gs_mappers[i],test_mappers[i], labels=labels)
        i+=1
    
    print("Confusion Matrix:\n", matrix)
    
    precision_arr = []
    f1score_arr = []
    specificity_arr = []
    sensitivity_arr = []
    accuracy_arr = []
    Total = np.sum(matrix[:,:])
    for i in range(3):
        TP = matrix[i,i]
        FN = np.sum(matrix[i,:]) - TP
        FP = np.sum(matrix[:,i]) - TP
        TN = Total - (FN + FP + TP)
        print("{}\nLabel: {}\nTP:{} FN:{} FP:{} TN:{}\n{}".format('-'*50,labels[i],TP,FN,FP,TN,'-'*50))
        result = calc_scores(TP, FN, FP, TN)   
        accuracy_arr.append(result[0])
        precision_arr.append(result[2])
        sensitivity_arr.append(result[3])
        specificity_arr.append(result[4])
        f1score_arr.append(result[5])
    
    print("=-"*25)
    print("Avg. Accuracy: {:.5f}\nAvg. Precision: {:5f}\nAvg. F1-Score: {:5f}\nAvg. Sensitivity: {:5f}\nAvg. Specificity: {:5f}".format(np.mean(accuracy_arr),np.mean(precision_arr),np.mean(f1score_arr),np.mean(sensitivity_arr),np.mean(specificity_arr)))
    print("=-"*25) 

x_sents = []
y_preds, y_targets = [], []
with torch.no_grad():
    # for inputs, targets, masks in get_batches(testing_data, BATCH_SIZE):
    for inputs, targets, masks in get_batches_by_doc(testing_data, testing_data_doc_sents):
        inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
        _, tags = model(inputs,masks)
        # print(targets, tags, sep='\n')
        for idx, (pred,label) in enumerate(zip(tags, targets)):
            x_sents.append(testing_data[idx][0])
            length = len(x_sents[-1])
            temp_preds = [ix_to_tag[i] for i in pred[:length]]
            temp_preds = temp_preds + ['O'] * (length - len(temp_preds))
            y_preds.append(temp_preds)
            y_targets.append([ix_to_tag[i] for i in label.detach().cpu().tolist()[:length]])

print('<'*30,'TEST Data','>'*30)
get_metrics(x_sents, y_preds, y_targets, mappers=True)
print('SeqEval Metrics:')
print(classification_report(y_targets, y_preds, mode='strict', scheme=IOB2))
print("Precision given by SeqEval: {:.2f}%".format(precision_score(y_targets, y_preds)*100))
print("Recall given by SeqEval: {:.2f}%".format(recall_score(y_targets, y_preds)*100))
print("F1-Score given by SeqEval: {:.2f}%".format(f1_score(y_targets, y_preds)*100))
print("Accuracy given by SeqEval: {:.2f}%".format(accuracy_score(y_targets, y_preds)*100))

x_sents = []
y_preds, y_targets = [], []
with torch.no_grad():
    # for inputs, targets, masks in get_batches(gs_data, BATCH_SIZE):
    for inputs, targets, masks in get_batches_by_doc(gs_data, gs_data_doc_sents):
        inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
        _, tags = model(inputs,masks)
        # print(targets, tags, sep='\n')
        for idx, (pred,label) in enumerate(zip(tags, targets)):
            x_sents.append(gs_data[idx][0])
            length = len(x_sents[-1])
            temp_preds = [ix_to_tag[i] for i in pred[:length]]
            temp_preds = temp_preds + ['O'] * (length - len(temp_preds))
            y_preds.append(temp_preds)
            y_targets.append([ix_to_tag[i] for i in label.detach().cpu().tolist()[:length]])

print('<'*30,'GS Data','>'*30)
get_metrics(x_sents, y_preds, y_targets, mappers=True)
print('SeqEval Metrics:')
print(classification_report(y_targets, y_preds))
print("Precision given by SeqEval: {:.2f}%".format(precision_score(y_targets, y_preds)*100))
print("Recall given by SeqEval: {:.2f}%".format(recall_score(y_targets, y_preds)*100))
print("F1-Score given by SeqEval: {:.2f}%".format(f1_score(y_targets, y_preds)*100))
print("Accuracy given by SeqEval: {:.2f}%".format(accuracy_score(y_targets, y_preds)*100))

c=0
for x,y,z in zip(x_sents,y_preds, y_targets):
    for pred in y:
        if pred not in ['O','B-KP','I-KP']:
            print(x,y,z,sep='\n')
            print('-'*20)  
    c+=1        