#########################################
## Usage: Run 50 times Experiment Simulation for fine-tuning Synthetic CDSS sciSpacy with Random GS 66 + Test on Random GS 67
## Author: Rohan Goli
## Date: 09/11/2022
#########################################

import json
import csv
import os
import sys
import unidecode
import contractions
from nltk import sent_tokenize, word_tokenize
import nltk
import logging
import re
import string
import spacy
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from seqeval.scheme import IOB2
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
stop_words = nltk.corpus.stopwords.words('english')
#deselect 'no' and 'not' from stop words
stop_words.remove('no')
stop_words.remove('not')

## Do not remove '.' ',' ';' ':'
english_punctuations = ['``','?', '（','）','(', ')',
                '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{','=','/','>','<','|','+','_','~']
english_punctuations_arr= ''.join(english_punctuations)

def read_json_array(path):
    '''Read JSON given array of dictionaries'''
    with open(path, "r", encoding="utf-8") as fp:
        datas = json.loads(fp.read())
    return datas

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

def createBERTtSV(data,filename):
    # import csv
    # trainCSV = "../data/trainBERT.tsv"
    f = open(filename,'w')
    f.close()

    for article in data:
        textData = []
        for sent in sent_tokenize(article['title']+' '+article['abstract']):
            words, _, _, _ = sent2words(sent)
            textData.append(words)
        _, mapper, tokens = mark_keyword_all_sentences(article['keywords'],textData)
        with open(filename,'a') as f:
            print("-DOCSTART- -X- O O", file=f)
            for sentM, sentT in zip(mapper, tokens):
                for wordM, wordT in zip(sentM, sentT):
                    print("{}\t{}".format(wordT,wordM), file=f)

def getTargetPreds(data):
    comb_arr = []
    for article in data:
        comb_arr.append(article['title']+' '+article['abstract'])

    y_preds, y_targets = [], []
    for idx, doc in enumerate(nlp.pipe(comb_arr)):

        kws = [str(x) for x in doc.ents]
        textData = []
        for sent in sent_tokenize(article['title']+' '+article['abstract']):
            words, _, _, _ = sent2words(sent)
            textData.append(words)
        # textData = [word_tokenize(sent) for sent in sent_tokenize(data[idx]['title']+' '+data[idx]['abstract'])]

        _, target, _ = mark_keyword_all_sentences(data[idx]['keywords'],textData)
        _, predict, _ = mark_keyword_all_sentences(kws,textData)
        #print(gs40Data[idx]['id'],gs40Data[idx]['keywords'],kws,sep='\n')

        #print(target,predict,sep='\n')
        y_preds.extend(predict)
        y_targets.extend(target)
    
    return y_preds, y_targets

def printMetrics(y_targets, y_preds,custom='',experiment=0):
    print("\nSeqEval Metrics on sciSpacy {}:\n".format(custom))
    print(classification_report(y_targets, y_preds, mode='strict', scheme=IOB2))
    precision = precision_score(y_targets, y_preds)*100
    recall = recall_score(y_targets, y_preds)*100
    f1score = f1_score(y_targets, y_preds)*100
    accuracy = accuracy_score(y_targets, y_preds)*100
    with open('cdssSpacyExp_results.csv', 'a', newline='') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow([experiment, precision, recall, f1score, accuracy])
        
    print("Precision given by SeqEval: {:.2f}%".format(precision))
    print("Recall given by SeqEval: {:.2f}%".format(recall))
    print("F1-Score given by SeqEval: {:.2f}%".format(f1score))
    print("Accuracy given by SeqEval: {:.2f}%".format(accuracy))
    print("-"*60)

SYNTH_SCISPACY_66GS_CONFIG = '''
# This is an auto-generated partial config. To use it with 'spacy train'
# you can run spacy init fill-config to auto-fill all default settings:
# python -m spacy init fill-config ./base_config.cfg ./config.cfg
[paths]
train = null
dev = null
vectors = null
[system]
gpu_allocator = "pytorch"

[nlp]
lang = "en"
pipeline = ["transformer","ner"]
batch_size = 128

[components]

[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "roberta-base"
#name = "allenai/scibert_scivocab_cased"
tokenizer_config = {"use_fast": true}

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.ner]
source = "../cdssSciSpacy/model-best"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = false
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[components.ner.model.tok2vec.pooling]
@layers = "reduce_mean.v1"

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0

[training]
accumulate_gradient = 3
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"

[training.optimizer]
@optimizers = "Adam.v1"

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 5e-5

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
size = 2000
buffer = 256

[initialize]
vectors = ${paths.vectors}

[initialize.before_init]
@callbacks: "spacy.copy_from_base_model.v1"
tokenizer: "../cdssSciSpacy/model-best"
vocab: "../cdssSciSpacy/model-best"
'''
with open('tempSpacy66GS.cfg','w') as f:
    f.write(SYNTH_SCISPACY_66GS_CONFIG)
        
gs42Data = read_json_array("../data/pubmed_data/json_data_synthetic_labels/v2_testgs42.json")
gs91Data = read_json_array("../data/pubmed_data/json_data_synthetic_labels/v2_testgs91.json")
gsCombData = gs42Data+gs91Data
print("Total DS: {}, Train: {} Val: {} Test: {}".format(133, 52, 14,67))

os.system('''
    source activate pytorch
    python -m spacy init fill-config tempSpacy66GS.cfg config.cfg''')

# with open('cdssSpacyExp_results.log','w') as f:
#     f.write("{}\t{}\t{}\t{}\t{}\n".format('No.','Precision','Recall','F1Score','Accuracy'))
with open('cdssSpacyExp_results.csv', 'w', newline='') as csvfile:
    logwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    logwriter.writerow(['No.','Precision','Recall','F1Score','Accuracy'])

## Number of times to Repeat Experiment
repeat = 50
for seed in range(repeat):
    print("Experiment: {}".format(seed+1))
    
    ## Split GS Dataset - 52 Train + 14 Validation + 67 Test
    gsTrainVal, gsTest = train_test_split(gsCombData, test_size=0.50, random_state=seed)
    gsTrain, gsVal = train_test_split(gsTrainVal, test_size=0.20, random_state=seed)
    # print(len(gsTrain),len(gsVal), len(gsTest),sep='\t')
    
    createBERTtSV(gsTrain,'../data/gs66Spacy_T.tsv')
    createBERTtSV(gsVal,'../data/gs66Spacy_V.tsv')

    os.system('''
    source activate pytorch
    cd ../data
    python -m spacy convert gs66Spacy_T.tsv ./ -t json -n 1 -c iob
    python -m spacy convert gs66Spacy_V.tsv ./ -t json -n 1 -c iob
    python -m spacy convert gs66Spacy_T.json ./ -t spacy
    python -m spacy convert gs66Spacy_V.json ./ -t spacy
    cd ../modules
    python -m spacy train -g 0 config.cfg --output ../cdssSciSpacyGS66 --paths.train ../data/gs66Spacy_T.spacy --paths.dev ../data/gs66Spacy_V.spacy''')

    nlp = spacy.load('../cdssSciSpacyGS66/model-best')

    y_preds, y_targets = getTargetPreds(gsTest)
    printMetrics(y_preds, y_targets, '+ Synthetic 1866 Train / 622 Val + 66GS 52 Train + 14 Validation + 67GS Test',seed+1)
