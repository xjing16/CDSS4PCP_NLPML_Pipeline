# -*- coding: utf-8 -*-
import json
import os
import re
from turtle import shape
import nltk
import string
from collections import Counter
import numpy as np
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from tqdm import tqdm
import logging
import unidecode
import contractions

from modules.utils import read_json_datas, string_escape, read_json_array
from modules.textrank import textrank
from modules.tf_idf import tf_idf

import config

stop_words = nltk.corpus.stopwords.words('english')

#deselect 'no' and 'not' from stop words
stop_words.remove('no')
stop_words.remove('not')

english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')',
                '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{','=','/','>','<','|','+','_','~']

english_punctuations_arr= ''.join(english_punctuations)
stemmer = PorterStemmer()

logger = logging.getLogger(__name__)

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
    sent = re.sub('\d+\.*\d*\-+\d+\.*\d*', ' ', sent)
    
    # Remove punctuation characters
    sent = sent.translate(str.maketrans(english_punctuations_arr,' '*len(english_punctuations_arr)))
    
    pos, length = {}, {}
    words = word_tokenize(sent)
    for i, (w, p) in enumerate(nltk.pos_tag(words)):
        # Computational features
        pos[w] = p
        length[w] = len(w)
    
    # Remove Stop Words
    words = [word for word in word_tokenize(sent) if not word in stop_words]
    
    # Remove punctuation only strings from sentence
    words = [word for word in words if not all(c in string.punctuation for c in word) and len(word)>1]
    
    # Remove Numericals from sentence
    words = [x for x in words if not all(c.isdigit() for c in x)] 
    
    ## Return words, POS tags, lengths of words, processed sentence
    return words, pos, length, ' '.join(words)

# Mark Keyphrases with BIO Format Labels
def mark_keyword_all_sentences(keywords, sentences):

    sentence_lengths = [len(x.split()) for x in sentences]
    logger.debug("Sentence length: %s"%sentence_lengths)

    complete_text = ' '.join(sentences).split()
    logger.debug("Complete Text: %s"%complete_text)

    complete_text_len = len(complete_text)
    mapper = ['O']*complete_text_len
    kws = [sent2words(x)[0] for x in keywords]
    # print("Processed Keywords")
    # print(kws)

    for kw in kws:
        kw_len=len(kw)
        if kw_len == 0:
            continue
        i=0
        while i<complete_text_len-kw_len:
            if complete_text[i:i+kw_len]==kw and mapper[i:i+kw_len]==['O']*kw_len:
                mapper[i:i+kw_len]=['I-KP']*kw_len
                mapper[i]='B-KP'

                # if kw_len>=2:
                #     mapper[i+kw_len-1]="I-KP"
                # if kw_len==1:
                #     mapper[i]="B-KP"

            i+=1

    # print(complete_text)
    # print(mapper)
    # for i,j in zip(complete_text,mapper):
    #     print(j,'\t',i)

    final_mapper = []
    final_tokens = []
    start=0
    # print(sentence_lengths, len(mapper))
    for slen in sentence_lengths:
        final_mapper.append(mapper[start:start+slen])
        final_tokens.append(complete_text[start:start+slen])
        start+=slen

    # for i,j in zip(sentences,final_mapper):
    #     print(i,j)
    
    return complete_text, final_mapper, final_tokens

# Mark Keywords
def mark_keyword(keywords, words):
    # Initial marker variable
    text = " ".join(words)
    position, text_len  = ['O'] * len(words), len(text)
    for keyword in keywords:
        keyword_length, kw_num = len(keyword), len(keyword.split(" "))
        index_list = [(i.start(), i.end()) for i in re.finditer(keyword.replace("+", "\+"), text)]
        for start, end in index_list:
            if text_len == keyword_length \
                   or (end == text_len and text[start - 1] == ' ') \
                   or (start == 0 and text[end] == ' ') \
                   or (text[start - 1] == ' ' and text[end] == ' '):
                p_index = len(text[:start].split(" ")) - 1
                if kw_num == 1:
                    position[p_index] = 'B-KP'
                else:
                    position[p_index] = 'B-KP'
                    for i in range(1, kw_num - 1):
                        position[p_index + i] = 'I-KP'
#                     position[p_index + kw_num - 1] = 'L-KP'
    return " ".join(words), position

# Construct features of data
def build_features_infer(document, sections, mode, save_folder):
    feature_data=[]
    wfof_dict_total, complete_text_total, tokens_list_total, labels_list_total  = [], [], [], []

    # sentence segmentation
    sentences = []
    for section in sections:
        text = document[section]
        if type(text)==str:
            sentences.extend(sent_tokenize(text))
        elif type(text)==list:
            for item in text:
                sentences.extend(sent_tokenize(item))
    logger.debug(sentences)

    # Empty Keywords array
    keywords = document['keywords']

    ## Complete text of all sections
    plain_text = ' '.join(sentences).lower().strip()


    ## feature word first occurence in complete text - WFOF
    wfof_dict = {}
    for word in word_tokenize(plain_text):
        # Skip the word if it is puncutation or a stop word
        if word in english_punctuations+stop_words: continue

        # Search word in the complete text
        item = re.search(string_escape(word),plain_text)
        if item!=None:
            wfof_dict[word] = item.start()/len(plain_text)

    # features POS tag and length (LEN)
    pos_list, length_list = [], []
    tokens_list, labels_list = [], []
        
    for idx, sentence in enumerate(sentences):
        
        # Pre-process sentence to words
        words, pos, length, sentence = sent2words(sentence)
        
        # replace the original sentence with punct removed sentence
        sentences[idx]=sentence
            
        pos_list.append(pos)
        length_list.append(length)
            
    ## Add BIO Labels to the sentences
    complete_text, labels, tokens = mark_keyword_all_sentences(keywords, sentences)
    tokens_list.extend(tokens)
    labels_list.extend(labels)

    # wfof_dict_total.append(wfof_dict)
    # tokens_list_total.append(tokens_list)
    # labels_list_total.append(labels_list)
    complete_text_total.append(complete_text)

    features = {
        'wfof': wfof_dict,
        'pos': pos_list,
        'length': length_list
    }

    feature_data.append([document['id'], tokens_list, labels_list, keywords, features])

    ## features TF-IDF and Text Rank
    tf_idf_dict = tf_idf(complete_text_total)
    textrank_dict = textrank(complete_text_total)

    ## Merge Informations
    return_datas, id_keywords = {}, {}
    for d_index, data in enumerate(feature_data):
        p_id, rev_data = data[0], []
        id_keywords[p_id] = data[3] # Assignment Keywords
        # labels = data[2]
        for i, (tokens, labels) in enumerate(zip(data[1], data[2])):
            if len(tokens) <= 3: continue
            POS = []
            LEN = np.zeros(shape=(len(tokens)))
            WFOF = np.zeros(shape=(len(tokens)))
            TI = np.zeros(shape=(len(tokens)))
            TR = np.zeros(shape=(len(tokens)))
            for j in range(len(tokens)):
                token = tokens[j]
                try:
                    POS.append(data[4]['pos'][i][token])
                    LEN[j] = data[4]['length'][i][token]
                    WFOF[j] = data[4]['wfof'][token]
                    TI[j] = tf_idf_dict[d_index][token]
                    TR[j] = textrank_dict[d_index][token]
                except:
                    pass
            rev_data.append([" ".join(tokens), labels, POS, LEN, WFOF, TI, TR])
        return_datas[p_id] = rev_data

    return return_datas, id_keywords

# Construct features of data
def build_features(path, sections, mode, save_folder):
    feature_data=[]
    wfof_dict_total, complete_text_total, tokens_list_total, labels_list_total  = [], [], [], []

    temp_json_read = read_json_array(path)
    count=0
    with tqdm(enumerate(temp_json_read),total=len(temp_json_read)) as pbar1:
        for pidx, document in pbar1:

            # sentence segmentation
            sentences = []
            for section in sections:
                text = document[section]
                if type(text)==str:
                    sentences.extend(sent_tokenize(text))
                elif type(text)==list:
                    for item in text:
                        sentences.extend(sent_tokenize(item))
            logger.debug(sentences)

            # pre-process keywords - remove punctuations
            keywords = document['keywords']
            # keywords = [kw for kw in document['keywords']]
            logger.debug(keywords)

            ## Complete text of all sections
            plain_text = ' '.join(sentences).lower().strip()

            # if not os.path.exists(cache_path):

            ## feature word first occurence in complete text - WFOF
            wfof_dict = {}
            for word in word_tokenize(plain_text):
                # Skip the word if it is puncutation or a stop word
                if word in english_punctuations+stop_words: continue

                # Search word in the complete text
                item = re.search(string_escape(word),plain_text)
                if item!=None:
                    wfof_dict[word] = item.start()/len(plain_text)

            # features POS tag and length (LEN)
            pos_list, length_list = [], []
            tokens_list, labels_list = [], []
            
            for idx, sentence in enumerate(sentences):
                
                # Pre-process sentence to words
                words, pos, length, sentence = sent2words(sentence)
                
                # replace the original sentence with punct removed sentence
                sentences[idx]=sentence
                    
                pos_list.append(pos)
                length_list.append(length)
                
            ## Add BIO Labels to the sentences
            complete_text, labels, tokens = mark_keyword_all_sentences(keywords, sentences)
            tokens_list.extend(tokens)
            labels_list.extend(labels)

            logger.debug(wfof_dict)
            logger.debug(pos_list)
            logger.debug(length_list)
            logger.debug(labels_list)
            logger.debug(tokens_list)

            # wfof_dict_total.append(wfof_dict)
            # tokens_list_total.append(tokens_list)
            # labels_list_total.append(labels_list)
            complete_text_total.append(complete_text)

            features = {
                'wfof': wfof_dict,
                'pos': pos_list,
                'length': length_list
            }

            feature_data.append([document['id']+str(pidx), tokens_list, labels_list, keywords, features])

            pbar1.set_description("%s-th document processed"%(pidx+1))  

    ## features TF-IDF and Text Rank
    tf_idf_dict = tf_idf(complete_text_total)
    textrank_dict = textrank(complete_text_total)

    ## Merge Informations
    return_datas, id_keywords = {}, {}
    with tqdm(enumerate(feature_data)) as pbar2:
        for d_index, data in pbar2:
            p_id, rev_data = data[0], []
            id_keywords[p_id] = data[3] # Assignment Keywords
            # labels = data[2]
            for i, (tokens, labels) in enumerate(zip(data[1], data[2])):
                if len(tokens) <= 3: continue
                POS = []
                LEN = np.zeros(shape=(len(tokens)))
                WFOF = np.zeros(shape=(len(tokens)))
                TI = np.zeros(shape=(len(tokens)))
                TR = np.zeros(shape=(len(tokens)))
                for j in range(len(tokens)):
                    token = tokens[j]
                    try:
                        POS.append(data[4]['pos'][i][token])
                        LEN[j] = data[4]['length'][i][token]
                        WFOF[j] = data[4]['wfof'][token]
                        TI[j] = tf_idf_dict[d_index][token]
                        TR[j] = textrank_dict[d_index][token]
                    except:
                        pass
                rev_data.append([" ".join(tokens), labels, POS, LEN, WFOF, TI, TR])
            return_datas[p_id] = rev_data

            pbar2.set_description("The %s-th document feature is completed" % (d_index + 1))
    return return_datas, id_keywords

def build_datas(path, fields, mode, save_folder):
    # Datas annotation and features combination
    global js
    wfof_dict_list, wff_dict_list, \
    wfr_dict_list, iwor_dict_list, \
    iwot_dict_list = [], [], [], [], []
    cache_path = os.path.join(save_folder, "cache_%s"%mode)
    if os.path.exists(cache_path): js = json.load(
        open(cache_path, 'r', encoding='utf-8'))
    p_id, id_datas = 0, []
    temp_json_read = read_json_array(path)
    with tqdm(enumerate(temp_json_read), total=len(temp_json_read)) as pbar1:
        for index, text in pbar1:
            # Sentence segmentation
            sentences = []
            for field in fields:
                content = text[field]
                if type(content) == str:
                    sentences.extend(sent_tokenize(content))
                elif type(content) == list:
                    for item in content:
                        sentences.extend(sent_tokenize(item))
            # Processing keywords
            keywords = []
            for keyphase in text['keywords']:
                keyphase = keyphase.lower().strip()
                tokens = [stemmer.stem(token.strip())
                          for token in sent2words(keyphase)[0]
                          if token.strip() not in english_punctuations]
                keywords.append(" ".join(tokens))
            # ============================================================
            wfof_dict, wff_dict, wfr_dict, \
            iwor_dict, iwot_dict = {}, {}, {}, {}, {}
            
            if not os.path.exists(cache_path):
                # feature 1: word first occurrence in full_text (position)
                plain_text = ' '.join(sentences).lower().strip()
                for word in word_tokenize(plain_text):
                    if word in english_punctuations + stop_words: continue
                    item = re.search(string_escape(word), plain_text)
                    if item != None:
                        wfof_dict[stemmer.stem(word)] = item.start()/len(plain_text)
                wfof_dict_list.append(wfof_dict)
                # feature 2: Full text word frequency count
                wff_dict = Counter([stemmer.stem(token.strip())
                                    for sentence in sent_tokenize(plain_text)
                                    for token in sent2words(sentence)[0]
                                    if token.strip() not in english_punctuations + stop_words])
                wff_dict_list.append(wff_dict)
                # feature 3: Does it appear in the title
                plain_text = text['title'].lower().strip()
                for word in word_tokenize(plain_text):
                    if word in english_punctuations + stop_words: continue
                    word = stemmer.stem(word.strip().lower())
                    iwot_dict[word] = 1
                iwot_dict_list.append(iwot_dict)
            else:
                wfof_dict = js["wfof_dict_list"][index]
                wff_dict = js["wff_dict_list"][index]
                iwot_dict = js["iwot_dict_list"][index]
            # fs.6-7 pos and length
            pos_list, length_list = [], []
            tokens_list, labels_list = [], []
            for sentence in sentences:
                ## Expand Words
                # sentence = contractions.fix(sentence)
                
                ## Convert sentence to lowercase and strip whitespaces
                # sentence = sentence.lower().strip()

                ## Remove Non-ASCII characters from the sentence
                # sentence = unidecode.unidecode(sentence)
                # sentence = sentence.encode("ascii", "ignore").decode()
                
                ## Remove URL from sentence
                # sentence = re.sub('http[s]?://\S+', '', sentence)

                ## Remove words like 2.0-2
                # sentence = re.sub('\d+\.*\d*\-+\d+\.*\d*', ' ', sentence)

                # words = word_tokenize(sentence)
                
                ## Remove punctuation only strings from sentence
                # words = [word for word in words if not all(c in string.punctuation for c in word) and len(word)>1]
                
                ## Remove Numericals from sentence
                # words = [x for x in words if not all(c.isdigit() for c in x)] 
                
                words = sent2words(sentence)[0]
                # print(words)
                
                if len(words) <= 3: continue

                pos, length = {}, {}
                for i, (w, p) in enumerate(nltk.pos_tag(words)):
                    # Stem extraction
                    w_stem = stemmer.stem(w)
                    # Computational features
                    pos[w_stem] = p
                    length[w_stem] = len(w_stem)
                pos_list.append(pos)
                length_list.append(length)

                # Stemming on Tokens
                tokens = [stemmer.stem(token.strip()) for token in words
                          if token.strip() not in english_punctuations + stop_words]

                _, labels = mark_keyword(keywords, tokens)
                labels_list.append(labels)
                tokens_list.append(tokens)
            # features = {
            #     'wff': wff_dict,
            #     'wfof': wfof_dict,
            #     'iwot': iwot_dict,
            #     'pos': pos_list,
            #     'length': length_list
            # }
            
            features = {
                'wfof': wfof_dict,
                'pos': pos_list,
                'length': length_list
            }
            
            id_datas.append([text['id']+str(p_id), tokens_list, labels_list, keywords, features])
            # Document ID suffix
            p_id += 1
            pbar1.set_description("The %s-th document processing is completed!"%p_id)
    if not os.path.exists(cache_path):
        json.dump({
            "wfof_dict_list": wfof_dict_list,
            "wff_dict_list": wff_dict_list,
            "iwot_dict_list": iwot_dict_list
        }, open(cache_path, 'w', encoding='utf-8'))
    # ======================================================
    # Datas transferred to TF-IDF and textrank
    texts_words = []
    for data in id_datas:
        tokens_list = data[1]
        tokens = [token for tokens in tokens_list for token in tokens]
        texts_words.append(tokens)
    # features 8-9: The TF-IDF and textrank features are calculated
    tf_idf_dict = tf_idf(texts_words)
    textrank_dict = textrank(texts_words)
    # Merge informations
    return_datas, id_keywords = {}, {}
    with tqdm(enumerate(id_datas)) as pbar2:
        for d_index, data in pbar2:
            p_id, rev_data = data[0], []
            keywords = data[3]
            for i, (tokens, labels) in enumerate(zip(data[1], data[2])):
                if len(tokens) <= 3: continue
                POS = []
                # WFF = np.zeros(shape=(len(tokens)))
                WFOF = np.zeros(shape=(len(tokens)))
                # IWOT = np.zeros(shape=(len(tokens)))
                LEN = np.zeros(shape=(len(tokens)))
                TI = np.zeros(shape=(len(tokens)))
                TR = np.zeros(shape=(len(tokens)))
                for j in range(len(tokens)):
                    token = tokens[j]
                    try:
                        POS.append(data[4]['pos'][i][token])
                        LEN[j] = data[4]['length'][i][token]
                        # WFF[j] = data[4]['wff'][token]
                        WFOF[j] = data[4]['wfof'][token]
                        TI[j] = tf_idf_dict[d_index][token]
                        TR[j] = textrank_dict[d_index][token]
                        # IWOT[j] = data[4]['iwot'][token]
                    except:
                        pass
                assert len(tokens) == len(POS)
                rev_data.append([" ".join(tokens), labels, POS, LEN, WFOF, TI, TR])
            return_datas[p_id] = rev_data
            id_keywords[p_id] = keywords
            pbar2.set_description("The %s-th document feature is completed" % (d_index + 1))
    return return_datas, id_keywords

def save_datas(datas, keywords, data_type, save_folder):
    '''Save Dataset with Features'''
    POS_VOCAB, WORD_VOCAB, CHAR_VOCAB = [], [], []
    texts, curr_index, last_index, infos = [], 0, 0, {}
    for key, data in datas.items():
        count, sentences = 0, []
        for item in data:
            temp, features, sentence = [], [], item[0]
            if sentence == None: continue
            POS_VOCAB.extend(item[2])
            WORD_VOCAB.extend(word_tokenize(sentence))
            CHAR_VOCAB.extend([i for i in sentence])
            for index in range(1, len(item)):
                if type(item[index]) == list:
                    val = " ".join(item[index])
                    features.append(val)
                else:
                    val = " ".join(np.round(item[index], 8).astype(np.str).tolist())
                    features.append(val)
            sentences.append(sentence)
            temp.append(sentence)
            temp.extend(features)
            texts.append(temp)
            count += 1
        infos[key] = [keywords[key]]
        infos[key].append(last_index)
        curr_index += count
        last_index = curr_index # lines of data, keywords associated in train.txt file
        infos[key].append(curr_index)
    with open(os.path.join(save_folder, data_type+".txt"), "w", encoding="utf-8") as fp:
        for text in texts: fp.write(" <sep> ".join(text) + '\n')
    with open(os.path.join(save_folder, data_type+"_info.json"), "w", encoding="utf-8") as fp:
        for key, val in infos.items():
            v = json.dumps({key: val})
            fp.write(v + '\n')
    print("\033[34mINFO: %s data saving completed!\033[0m" % data_type)
    return POS_VOCAB, WORD_VOCAB, CHAR_VOCAB

# Partition data-set
def build_data_sets(json_data_folder, sections, interim_folder, stemming='no'):
    
    # 1. Building configuration folder
    root_vocab_path = os.path.join(interim_folder, "vocab")
    if not os.path.exists(root_vocab_path): os.mkdir(root_vocab_path)
    # else:
    #     for file_name in os.listdir(root_vocab_path):
    #         os.remove(os.path.join(root_vocab_path, file_name))
    
    # 2. Construction datas
    train_path = os.path.join(json_data_folder, 'train.json')
    test_path = os.path.join(json_data_folder, 'test.json')
    
    logger.info('in build data set')
    
    if stemming=="no":
        train_features, train_keywords = build_features(train_path, sections, 'train', interim_folder)
        test_features, test_keywords = build_features(test_path, sections, 'test', interim_folder)
    else:
        train_features, train_keywords = build_datas(train_path, sections, 'train', interim_folder)
        test_features, test_keywords = build_datas(test_path, sections, 'test', interim_folder)       

    POS_VOCAB1, WORD_VOCAB1, CHAR_VOCAB1 = save_datas(train_features, train_keywords, "train", interim_folder)
    POS_VOCAB2, WORD_VOCAB2, CHAR_VOCAB2 = save_datas(test_features, test_keywords, "test", interim_folder)
    WORD_VOCAB = WORD_VOCAB1 + WORD_VOCAB2
    CHAR_VOCAB = CHAR_VOCAB1 + CHAR_VOCAB2
    POS_VOCAB = POS_VOCAB1 + POS_VOCAB2
    # 3. Save word dictionary
    WORD_VOCAB = ["[PAD]", "[UNK]"] + list(set(WORD_VOCAB))
    with open(os.path.join(root_vocab_path, 'word_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(WORD_VOCAB))
    # Save character dictionary
    CHAR_VOCAB = ["[PAD]", "[UNK]"] + list(set(CHAR_VOCAB))
    with open(os.path.join(root_vocab_path, 'char_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(CHAR_VOCAB))
    # Save Dictionary of part of speech categories
    POS_VOCAB = ["[PAD]", "[UNK]"] + list(set(POS_VOCAB))
    with open(os.path.join(root_vocab_path, 'pos_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(POS_VOCAB))
    print("\033[34mINFO: Dataset partition completed\033[0m")
    
def build_testgs_data_set(json_data_folder, sections, interim_folder, stemming='no', ds_filename = 'testgs.json'):      
    testgs_path = os.path.join(json_data_folder, ds_filename)
    
    ds_type = ds_filename.split('.')[0]

    if stemming=="no":
        predict_features, predict_keywords = build_features(testgs_path, sections, ds_type, interim_folder)
    else:
        predict_features, predict_keywords = build_datas(testgs_path, sections, ds_type, interim_folder)      

    save_datas(predict_features, predict_keywords, ds_type, interim_folder)

def build_infer_data_set(data, sections, interim_folder, stemming='no'):
    
    predict_features, predict_keywords = build_features_infer(data, sections, 'infer', interim_folder)

    # if stemming=="no":
    #     pass
    #     #predict_features, predict_keywords = build_features(json_file, sections, 'predict', interim_folder)
    # else:
    #     pass
    #     #predict_features, predict_keywords = build_datas(json_file, sections, 'predict', interim_folder)      

    save_datas(predict_features, predict_keywords, "predict", interim_folder)