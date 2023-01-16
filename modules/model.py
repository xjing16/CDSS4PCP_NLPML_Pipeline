# -*- coding: utf-8 -*-
import os
import json
from copy import deepcopy

import pandas as pd
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
torch.manual_seed(2021)

from config import MODEL_FOLDER, RESULT_FOLDER
from modules.crf import CRF
from modules.utils import  performance_metrics, read_text, parse_tags, get_metrics
from modules.preProcessData import mark_keyword_all_sentences

from seqeval.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from seqeval.scheme import IOB2

from sklearn_crfsuite import metrics as crfmetrics

names = ['text', 'label', 'POS', 'LEN', 'WFOF', 'TI', 'TR']
# Text processing object
class TextDataSet(Dataset):

    def __init__(self, data_path, vocab_path, texts_features, configs):
        super(TextDataSet, self).__init__()
        word_vocab_path = os.path.join(vocab_path, 'word_vocab.txt')
        char_vocab_path = os.path.join(vocab_path, 'char_vocab.txt')
        pos_vocab_path = os.path.join(vocab_path, 'pos_vocab.txt')

        self.texts_features = deepcopy(texts_features)
        self.word_max_len = configs['word_max_len']
        self.sentence_max_len = configs['sentence_max_len']
        self.data = pd.read_csv(data_path, sep=' <sep> ', engine='python', names=names)
        self.tags = {key:i for i, key in enumerate(read_text(configs["tags_path"]))}
        self.word_vocab = {key:i for i, key in enumerate(read_text(word_vocab_path))}
        self.char_vocab = {key:i for i, key in enumerate(read_text(char_vocab_path))}
        self.pos_vocab = {key:i for i, key in enumerate(read_text(pos_vocab_path))}

    def __getitem__(self, item):
        words2ix = np.zeros(self.sentence_max_len)
        label = np.zeros(self.sentence_max_len)
        words = self.data['text'].iloc[item].split(' ')
        sentence_len = min([len(words), self.sentence_max_len])
        # Construct word encoding sequence
        for index in range(sentence_len):
            word = words[index]
            if word in self.word_vocab.keys():
                words2ix[index] = self.word_vocab[word]
            else:
                words2ix[index] = self.word_vocab['[UNK]']
        label[:sentence_len] = [self.tags[tag] for tag in
                self.data['label'].iloc[item].split(' ')[:sentence_len]]
        # Feature set
        if 'CEM' in self.texts_features: self.texts_features.remove('CEM')
        features = np.zeros((len(self.texts_features), self.sentence_max_len))
        if len(self.texts_features) != 0:
            for i, feature in enumerate(self.texts_features):
                fs_vals = self.data[feature].iloc[item].split(' ')[:sentence_len]
                if feature == 'POS':
                    feature_values = []
                    for val in fs_vals:
                        if val in self.pos_vocab.keys():
                            feature_values.append(self.pos_vocab['[UNK]'])
                        else:
                            feature_values.append(self.pos_vocab[val])
                    features[i][:sentence_len] = feature_values
                else:
                    feature_values = [float(i) for i in fs_vals]
                    features[i][:sentence_len] = feature_values
        # Construct character encoding sequence
        chars2ix = np.zeros(shape=(self.sentence_max_len, self.word_max_len))
        for rank in range(sentence_len):
            chars = []
            for idex, ch in enumerate(words[rank]):
                if ch in self.char_vocab.keys():
                    chars.append(self.char_vocab[ch])
                else:
                    chars.append(self.char_vocab['[UNK]'])
            chars_len = len(chars)

            ## Slice upto maximum word max length or pad zeros if small sentence
            if chars_len >= self.word_max_len:
                chars = chars[:self.word_max_len]
            else:
                zero_pad = [0] * (self.word_max_len - chars_len)
                chars.extend(zero_pad)
            assert len(chars) == self.word_max_len
            chars2ix[rank] = chars
        # Attention mask
        attention_mask = (words2ix > 0)
        return words2ix, chars2ix, label, attention_mask, features, np.array(sentence_len)

    def __len__(self):
        return self.data.shape[0]

# Character-level encoding
class CharEncode(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=1, is_bidirectional=True, dropout=0.5,
                 device=torch.device('cpu')):
        super(CharEncode, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.bidire = 2 if is_bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim,
                              num_layers=num_layers, bidirectional=is_bidirectional)
        self.out = nn.Linear(self.bidire*hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.embedding(input)
        (B, S, W, E) = x.size()
        x = x.reshape(-1, W, E)
        x = x.transpose(0 ,1)
        _, (hn, cn) = self.bilstm(x)
        x = torch.cat([hn[-2], hn[-1]], dim=-1)
        x = self.dropout(x)
        x = self.out(x)
        x = x.reshape(-1, S, E)
        return x

# BilSTM-CRF Model
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_path, configs, features, device=torch.device('cpu')):
        super(BiLSTM_CRF, self).__init__()
        word_vocab_path = os.path.join(vocab_path, 'word_vocab.txt')
        char_vocab_path = os.path.join(vocab_path, 'char_vocab.txt')
        # parameters
        self.hidden_dim = configs["hidden_dim"]
        self.features = features
        self.is_bidirectional = True
        self.tagset_size = len(read_text(configs["tags_path"]))
        self.word_vocab_size = len(read_text(word_vocab_path))
        self.char_vocab_size = len(read_text( char_vocab_path))
        self.use_crf = False if configs["use_crf"] == 'False' else True
        self.is_bidirectional = False if configs["is_bidirectional"] == 'False' else True
        self.num_directional = 2 if self.is_bidirectional else 1
        self.target_size = self.tagset_size + 2 if self.use_crf else self.tagset_size

        # Calculate vector dimension
        if 'CEM' in features:
            self.feature_size = len(features) - 1
            self.input_to_lstm_dim = configs["word_embedding_dim"] + configs["char_embedding_dim"] + self.hidden_dim//2
            self.lstm_to_fc_dim = self.hidden_dim * self.num_directional + configs["char_embedding_dim"] + self.hidden_dim//2
        else:
            self.feature_size = len(features)
            self.input_to_lstm_dim = configs["word_embedding_dim"] + self.hidden_dim//2
            self.lstm_to_fc_dim = self.hidden_dim * self.num_directional + self.hidden_dim//2

        # Embedding layer
        self.word_embedding = nn.Embedding(self.word_vocab_size, configs["word_embedding_dim"], padding_idx=0)
        self.char_embedding = CharEncode(self.char_vocab_size,
                            configs["char_embedding_dim"], configs["char_embedding_dim"], device=device)

        # Processing layers
        self.feature_mapping = nn.Linear(self.feature_size, self.hidden_dim//2)
        self.bilstm = nn.LSTM(self.input_to_lstm_dim, self.hidden_dim,
                num_layers=configs["lstm_num_layers"], bidirectional= self.is_bidirectional)
        self.hidden2tag = nn.Linear(self.lstm_to_fc_dim, self.target_size)
        self.crf_layer = CRF(target_size=self.tagset_size, device=device)

        # Prevent over fitting
        self.dropout = nn.Dropout(p=configs['dropout'])
        self.batch_norm = nn.BatchNorm1d(num_features=configs["sentence_max_len"])
        self.layer_norm1 = nn.LayerNorm(normalized_shape=[configs["sentence_max_len"], self.input_to_lstm_dim])
        self.layer_norm2 = nn.LayerNorm(normalized_shape=[configs["sentence_max_len"], self.lstm_to_fc_dim])

        # Loss function
        self.loss_fuc_crf = self.crf_layer.neg_log_likelihood_loss
        self.loss_fuc_cel = nn.CrossEntropyLoss()

    # Get LSTM outputs
    def lstm_feats(self, input, mask=None):
        char_embed,  features_values = None, input['features'].transpose(1, 2)
        features_values = self.feature_mapping(features_values)
        # Encoding (word and character encoding)
        word_embed = self.word_embedding(input['words2ix'])
        if "CEM" in self.features:
            char_embed = self.char_embedding(input['chars2ix'])
            vec = torch.cat([word_embed, char_embed, features_values], dim=-1)
        else:
            vec = torch.cat([word_embed, features_values], dim=-1)
        vec = self.dropout(vec)
        # vec = self.layer_norm1(vec)

        # input into BILSTM
        vec = vec.transpose(0, 1)
        vec, _ = self.bilstm(vec)
        vec = vec.transpose(0, 1)
        if "CEM" in self.features:
            vec = torch.cat([vec, char_embed, features_values], dim=-1)
        else:
            vec = torch.cat([vec,  features_values], dim=-1)
        vec = self.dropout(vec)
        # vec = self.batch_norm2(vec)
        # vec = self.batch_norm(vec)
        feats = self.hidden2tag(vec)
        return feats

    # Calculate outputs
    def outputs(self, feats, mask=None):
        outputs = self.crf_layer(feats, mask) if self.use_crf \
                      else None, torch.argmax(torch.softmax(feats, dim=-1), dim=-1)
        return outputs

    # Calculate loss values
    def loss(self, input, labels):
        mask = input['attention_mask']
        feats = self.lstm_feats(input, mask)
        loss = self.loss_fuc_crf(feats, mask, labels) if self.use_crf else self.loss_fuc_cel(
                feats.reshape(-1, self.target_size), labels.reshape(-1))
        return loss

    def forward(self, input):
        mask = input['attention_mask']
        feats = self.lstm_feats(input, mask)
        outputs = self.outputs(feats, mask)
        return outputs

# Model training function
def train(file_folder, corpus_type, texts_features=None):
    # File Path
    train_path = os.path.join(file_folder, 'train.txt')
    dev_path = os.path.join(file_folder, 'test.txt')
    dev_info_path = os.path.join(file_folder, 'test_info.json')
    vocab_path = os.path.join(file_folder, 'vocab')

    # Create Folders for Model and Results
    BIN_FODLER = os.path.join(MODEL_FOLDER,'bin')
    if not os.path.exists(MODEL_FOLDER): os.mkdir(MODEL_FOLDER)
    if not os.path.exists(MODEL_FOLDER): os.mkdir(BIN_FODLER)
    if not os.path.exists(RESULT_FOLDER): os.mkdir(RESULT_FOLDER)

    # Load configuration
    config_path = os.path.join(file_folder, "config.json")
    configs = json.load(open(config_path, 'r', encoding='utf-8'))
    print("\033[034mPARAMS: %s\033[0m"%configs)
    ix2tag = {i:key for i, key in enumerate(
            pd.read_csv(configs['tags_path'], header=None, lineterminator='\n').values.flatten().tolist())}
    ix2tag[len(ix2tag)] = 'O'
    ix2tag[len(ix2tag)] = 'O'
    # Dataset processing
    train_dataset = TextDataSet(data_path = train_path,
                    vocab_path = vocab_path, texts_features=texts_features, configs = configs)
    dev_dataset = TextDataSet(data_path = dev_path,
                    texts_features=texts_features, vocab_path = vocab_path, configs = configs)
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size = configs['batch_size'])
    dev_loader = DataLoader(dev_dataset, batch_size = configs['batch_size'])
    # Create BiLSTM-CRF model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_CRF(vocab_path = vocab_path, configs = configs,
                       features=texts_features, device=device).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr = configs['lr'], weight_decay = configs['weight_decay'])
    # Learning rate decay
    scheduler = ReduceLROnPlateau(optimizer, factor = configs['optim_scheduler_factor'],
                                  patience = configs['optim_scheduler_patience'],
                                  verbose = bool(configs['optim_scheduler_verbose']))
    best_p, best_r, best_f1 = 0, 0, 0
    for epoch in range(configs['epochs']):
        model.train()
        losses = []
        with tqdm(train_loader) as pbar_train:
            for words2ix, chars2ix, labels, attention_mask, features, sentence_len in pbar_train:
                inputs = {
                    'words2ix': torch.as_tensor(words2ix, dtype=torch.long).to(device),
                    'chars2ix': torch.as_tensor(chars2ix, dtype=torch.long).to(device),
                    'attention_mask': torch.as_tensor(attention_mask).to(device),
                    'features': torch.as_tensor(features, dtype=torch.float).to(device),
                }
                labels = torch.as_tensor(labels, dtype=torch.long).to(device)
                loss = model.loss(inputs, labels)
                # _, outputs = model(inputs)
                # for ix, (pred, tag) in enumerate(zip(outputs, labels)):
                #     print(pred.detach().cpu().tolist()[: sentence_len[ix]])
                model.zero_grad()
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
                losses.append(loss.item())
                pbar_train.set_description("loss:%s"%round(loss.item(),3))
        scheduler.step(np.average(losses))
        model.eval()
        y_preds, y_targets = [], []
        with tqdm(dev_loader) as pbar_test:
            for words2ix, chars2ix, labels, attention_mask, features, sentence_len in pbar_test:
                inputs = {
                    'words2ix': torch.as_tensor(words2ix, dtype=torch.long).to(device),
                    'chars2ix': torch.as_tensor(chars2ix, dtype=torch.long).to(device),
                    'attention_mask': torch.as_tensor(attention_mask).to(device),
                    'features': torch.as_tensor(features, dtype=torch.float).to(device),
                }
                _, outputs = model(inputs)
                for ix, (pred, tag) in enumerate(zip(outputs, labels)):
                    y_preds.append([ix2tag[i] for i in pred.detach().cpu().tolist()[: sentence_len[ix]]])
                    y_targets.append([ix2tag[i] for i in tag.detach().cpu().tolist()[: sentence_len[ix]]])
                pbar_test.set_description("finish")
        p, r, f1, NUM = performance_metrics(y_preds, dev_info_path)
        if f1 > best_f1:
            best_p, best_r, best_f1 = p, r, f1
            # save model
            tfs_type = "".join([i[0] for i in texts_features]).upper()
            model_path = os.path.join(MODEL_FOLDER, 'bin', "%s_%s.bin" % (corpus_type, tfs_type))
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
            }, open(model_path, 'wb'))
        print("\033[034mEpoch: %s, loss: %s, p: %s, r: %s, f1: %s\033[0m"%(epoch+1, round(np.average(losses), 3), p, r, f1))
    print("\033[034mBEST: best_p: %s, best_r: %s, best_f1: %s\033[0m" % (best_p, best_r, best_f1))
    return best_p, best_r, best_f1

# Model test function
def test(file_folder, corpus_type, data_type, texts_features=None):
    # File path
    test_path = os.path.join(file_folder, '%s.txt'%data_type)
    test_info_path = os.path.join(file_folder, '%s_info.json'%data_type)
    vocab_path = os.path.join(file_folder, 'vocab')
    # Load configuration file
    config_path = os.path.join(file_folder, "config.json")
    configs = json.load(open(config_path, 'r', encoding='utf-8'))
    print("\033[034mPARAMS: %s\033[0m"%configs)
    ix2tag = {i:key for i, key in enumerate(
            pd.read_csv(configs['tags_path'], header=None, sep='\n').values.flatten().tolist())}
    ix2tag[len(ix2tag)] = 'O'
    ix2tag[len(ix2tag)] = 'O'
    test_dataset = TextDataSet(data_path = test_path,
                    texts_features=texts_features, vocab_path = vocab_path, configs = configs)
    # Data Loader
    test_loader = DataLoader(test_dataset, batch_size = configs['batch_size'])
    # Create BiLSTM-CRF model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_CRF(vocab_path = vocab_path, configs = configs,
                       features=texts_features, device=device).to(device)
    print(model)
    # Loading the pre-trained
    tfs_type = "".join([i[0] for i in texts_features]).upper()
    model_path = os.path.join(MODEL_FOLDER, 'bin', "%s_%s.bin" % (corpus_type, tfs_type))
    model_config = torch.load(open(model_path, 'rb'))
    model.load_state_dict(model_config['model'])
    model.eval()
    y_preds, y_targets = [], []
    with tqdm(test_loader) as pbar_test:
        for words2ix, chars2ix, labels, attention_mask, features, sentence_len in pbar_test:
            inputs = {
                'words2ix': torch.as_tensor(words2ix, dtype=torch.long).to(device),
                'chars2ix': torch.as_tensor(chars2ix, dtype=torch.long).to(device),
                'attention_mask': torch.as_tensor(attention_mask).to(device),
                'features': torch.as_tensor(features, dtype=torch.float).to(device),
            }
            _, outputs = model(inputs)
            for ix, (pred, tag) in enumerate(zip(outputs, labels)):
                y_preds.append([ix2tag[i] for i in pred.detach().cpu().tolist()[: sentence_len[ix]]])
                y_targets.append([ix2tag[i] for i in tag.detach().cpu().tolist()[: sentence_len[ix]]])
            pbar_test.set_description("finish")
    p, r, f1, NUM = performance_metrics(y_preds, test_info_path)
    print("\033[034mINFO: p: %s, r: %s, f1: %s\033[0m"%(p, r, f1))
    
    pdata = pd.read_csv(test_path, sep=' <sep> ', engine='python', names=names)
    sentences = pdata['text'][:].values
    # for pred, target in zip(y_preds,y_targets):
    #     x=len
    #     for val in pred:
    #         if val not in ['B-KP','I-KP','O']:
    #             print(pred)
    #     for val in target:
    #         if val not in ['B-KP','I-KP','O']:
    #             print(target)
    # print(sentences[0])
    # print(y_preds[0])
    # print(y_targets[0])
    get_metrics(sentences, y_preds, y_targets, mappers=True)
    
    print('SeqEval Metrics:')
    print(classification_report(y_targets, y_preds, mode='strict', scheme=IOB2))
    print("Precision given by SeqEval: {:.2f}%".format(precision_score(y_targets, y_preds)*100))
    print("Recall given by SeqEval: {:.2f}%".format(recall_score(y_targets, y_preds)*100))
    print("F1-Score given by SeqEval: {:.2f}%".format(f1_score(y_targets, y_preds)*100))
    print("Accuracy given by SeqEval: {:.2f}%".format(accuracy_score(y_targets, y_preds)*100))

    # save result
    tfs_type = "".join([i[0] for i in texts_features]).upper()
    save_path = os.path.join(RESULT_FOLDER, "%s_%s_%s.txt"%(data_type, corpus_type, tfs_type))
    with open(save_path, 'w', encoding='utf-8') as fp:
        fp.write("%s\t%s\t%s"%(p, r, f1))
    return p, r, f1

# Model testgs predict function
def testgs(file_folder, file_name, corpus_type, data_type, texts_features=None):
    # File path
    testgs_path = os.path.join(file_folder, '%s.txt'%data_type)
    predict_info_path = os.path.join(file_folder, '%s_info.json'%data_type)
    vocab_path = os.path.join(file_folder, 'vocab')
    # Load configuration file
    config_path = os.path.join(file_folder, "config.json")
    configs = json.load(open(config_path, 'r', encoding='utf-8'))
    print("\033[034mPARAMS: %s\033[0m"%configs)
    ix2tag = {i:key for i, key in enumerate(
            pd.read_csv(configs['tags_path'], header=None, sep='\n').values.flatten().tolist())}
    ix2tag[len(ix2tag)] = 'O'
    ix2tag[len(ix2tag)] = 'O'
    test_dataset = TextDataSet(data_path = testgs_path,
                    texts_features=texts_features, vocab_path = vocab_path, configs = configs)
    # Data Loader
    test_loader = DataLoader(test_dataset, batch_size = configs['batch_size'])
    # Create BiLSTM-CRF model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_CRF(vocab_path = vocab_path, configs = configs,
                       features=texts_features, device=device).to(device)
    print(model)
    # Loading the pre-trained
    tfs_type = "".join([i[0] for i in texts_features]).upper()
    model_path = os.path.join(MODEL_FOLDER, 'bin', "%s_%s.bin" % (corpus_type, tfs_type))
    model_config = torch.load(open(model_path, 'rb'))
    model.load_state_dict(model_config['model'])
    model.eval()
    y_preds, y_targets = [], []
    with tqdm(test_loader) as pbar_test:
        for words2ix, chars2ix, labels, attention_mask, features, sentence_len in pbar_test:
            inputs = {
                'words2ix': torch.as_tensor(words2ix, dtype=torch.long).to(device),
                'chars2ix': torch.as_tensor(chars2ix, dtype=torch.long).to(device),
                'attention_mask': torch.as_tensor(attention_mask).to(device),
                'features': torch.as_tensor(features, dtype=torch.float).to(device),
            }
            _, outputs = model(inputs)
            for ix, (pred, tag) in enumerate(zip(outputs, labels)):
                y_preds.append([ix2tag[i] for i in pred.detach().cpu().tolist()[: sentence_len[ix]]])
                y_targets.append([ix2tag[i] for i in tag.detach().cpu().tolist()[: sentence_len[ix]]])
            pbar_test.set_description("finish")
    
    pdata = pd.read_csv(testgs_path, sep=' <sep> ', engine='python', names=names)
    target_kp = []
    # for sent,pred,label in zip(pdata['text'],y_preds,y_targets):
    #     print("Actual Sentence: ", sent)
    #     # sent_len = len(pred)
    
    complete_text = '. '.join(pdata['text'][:].values)
    sentences = pdata['text'][:].values
    
    pred_kws = parse_tags(sentences, y_preds)
    pred_kws.sort(reverse=True, key=len)
    # print(pred_kws, end='\n\n')
        
    # markings = mark_keyword_all_sentences(pred_kws,sentences)
    # complt_text = markings[0]
    # mark_kws = markings[1]
    # mark_tokens = markings[2]
    # # print(complt_text)
    
    # for sent, mark, token, pred, targ in zip(sentences, mark_kws, mark_tokens,y_preds,y_targets):
    #     if mark!=pred:
    #         print('Sentence: ',sent,'\nTokens:',token, '\nPrep.Func. Markings:',mark,'\nKP Predictions',pred, '\nGS Markings',targ ,end='\n\n')
    
    # print("Complete Text:\n",complete_text,end='\n\n')
    # print(pred_kws, end='\n\n')
    
    get_metrics(sentences, y_preds, y_targets, mappers=True)
    
    print('SeqEval Metrics:')
    print(classification_report(y_targets, y_preds, mode='strict', scheme=IOB2))
    print("Precision given by SeqEval: {:.2f}%".format(precision_score(y_targets, y_preds)*100))
    print("Recall given by SeqEval: {:.2f}%".format(recall_score(y_targets, y_preds)*100))
    print("F1-Score given by SeqEval: {:.2f}%".format(f1_score(y_targets, y_preds)*100))
    print("Accuracy given by SeqEval: {:.2f}%".format(accuracy_score(y_targets, y_preds)*100))
    
    # print('CRF Suite Classification Report:')
    # labels=['B-KP','I-KP','O']
    # labels.remove('O')
    # sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
    # print(crfmetrics.flat_classification_report(y_targets, y_preds, sorted_labels))
    
    # for i,j in zip(y_preds,y_targets):
    #     print(i,j,sep='\n',end='\n\n')
    
    # print("Text:\n", all_sentences, end='\n\n')

    #     i=0
    #     pred_words = []
    #     while i<sent_len:
    #         if pred[i]=='O':
    #             i+=1
    #             continue

    #         if pred[i]=='K_B':
    #             pred_words.append([pred[i]])
    #         else:
    #             pred_words[-1].extend(pred[i])
            
    #         i+=1
            
    # p, r, f1, NUM = performance_metrics(y_preds, predict_info_path)
    # print("\033[034mINFO: p: %s, r: %s, f1: %s\033[0m"%(p, r, f1))
    # save result
    # tfs_type = "".join([i[0] for i in texts_features]).upper()
    # save_path = os.path.join(RESULT_FOLDER, "%s_%s_%s.txt"%(data_type, corpus_type, tfs_type))
    # with open(save_path, 'w', encoding='utf-8') as fp:
    #     fp.write("%s\t%s\t%s"%(p, r, f1))
    # return p, r, f1

# Model Infer KWs abstract by abstract
def predict_kws(file_folder, corpus_type, data_type, texts_features=None):
    # File path
    predict_path = os.path.join(file_folder, '%s.txt'%data_type)
    predict_info_path = os.path.join(file_folder, '%s_info.json'%data_type)
    vocab_path = os.path.join(file_folder, 'vocab')
    # Load configuration file
    config_path = os.path.join(file_folder, "config.json")
    configs = json.load(open(config_path, 'r', encoding='utf-8'))

    ix2tag = {i:key for i, key in enumerate(
            pd.read_csv(configs['tags_path'], header=None, sep='\n').values.flatten().tolist())}
    ix2tag[len(ix2tag)] = 'O'
    ix2tag[len(ix2tag)] = 'O'
    test_dataset = TextDataSet(data_path = predict_path,
                    texts_features=texts_features, vocab_path = vocab_path, configs = configs)
    # Data Loader
    test_loader = DataLoader(test_dataset, batch_size = configs['batch_size'])
    # Create BiLSTM-CRF model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_CRF(vocab_path = vocab_path, configs = configs,
                       features=texts_features, device=device).to(device)
    # Loading the pre-trained
    tfs_type = "".join([i[0] for i in texts_features]).upper()
    model_path = os.path.join(MODEL_FOLDER, 'bin', "%s_%s.bin" % (corpus_type, tfs_type))
    model_config = torch.load(open(model_path, 'rb'))
    model.load_state_dict(model_config['model'])
    model.eval()
    y_preds, y_targets = [], []
    with tqdm(test_loader) as pbar_test:
        for words2ix, chars2ix, labels, attention_mask, features, sentence_len in pbar_test:
            inputs = {
                'words2ix': torch.as_tensor(words2ix, dtype=torch.long).to(device),
                'chars2ix': torch.as_tensor(chars2ix, dtype=torch.long).to(device),
                'attention_mask': torch.as_tensor(attention_mask).to(device),
                'features': torch.as_tensor(features, dtype=torch.float).to(device),
            }
            _, outputs = model(inputs)
            for ix, (pred, tag) in enumerate(zip(outputs, labels)):
                y_preds.append([ix2tag[i] for i in pred.detach().cpu().tolist()[: sentence_len[ix]]])
                y_targets.append([ix2tag[i] for i in tag.detach().cpu().tolist()[: sentence_len[ix]]])
            pbar_test.set_description("finish")
    
    pdata = pd.read_csv(predict_path, sep=' <sep> ', engine='python', names=names)
    target_kp = []
    # for sent,pred,label in zip(pdata['text'],y_preds,y_targets):
    #     print("Actual Sentence: ", sent)
    #     # sent_len = len(pred)
    
    complete_text = '. '.join(pdata['text'][:].values)
    sentences = pdata['text'][:].values
    
    pred_kws = parse_tags(sentences, y_preds)
    pred_kws.sort(reverse=True, key=len)
    return pred_kws
    # print(pred_kws, end='\n\n')
        
    # markings = mark_keyword_all_sentences(pred_kws,sentences)
    # complt_text = markings[0]
    # mark_kws = markings[1]
    # mark_tokens = markings[2]
    # # print(complt_text)
    
    # for sent, mark, token, pred, targ in zip(sentences, mark_kws, mark_tokens,y_preds,y_targets):
    #     if mark!=pred:
    #         print('Sentence: ',sent,'\nTokens:',token, '\nPrep.Func. Markings:',mark,'\nKP Predictions',pred, '\nGS Markings',targ ,end='\n\n')

 