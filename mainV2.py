#################################
## Author: Rohan Goli
## Usage: Document-level attention by Hierarchical Attention BiLSTM-CRF
## Date: 09/21/22
#################################

import argparse
import configV2 as config
import os
import shutil
import time
import sys
import json

from collections import defaultdict

import gensim
from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split
import random
import csv

from modules import utils
from modules import createDataset as data
from modules import preProcessData as ppd

from modules.languageModel import LanguageModel, WordLSTM_v2, Trainer, EmbeddingsReader

## For Hierarchical-Attention based Sent-Level Bi-LSTM-CRF
from modules.hierAttnNtwk import BiLSTM_CRF, hierAttnNtwkTrainer

## For Naive Bi-LSTM-CRF
# from modules.hierAttnNtwk import hierAttnNtwkTrainer
# from modules.biLSTMCRF import BiLSTM_CRF

import numpy as np

from GPUtil import showUtilization as gpu_usage

import torch
torch.manual_seed(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the absolute path under the current working directory
WORKING_PATH = os.getcwd()
BI_LM_MODEL_FILE = "BiLM.pt"
WORD_EMBED_FILE = 'LM_word2vec.bin'
BILSTMCRF_MODEL_NAME = 'hierAttn-BiLSMT-CRF'
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300
DROPOUT = 0.2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyphrase extraction using hierAttn-BiLSTM-CRF")

    parser.add_argument('--mode','-m', default='train', type=str, help='Mode of opertion: train/test/testgs/predict/createData/preTrain')
    parser.add_argument('--sections','-s', default=['title','abstract'], type=str, nargs='+', help='Sections in research paper to include: title/abstract/methods/results')
    parser.add_argument('--features','-f', default=['CEM','WFOF','POS','LEN','TI','TR'], type=str, nargs='+',
                        help=("Features to include:",
                        "CEM - Char Embeddings",
                        "DEM - Document Embeddings",
                        "PEM - Phrase Embeddings",
                        "TI - TFIDF(Term Frequency Inverse Document Frequency)",
                        "TR - Text Rank",
                        "TopR - Topic Rank",
                        "PosR - Position Rank "
                        "MR - Multi-partite Rank",
                        "WFOT - Word First Occurence in Text",
                        "LEN - Length of word",
                        ))
    parser.add_argument("--stem", "-stem",default="no",help="Vocabulary Stemming - yes/no")
    parser.add_argument("--minSentences", "-minSentences", default=3, help="CreateSynthDS :::: Minimum Sentences in Each Text")
    parser.add_argument("--filterKP","-filterKP", default=False, help="CreateSynthDS :::: Filter KP generation for DISEASE/GPE/HEALTH/MEDICATION")
    parser.add_argument("-seqLen","--seqLength", type=int, default = 5, help="BiLM :::: To create Train Dataset with sequence length sliding window over sentence. Deafult(5)")
    
    args = parser.parse_args()

    print("="*100)
    print("Mode: {}".format(args.mode))
    print("="*100)

    json_data_folder = os.path.join(WORKING_PATH, config.JSON_DATA)
    if not os.path.exists(json_data_folder): os.mkdir(json_data_folder)
    interim_data_folder = os.path.join(WORKING_PATH, config.INTERIM_DATA)
    if not os.path.exists(interim_data_folder): os.mkdir(interim_data_folder)

    if args.mode == "createData":
        ## Read PubMed XML processed Data - For all CDSS
        pubmed_data_folder = os.path.join(WORKING_PATH, config.PUBMED_PROCESSED_DATA)
        fullData = data.load_pubmed_data(pubmed_data_folder)

        pubmed_data_folder = os.path.join(WORKING_PATH, config.PUBMED_PROCESSED_DATA_GS)
        test42Abstracts = data.load_pubmed_data(pubmed_data_folder)

        ## Read Manual Formatted JSON ACM Data - 12
        miscData = utils.read_json_array(os.path.join(config.DATA_FOLDER,"ACM_MiscPubMed_Abstracts.json"))

        ## Generate JSON Files from XML
        data.generate_synthetic_dataset(
            fullData+test42Abstracts,
            json_data_folder,
            min_sentences=args.minSentences,
            filter_kp=args.filterKP,
            model_name=os.path.join(WORKING_PATH,"cdssSciSpacy/model-best"),
            additional_data = miscData
            )

        ## Annotated GS Labels
        GS_DATA_FILES = ["4Rohan_FirstAnnotationResults_GS_CleanedAfter.txt", "4Rohan_SecondTimeGS.txt", "4Rohan_ACM_GS.txt"]
        testGSAbs = []
        Labels = ["First-Time","Second-Time","Second-Time ACM"]
        for idx, datFile in enumerate(GS_DATA_FILES):
            with open(os.path.join(WORKING_PATH, config.DATA_FOLDER, datFile),'r') as f:
                testGSAbs.append(f.read().split('\n\n'))
            print("{} GS: {}".format(Labels[idx],len(testGSAbs[-1])))

        ## Read Synthetic Labeled Data
        trainData = utils.read_json_array(os.path.join(config.JSON_DATA, "train.json"))
        testData = utils.read_json_array(os.path.join(config.JSON_DATA, "test.json"))
        totalData = trainData + testData

        ## Read GS Labels - 42 GS
        gs42Abs = {}
        for abstract in testGSAbs[0]:
            temp = abstract.split('\n')
            gs42Abs[temp[0].split(':')[1]] = temp[1:]
        ## Read GS Labels - 91 GS
        gs91Abs = {}
        for abstract in testGSAbs[1]:
            temp = abstract.split('\n')
            gs91Abs[temp[0].split(':')[1]] = temp[1:]
        ## Read GS Labels - 8 ACM GS
        for abstract in testGSAbs[2]:
            temp = abstract.split('\n')
            gs91Abs[temp[0]] = temp[1:]        
        
        ## Remove GS Data from totalData
        ## 1. FULL DATA with Synthetic Labels
        ## 2. Replace Synthetic with GS
        count = 0
        gs91Data = []
        gs42Data = []
        visited = set()
        i=0
        j=0
        print("Total data length: ", len(totalData))
        while i<len(totalData):
            abstract = totalData[i]
            if abstract['id'] in gs91Abs and abstract['id'] not in visited:
                abstract['keywords'] = gs91Abs[abstract['id']]
                gs91Data.append(abstract)
                visited.add(abstract['id'])
                count+=1
            elif abstract['id'] in gs42Abs and abstract['id'] not in visited:
                abstract['keywords'] = gs42Abs[abstract['id']]
                gs42Data.append(abstract)
                visited.add(abstract['id'])
                count+=1
            else:
                totalData[i] = totalData[j]
                j+=1
            i+=1
        totalData[j:]=[]
        
        ## Write JSON Data
        print("Saving GS91: ", len(gs91Data))
        utils.save_json(gs91Data, os.path.join(config.JSON_DATA,"testgs91.json"))

        print("Saving GS42: ", len(gs42Data))
        utils.save_json(gs42Data, os.path.join(config.JSON_DATA,"testgs42.json"))

        dataset_length = len(totalData)
        print("Saving Train: ", dataset_length//3)
        utils.save_json(totalData[:dataset_length//3],os.path.join(config.JSON_DATA,"train.json"))

        print("Saving Test: ", dataset_length-dataset_length//3)
        utils.save_json(totalData[dataset_length//3:],os.path.join(config.JSON_DATA,"test.json"))
        pass

    elif args.mode == "preTrain":
        ###############
        ## Load Data ##
        ###############
        trainData = utils.read_json_array(os.path.join(config.JSON_DATA, "train.json"))
        testData = utils.read_json_array(os.path.join(config.JSON_DATA, "test.json"))

        ## Store All Sentences of Corpus
        train_corpus_sentences = []
        for x in trainData:
            train_corpus_sentences.extend(x['sentences'])
        print("Total Sentences in Train Corpus: ", len(train_corpus_sentences))
        test_corpus_sentences = []
        for x in testData:
            test_corpus_sentences.extend(x['sentences'])
        print("Total Sentences in Test Corpus: ", len(test_corpus_sentences))

        ## Total Data
        total_corpus_sentences = train_corpus_sentences+test_corpus_sentences

        ############################
        ## Create Word Embeddings ##
        ############################
        modelW2V = gensim.models.Word2Vec(
                    sentences=total_corpus_sentences,
                    vector_size=300, window=5,
                    min_count=0,workers=10,
                    sg=1,seed=42,compute_loss=True
                    )
        
        ## Save Word Embeddings
        modelW2V.wv.save_word2vec_format(os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE), binary=True)

        ## Load Word Embeddings
        modelW2V = KeyedVectors.load_word2vec_format(os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE), binary=True)

        ## Create Integer-to-Token mapping
        int2token = {0:'<UNK>'}
        cnt = 1
        for word in modelW2V.key_to_index.keys():
            int2token[cnt] = word
            cnt+=1
        ## Create Token-to-Integer mapping
        token2int = {t: i for i, t in int2token.items()}

        ## Store Vocab Size - For further reference
        vocab_size = len(int2token)
        print("Vocabulary Size: {}".format(vocab_size))

        ## Save Int2Token and Token2Int mappings
        utils.save_json(token2int, os.path.join(config.INTERIM_DATA,"token2int.json"))
        utils.save_json(int2token, os.path.join(config.INTERIM_DATA,"int2token.json"))

        ##########################################
        ## Create Bi-Directional Language Model ##
        ##########################################
        lang_model = LanguageModel(token2int=token2int)

        ## Create Data Input/Targets for BiLM
        def splitData_Input_Targets(csLM, corpus_sents, token2int):
            ## Create Sequences
            sequences = [csLM.create_seq(i, seq_len=args.seqLength) for i in corpus_sents]
            ## Merge list-of-lists into a single list
            sequences = sum(sequences, [])

            ## Split Data into Input/Target text
            inputs, targets = [], []
            for sequence in sequences:
                ## Split Sentence into Tokens
                seqTokens = sequence.split()
                ## Convert Text Sequences to Integer Sequences
                seqInts = csLM.get_integer_seq(seqTokens, tokenized=True)
                
                if len(seqInts) != args.seqLength+1:
                    continue
                
                inputs.append(seqInts[:-1])
                targets.append(seqInts[1:])
            return np.array(inputs, dtype="int32"),np.array(targets, dtype="int32")

        x_train, y_train = splitData_Input_Targets(lang_model, train_corpus_sentences, token2int)
        x_test, y_test = splitData_Input_Targets(lang_model, test_corpus_sentences, token2int)

        ## Initiate Bi-LM Network
        biLMNet = WordLSTM_v2(
            n_layers = 1,
            bidirection = True,
            pretrainedWE = os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE),
            vocab = token2int
            )

        ## Move Bi-LM Network to GPU
        biLMNet.cuda()
        print("Bi-Directional Language Model :")
        print(biLMNet)

        ## Initiate Trainer
        trainer = Trainer(
            token2int=token2int,
            int2token=int2token,
            X_train = x_train,
            y_train = y_train,
            X_test = x_test,
            y_test = y_test
            )

        ## Initiate BiLM Training
        trainer.train(
            biLMNet,
            batch_size = 64,
            epochs=20
            )

        ## Save BiLM Model
        trainer.saveModel(
            biLMNet,
            os.path.join(config.INTERIM_DATA,BI_LM_MODEL_FILE)
        )

        print(trainer.sample(biLMNet, 45, prime = "clinical decision support systems"),"\n")
        print(trainer.sample(biLMNet, 45, prime = "prescription errors"),"\n")
        print(trainer.sample(biLMNet, 45, prime = "one of the"),"\n")
        print(trainer.sample(biLMNet, 45, prime = "asthma management programs for primary care providers increasing adherence to asthma guidelines"),"\n")
        
        # (pytorch) [rgoli@node1397 NLP_KPIdentify]$ python mainV2.py -m preTrain
        # Word Embedding Type:  word2vec
        # ====================================================================================================
        # Mode: preTrain
        # ====================================================================================================
        # Total Sentences in Train Corpus:  13603
        # Total Sentences in Test Corpus:  22766
        # Vocabulary Size: 5043
        # /home/rgoli/.local/lib/python3.8/site-packages/torch/nn/modules/rnn.py:62: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.3 and num_layers=1
        #   warnings.warn("dropout option adds dropout after all but last "
        # Bi-Directional Language Model :
        # WordLSTM_v2(
        #   (emb_layer): Embedding(5043, 300)
        #   (lstm): LSTM(300, 128, batch_first=True, dropout=0.3, bidirectional=True)
        #   (dropout): Dropout(p=0.3, inplace=False)
        #   (fc): Linear(in_features=256, out_features=5043, bias=True)
        # )
        # Epoch: 1/20... mean_loss : 0.88, Perplexity : 2.41
        # Epoch: 2/20... mean_loss : 0.23, Perplexity : 1.26
        # Epoch: 3/20... mean_loss : 0.15, Perplexity : 1.17
        # Epoch: 4/20... mean_loss : 0.11, Perplexity : 1.11
        # Epoch: 5/20... mean_loss : 0.07, Perplexity : 1.08
        # Epoch: 6/20... mean_loss : 0.05, Perplexity : 1.06
        # Epoch: 7/20... mean_loss : 0.04, Perplexity : 1.04
        # Epoch: 8/20... mean_loss : 0.03, Perplexity : 1.03
        # Epoch: 9/20... mean_loss : 0.03, Perplexity : 1.03
        # Epoch: 10/20... mean_loss : 0.02, Perplexity : 1.02
        # Epoch: 11/20... mean_loss : 0.02, Perplexity : 1.02
        # Epoch: 12/20... mean_loss : 0.02, Perplexity : 1.02
        # Epoch: 13/20... mean_loss : 0.02, Perplexity : 1.02
        # Epoch: 14/20... mean_loss : 0.01, Perplexity : 1.02
        # Epoch: 15/20... mean_loss : 0.01, Perplexity : 1.01
        # Epoch: 16/20... mean_loss : 0.01, Perplexity : 1.01
        # Epoch: 17/20... mean_loss : 0.01, Perplexity : 1.01
        # Epoch: 18/20... mean_loss : 0.01, Perplexity : 1.01
        # Epoch: 19/20... mean_loss : 0.01, Perplexity : 1.01
        # Epoch: 20/20... mean_loss : 0.01, Perplexity : 1.01
        # Test Perplexity: 31.14
        # clinical decision support systems to healthcare smith data sources of the evidence the evidence and evidence statements evidence statements pertaining statements and pertaining to medication events addressing the medication profiles which were presented to phealth phealth phealth health questionnaire pssuq network ssn network sensor data transfer tasks tasks e.g. 

        # prescription errors of review of evidence and current guidance on safe for medication use as an ehr an electronic medical electronic medical records medical record ehr medical ehr data is known for their use of the studies studies appeared could get a individualized therapy therapy and therapy 

        # one of the gl-dss of gl-dss of multiple than when than than 48 between 2013 between groups considered the interaction probability prespecification analysis is vital values increased in increased to increased to increased in children with minor head trauma and mht among among people among people can sustainably 

        # asthma management programs for primary care providers increasing adherence to asthma guidelines designed in need to our services to detect potential drug-drug potential to potential drug-drug interactions or discussed are currently represented without without external without any without ct without ct scan ct scan results no difference in terms between stakeholders and often and 3 and often
        
        pass

    elif args.mode == "train":
        trainData = utils.read_json_array(os.path.join(config.JSON_DATA, "train.json"))
        testData = utils.read_json_array(os.path.join(config.JSON_DATA, "test.json"))

        # ## Load Word Embeddings
        # modelW2V = KeyedVectors.load_word2vec_format(os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE), binary=True)

        # ## Create Integer-to-Token mapping
        # ix_to_word = {0:'<UNK>'}
        # cnt = 1
        # for word in modelW2V.key_to_index.keys():
        #     ix_to_word[cnt] = word
        #     cnt+=1
        # ## Create Token-to-Integer mapping
        # word_to_ix = {t: i for i, t in ix_to_word.items()}

        ix_to_word = utils.read_json_array(os.path.join(config.INTERIM_DATA,"int2token.json"))
        word_to_ix = utils.read_json_array(os.path.join(config.INTERIM_DATA,"token2int.json"))

        ## Store Vocab Size - For further reference
        vocab_size = len(ix_to_word)
        print("Vocabulary Size: {}".format(vocab_size))

        ## Load Word Embeddings Binary
        word_embeddings = EmbeddingsReader.from_binary(
            filename = os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE),
            vocab = word_to_ix
            )

        tag_to_ix = {"O": 0, "B-KP": 1, "I-KP": 2, START_TAG: 3, STOP_TAG: 4}
        ix_to_tag = {0:"O", 1: "B-KP", 2: "I-KP", 3: "O", 4: "O"}

        biLSTM_CRF_model = BiLSTM_CRF(
            vocab_size = len(word_to_ix),
            tag_to_ix = tag_to_ix,
            embedding_dim = EMBEDDING_DIM,
            dropout = DROPOUT,
            use_pretrained_embd=word_embeddings
            )
        
        hierAttnNtwk_trainer = hierAttnNtwkTrainer(
            model = biLSTM_CRF_model,
            train_data = trainData,
            test_data = testData,
            biLM_path = os.path.join(config.INTERIM_DATA,BI_LM_MODEL_FILE),
            word_to_ix = word_to_ix,
            tag_to_ix = tag_to_ix,
            ix_to_tag = ix_to_tag
        )
        ## Start Training
        hierAttnNtwk_trainer.train()
        ## Save Best Model
        hierAttnNtwk_trainer.saveModel(os.path.join(config.INTERIM_DATA, BILSTMCRF_MODEL_NAME))

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Start hierAttnNtwk-Bi-LSTM-CRF Train >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Epoch: 2/30 Loss: 4.761
        # Epoch: 4/30 Loss: 2.215
        # Epoch: 6/30 Loss: 1.512
        # Epoch: 8/30 Loss: 1.063
        # Epoch: 10/30 Loss: 0.900
        # Epoch: 12/30 Loss: 0.739
        # Epoch: 14/30 Loss: 0.670
        # Epoch: 16/30 Loss: 0.613
        # Epoch: 18/30 Loss: 0.534
        # Epoch: 20/30 Loss: 0.432
        # Epoch: 22/30 Loss: 0.413
        # Epoch: 24/30 Loss: 0.374
        # Epoch: 26/30 Loss: 0.339
        # Epoch: 28/30 Loss: 0.308
        # Epoch: 30/30 Loss: 0.279
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END hierAttnNtwk-Bi-LSTM-CRF Train >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START hierAttnNtwk-Bi-LSTM-CRF Test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Test Loss: 10.339
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END hierAttnNtwk-Bi-LSTM-CRF Test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # (pytorch) [rgoli@node1118 NLP_KPIdentify]$         
        pass

    elif args.mode == "test":
        testData = utils.read_json_array(os.path.join(config.JSON_DATA, "test.json"))
        
        ## Read Vocabulary Mapping
        ix_to_word = utils.read_json_array(os.path.join(config.INTERIM_DATA,"int2token.json"))
        word_to_ix = utils.read_json_array(os.path.join(config.INTERIM_DATA,"token2int.json"))

        ## Store Vocab Size - For further reference
        vocab_size = len(ix_to_word)
        print("Vocabulary Size: {}".format(vocab_size))

        ## Load Word Embeddings Binary
        word_embeddings = EmbeddingsReader.from_binary(
            filename = os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE),
            vocab = word_to_ix
            )

        tag_to_ix = {"O": 0, "B-KP": 1, "I-KP": 2, START_TAG: 3, STOP_TAG: 4}
        ix_to_tag = {0:"O", 1: "B-KP", 2: "I-KP", 3: "O", 4: "O"}

        ## Initialize Model Architecture with Word Embeddings, 
        biLSTM_CRF_model = BiLSTM_CRF(
            vocab_size = len(word_to_ix),
            tag_to_ix = tag_to_ix,
            embedding_dim = EMBEDDING_DIM,
            dropout = DROPOUT,
            use_pretrained_embd=word_embeddings
            )
        
        ## Initialize Model-Trainer with Model Architecture, Word Mappings
        hierAttnNtwk_trainer = hierAttnNtwkTrainer(
            model = biLSTM_CRF_model,
            train_data = None,
            test_data = None,
            biLM_path = os.path.join(config.INTERIM_DATA,BI_LM_MODEL_FILE),
            word_to_ix = word_to_ix,
            tag_to_ix = tag_to_ix,
            ix_to_tag = ix_to_tag
        )

        ## Load Best Model
        hierAttnNtwk_trainer.loadModel(os.path.join(config.INTERIM_DATA, BILSTMCRF_MODEL_NAME))

        ## Get Metrics on Test Data
        hierAttnNtwk_trainer.getMetricsOnData(testData)
        # (pytorch) [rgoli@node1118 NLP_KPIdentify]$ python mainV2.py -m test
        # Word Embedding Type:  word2vec
        # ====================================================================================================
        # Mode: test
        # ====================================================================================================
        # Vocabulary Size: 5043
        # Using pretrained word embeddings
        # /home/rgoli/.local/lib/python3.8/site-packages/torch/nn/modules/rnn.py:62: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
        #   warnings.warn("dropout option adds dropout after all but last "
        # Test Keyword Marking:
        # 100%|████████████████████████████████████████████████████████████████████| 2099/2099 [00:15<00:00, 132.40it/s]
        # Metrics on Test DS:
        # Confusion Matrix:
        #  [[ 39402.    884.  15876.]
        #  [  1197.  13285.   4920.]
        #  [ 10196.   2577. 379908.]]
        # --------------------------------------------------
        # Label: B-KP
        # TP:39402.0 FN:16760.0 FP:11393.0 TN:400690.0
        # --------------------------------------------------
        #     Accuracy:   0.939875
        #     Misclassification:  0.060125
        #     Precision:  0.775706
        #     Sensitivity/Recall: 0.701578
        #     Specificity:        0.972353
        #     F1 Score:   0.736782
        # --------------------------------------------------
        # Label: I-KP
        # TP:13285.0 FN:6117.0 FP:3461.0 TN:445382.0
        # --------------------------------------------------
        #     Accuracy:   0.979545
        #     Misclassification:  0.020455
        #     Precision:  0.793324
        #     Sensitivity/Recall: 0.684723
        #     Specificity:        0.992289
        #     F1 Score:   0.735034
        # --------------------------------------------------
        # Label: O
        # TP:379908.0 FN:12773.0 FP:20796.0 TN:54768.0
        # --------------------------------------------------
        #     Accuracy:   0.928309
        #     Misclassification:  0.071691
        #     Precision:  0.948101
        #     Sensitivity/Recall: 0.967472
        #     Specificity:        0.724790
        #     F1 Score:   0.957689
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Avg. Accuracy: 0.94924
        # Avg. Precision: 0.839044
        # Avg. F1-Score: 0.809835
        # Avg. Sensitivity: 0.784591
        # Avg. Specificity: 0.896477
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # SeqEval Metrics:
        #               precision    recall  f1-score   support

        #           KP       0.75      0.68      0.71     56162

        #    micro avg       0.75      0.68      0.71     56162
        #    macro avg       0.75      0.68      0.71     56162
        # weighted avg       0.75      0.68      0.71     56162

        # Precision given by SeqEval: 74.77%
        # Recall given by SeqEval: 67.66%
        # F1-Score given by SeqEval: 71.04%
        # Accuracy given by SeqEval: 92.39%
        pass

    elif args.mode == "testGS":
        ## Read Vocabulary Mapping
        ix_to_word = utils.read_json_array(os.path.join(config.INTERIM_DATA,"int2token.json"))
        word_to_ix = utils.read_json_array(os.path.join(config.INTERIM_DATA,"token2int.json"))

        ## Store Vocab Size - For further reference
        vocab_size = len(ix_to_word)
        print("Vocabulary Size: {}".format(vocab_size))

        ## Load Word Embeddings Binary
        word_embeddings = EmbeddingsReader.from_binary(
            filename = os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE),
            vocab = word_to_ix
            )

        tag_to_ix = {"O": 0, "B-KP": 1, "I-KP": 2, START_TAG: 3, STOP_TAG: 4}
        ix_to_tag = {0:"O", 1: "B-KP", 2: "I-KP", 3: "O", 4: "O"}

        ## Initialize Model Architecture with Word Embeddings, 
        biLSTM_CRF_model = BiLSTM_CRF(
            vocab_size = len(word_to_ix),
            tag_to_ix = tag_to_ix,
            embedding_dim = EMBEDDING_DIM,
            dropout = DROPOUT,
            use_pretrained_embd=word_embeddings
            )
        
        ## Initialize Model-Trainer with Model Architecture, Word Mappings
        hierAttnNtwk_trainer = hierAttnNtwkTrainer(
            model = biLSTM_CRF_model,
            train_data = None,
            test_data = None,
            biLM_path = os.path.join(config.INTERIM_DATA,BI_LM_MODEL_FILE),
            word_to_ix = word_to_ix,
            tag_to_ix = tag_to_ix,
            ix_to_tag = ix_to_tag
        )

        ## Load Best Model
        hierAttnNtwk_trainer.loadModel(os.path.join(config.INTERIM_DATA, BILSTMCRF_MODEL_NAME))

        ## Get Metrics on Test Data - GS42
        testGS42Data = utils.read_json_array(os.path.join(config.JSON_DATA, "testgs42.json"))
        hierAttnNtwk_trainer.getMetricsOnData(testGS42Data, "Test GS42")

        ## Get Metrics on Test Data - GS91
        testGS91Data = utils.read_json_array(os.path.join(config.JSON_DATA, "testgs91.json"))
        hierAttnNtwk_trainer.getMetricsOnData(testGS91Data, "Test GS91")
        
        # (pytorch) [rgoli@node1118 NLP_KPIdentify]$ python mainV2.py -m testGS
        # Word Embedding Type:  word2vec
        # ====================================================================================================
        # Mode: testGS
        # ====================================================================================================
        # Vocabulary Size: 5043
        # Using pretrained word embeddings
        # /home/rgoli/.local/lib/python3.8/site-packages/torch/nn/modules/rnn.py:62: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
        #   warnings.warn("dropout option adds dropout after all but last "
        # Test GS42 Keyword Marking:
        # 100%|███████████████████████████████████████████████████████████████| 42/42 [00:00<00:00, 105.31it/s]
        # Metrics on Test GS42 DS:
        # Confusion Matrix:
        #  [[ 567.   18.  463.]
        #  [  46.  230.  174.]
        #  [ 267.   63. 6665.]]
        # --------------------------------------------------
        # Label: B-KP
        # TP:567.0 FN:481.0 FP:313.0 TN:7132.0
        # --------------------------------------------------
        #     Accuracy:   0.906511
        #     Misclassification:  0.093489
        #     Precision:  0.644318
        #     Sensitivity/Recall: 0.541031
        #     Specificity:        0.957958
        #     F1 Score:   0.588174
        # --------------------------------------------------
        # Label: I-KP
        # TP:230.0 FN:220.0 FP:81.0 TN:7962.0
        # --------------------------------------------------
        #     Accuracy:   0.964559
        #     Misclassification:  0.035441
        #     Precision:  0.739550
        #     Sensitivity/Recall: 0.511111
        #     Specificity:        0.989929
        #     F1 Score:   0.604468
        # --------------------------------------------------
        # Label: O
        # TP:6665.0 FN:330.0 FP:637.0 TN:861.0
        # --------------------------------------------------
        #     Accuracy:   0.886142
        #     Misclassification:  0.113858
        #     Precision:  0.912764
        #     Sensitivity/Recall: 0.952823
        #     Specificity:        0.574766
        #     F1 Score:   0.932363
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Avg. Accuracy: 0.91907
        # Avg. Precision: 0.765544
        # Avg. F1-Score: 0.708335
        # Avg. Sensitivity: 0.668322
        # Avg. Specificity: 0.840885
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # SeqEval Metrics:
        #               precision    recall  f1-score   support

        #           KP       0.60      0.50      0.54      1048

        #    micro avg       0.60      0.50      0.54      1048
        #    macro avg       0.60      0.50      0.54      1048
        # weighted avg       0.60      0.50      0.54      1048

        # Precision given by SeqEval: 59.59%
        # Recall given by SeqEval: 50.10%
        # F1-Score given by SeqEval: 54.43%
        # Accuracy given by SeqEval: 87.86%
        # Test GS91 Keyword Marking:
        # 100%|███████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 147.61it/s]
        # Metrics on Test GS91 DS:
        # Confusion Matrix:
        #  [[ 1282.    62.  1042.]
        #  [   91.   419.   303.]
        #  [  601.   165. 15518.]]
        # --------------------------------------------------
        # Label: B-KP
        # TP:1282.0 FN:1104.0 FP:692.0 TN:16405.0
        # --------------------------------------------------
        #     Accuracy:   0.907817
        #     Misclassification:  0.092183
        #     Precision:  0.649443
        #     Sensitivity/Recall: 0.537301
        #     Specificity:        0.959525
        #     F1 Score:   0.588073
        # --------------------------------------------------
        # Label: I-KP
        # TP:419.0 FN:394.0 FP:227.0 TN:18443.0
        # --------------------------------------------------
        #     Accuracy:   0.968126
        #     Misclassification:  0.031874
        #     Precision:  0.648607
        #     Sensitivity/Recall: 0.515375
        #     Specificity:        0.987841
        #     F1 Score:   0.574366
        # --------------------------------------------------
        # Label: O
        # TP:15518.0 FN:766.0 FP:1345.0 TN:1854.0
        # --------------------------------------------------
        #     Accuracy:   0.891649
        #     Misclassification:  0.108351
        #     Precision:  0.920240
        #     Sensitivity/Recall: 0.952960
        #     Specificity:        0.579556
        #     F1 Score:   0.936314
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Avg. Accuracy: 0.92253
        # Avg. Precision: 0.739430
        # Avg. F1-Score: 0.699584
        # Avg. Sensitivity: 0.668545
        # Avg. Specificity: 0.842308
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # SeqEval Metrics:
        #               precision    recall  f1-score   support

        #           KP       0.61      0.50      0.55      2386

        #    micro avg       0.61      0.50      0.55      2386
        #    macro avg       0.61      0.50      0.55      2386
        # weighted avg       0.61      0.50      0.55      2386

        # Precision given by SeqEval: 60.68%
        # Recall given by SeqEval: 50.25%
        # F1-Score given by SeqEval: 54.97%
        # Accuracy given by SeqEval: 88.38%
        pass
    
    elif args.mode == "expMixDS":
        trainData = utils.read_json_array(os.path.join(config.JSON_DATA, "train.json"))
        testData = utils.read_json_array(os.path.join(config.JSON_DATA, "test.json"))
        testGS42Data = utils.read_json_array(os.path.join(config.JSON_DATA, "testgs42.json"))
        testGS91Data = utils.read_json_array(os.path.join(config.JSON_DATA, "testgs91.json"))
        
        ix_to_word = utils.read_json_array(os.path.join(config.INTERIM_DATA,"int2token.json"))
        word_to_ix = utils.read_json_array(os.path.join(config.INTERIM_DATA,"token2int.json"))

        ## Store Vocab Size - For further reference
        vocab_size = len(ix_to_word)
        print("Vocabulary Size: {}".format(vocab_size))

        tag_to_ix = {"O": 0, "B-KP": 1, "I-KP": 2, START_TAG: 3, STOP_TAG: 4}
        ix_to_tag = {0:"O", 1: "B-KP", 2: "I-KP", 3: "O", 4: "O"}

        with open('expMixDS.csv', 'w', newline='') as csvfile:
            logwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            logwriter.writerow(['Exp. No.','Combination','Train DS Length','Test DataSet','Test DS Length','Precision','Recall','F1Score','Accuracy'])

        # experiment_mix_combs = [(100, 0), (100, 3), (100, 6), (100, 12), (100, 24), (100, 36), (100, 48)]
        # experiment_mix_combs = [(100, 0)]
        experiment_mix_combs = [(100, 0), (100, 2), (100, 4), (100, 6), (100, 8), (100, 10), (100, 12)]
        for tBt, gsBt in experiment_mix_combs:
            ## Number of times to Repeat Experiment
            repeat = 10
            for seed in range(repeat):
                print("Combination: {} Train : {} GS, Experiment - #{}".format(tBt,gsBt,seed))
                ## Create Randomized Synthetic Train/Test Data for each experiment
                exp_trainData, exp_testData = train_test_split(trainData+testData, test_size=0.67, random_state=seed)
                
                ## Non-Shuffled Data Set with Train(begin), Test(Mid+End Blocks) 1:2 Train-Test Ratio Split
                # exp_trainData, exp_testData = trainData, testData
                
                ## Non-Shuffled Data Set with Train(End), Test(Begin+Mid Blocks) 1:2 Train-Test Ratio Split
                # totalData = trainData+testData
                # exp_trainData, exp_testData = totalData[2098:], totalData[:2098]
                
                ## Non-Shuffled Data Set with Train(Mid), Test(Begin+End Blocks) 1:2 Train-Test Ratio Split
                # exp_trainData, exp_testData = totalData[1049:2098], totalData[:1049]+totalData[2098:0]
                
                ## Shuffle GS 42, GS 91 
                exp_gs42Data, exp_gs91Data = train_test_split(testGS42Data+ testGS91Data, test_size=0.69, random_state=seed)

                i=j=0
                exp_MixData = []
                while i<len(exp_trainData):
                    while i%tBt==0 and j<gsBt:
                        exp_MixData.append(random.choice(exp_gs42Data))
                        j+=1
                    j=0
                    exp_MixData.append(exp_trainData[i])
                    i+=1

                ds_length = len(exp_MixData)
                # print(len(exp_trainData),len(exp_gs42Data))
                # print(len(exp_MixData))
                # print([data['id'] for data in exp_MixData[:20]])

                ## Load Word Embeddings Binary
                word_embeddings = EmbeddingsReader.from_binary(
                    filename = os.path.join(config.INTERIM_DATA,WORD_EMBED_FILE),
                    vocab = word_to_ix
                    )
                
                ## Initialize new model
                biLSTM_CRF_model = BiLSTM_CRF(
                    vocab_size = len(word_to_ix),
                    tag_to_ix = tag_to_ix,
                    embedding_dim = EMBEDDING_DIM,
                    dropout = DROPOUT,
                    use_pretrained_embd=word_embeddings ## Load Pre-Trained Word Embeddings
                    )
                
                ## Intialize new trainer
                hierAttnNtwk_trainer = hierAttnNtwkTrainer(
                    model = biLSTM_CRF_model,
                    train_data = exp_MixData, ## Random Mix Data
                    test_data = exp_testData,
                    biLM_path = os.path.join(config.INTERIM_DATA,BI_LM_MODEL_FILE), ## Load Pre-trained Bi-Language Model
                    word_to_ix = word_to_ix,
                    tag_to_ix = tag_to_ix,
                    ix_to_tag = ix_to_tag
                )

                ## Start Random Truth Mix Training
                hierAttnNtwk_trainer.train()
                
                ## Get Metrics on Test Data
                test_p,test_r,test_f1,test_acc = hierAttnNtwk_trainer.getMetricsOnData(exp_testData)

                ## Get Metrics on Test Data - GS91
                gs_p,gs_r,gs_f1,gs_acc = hierAttnNtwk_trainer.getMetricsOnData(exp_gs91Data, "Test GS91")

                experiment_name = "Synth{}:GS{}".format(tBt,gsBt)

                with open('expMixDS.csv', 'a', newline='') as csvfile:
                    logwriter = csv.writer(csvfile, delimiter=';',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    logwriter.writerow([seed+1 , experiment_name, ds_length, 'Test', len(exp_testData), test_p,test_r,test_f1,test_acc])
                    logwriter.writerow([seed+1 , experiment_name, ds_length, 'GS91', len(exp_gs91Data), gs_p,gs_r,gs_f1,gs_acc])

                # print("GPU Utilization - After Training: ")
                # gpu_usage()
                del word_embeddings
                del biLSTM_CRF_model
                del hierAttnNtwk_trainer
                torch.cuda.empty_cache()
                #torch.cuda.reset()
                print("PyTorch Memory Allocation ::",torch.cuda.memory_allocated())
                # print("GPU Utilization - After Clearing Cache: ")
                # gpu_usage() 
                del exp_trainData, exp_testData, exp_gs42Data, exp_gs91Data, exp_MixData
            pass

    else:
        print("Invalid option selected!!!\n Mode of opertion: train/test/predict/createData")
