import argparse
import os
import shutil
import time
from modules import createDataset as data
from modules import preProcessData as ppd
from modules import model as model
from modules import utils
import configV2 as config
import logging

# Get the absolute path under the current working directory
WORKING_PATH = os.getcwd()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Keyphrase extraction using BiLSTM-CRF and AL")
    
    parser.add_argument('--mode','-m', default='train', type=str, help='Mode of opertion: train/test/testgs/predict/createData')
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
                        "WFOF - Word First Occurence in Text",
                        "LEN - Length of word",
                        ))
    parser.add_argument("--log", "-log",default="warn",
                        help=("Provide logging level. "
                              "Example --log debug', default='warning'"))
    parser.add_argument("--stem", "-stem",default="no",help="Vocabulary Stemming - yes/no")
    parser.add_argument("--minSentences", "-minSentences", default=3, help="Minimum Sentences in Each Text")
    parser.add_argument("--filterKP","-filterKP", default=False, help="Filter KP generation for DISEASE/GPE/HEALTH/MEDICATION")
    
    args = parser.parse_args()
    
    levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
    }
    
    level = levels.get(args.log.lower())
    if level is None:
        raise ValueError(
            f"log level given: {args.log}"
            f" -- must be one of: {' | '.join(levels.keys())}")
    logging.basicConfig(level=level)
    logger = logging.getLogger(__name__)
    
    if args.features == ['']:
        args.features = []

    print("="*100)
    print("Mode: {}".format(args.mode))
    print("Sections Included: {}".format(args.sections))
    print("Features: {}".format(args.features))
    print("Vocab Stemming: {}".format(args.stem))
    print("Logging Level: {}".format(args.log))
    print("="*100)
    
    
    json_data_folder = os.path.join(WORKING_PATH, config.JSON_DATA)
    if not os.path.exists(json_data_folder): os.mkdir(json_data_folder)
    interim_data_folder = os.path.join(WORKING_PATH, config.INTERIM_DATA)
    if not os.path.exists(interim_data_folder): os.mkdir(interim_data_folder)
        
    # Fields contained in the dataset
    corpus_type = "".join([i[0] for i in args.sections]).upper()

    if args.mode=="testgs42":
        s_time = time.time()
       
        ## Pre-Process JSON with Data Features
        ppd.build_testgs_data_set(
            json_data_folder, args.sections, interim_data_folder, args.stem,
            'testgs42.json')
        
        ## Load pre-trained model
        ## Infer/Predict on Test Gold Standard
        model.testgs(interim_data_folder, 'testgs42', corpus_type, args.mode, args.features)
        print("\033[031mINFO: Total time of model training:%sS\033[0m"%round(time.time() - s_time, 3))
        # (pytorch) [rgoli@node1118 NLP_KPIdentify]$ python main.py -m testgs42   ====================================================================================================
        # Mode: testgs42
        # Sections Included: ['title', 'abstract']
        # Features: ['CEM', 'WFOF', 'POS', 'LEN', 'TI', 'TR']
        # Vocab Stemming: no
        # Logging Level: warn
        # ====================================================================================================
        # 42-th document processed: 100%|███████████████████████████████████████████████████| 42/42 [00:01<00:00, 38.61it/s]
        # The 42 document calculation TF-IDF completed: 100%|█████████████████████████████| 42/42 [00:00<00:00, 9228.88it/s]
        # The 42 document has been calculated TextRank completed: 100%|████████████████████| 42/42 [00:00<00:00, 728.09it/s]
        # The 42-th document feature is completed: : 42it [00:00, 4513.59it/s]
        # INFO: testgs42 data saving completed!
        # PARAMS: {'word_embedding_dim': 128, 'char_embedding_dim': 32, 'word_max_len': 32, 'sentence_max_len': 256, 'hidden_dim': 128, 'use_crf': 'True', 'lstm_num_layers': 1, 'is_bidirectional': 'True', 'dropout': 0.5, 'tags_path': './interim_data/tags', 'epochs': 20, 'batch_size': 128, 'lr': 0.01, 'weight_decay': 0.0001, 'optim_scheduler_factor': 0.1, 'optim_scheduler_patience': 1, 'optim_scheduler_verbose': 'True'}
        # BiLSTM_CRF(
        #   (word_embedding): Embedding(4548, 128, padding_idx=0)
        #   (char_embedding): CharEncode(
        #     (embedding): Embedding(41, 32, padding_idx=0)
        #     (bilstm): LSTM(32, 32, bidirectional=True)
        #     (out): Linear(in_features=64, out_features=32, bias=True)
        #     (dropout): Dropout(p=0.5, inplace=False)
        #   )
        #   (feature_mapping): Linear(in_features=5, out_features=64, bias=True)
        #   (bilstm): LSTM(224, 128, bidirectional=True)
        #   (hidden2tag): Linear(in_features=352, out_features=5, bias=True)
        #   (crf_layer): CRF()
        #   (dropout): Dropout(p=0.5, inplace=False)
        #   (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (layer_norm1): LayerNorm((256, 224), eps=1e-05, elementwise_affine=True)
        #   (layer_norm2): LayerNorm((256, 352), eps=1e-05, elementwise_affine=True)
        #   (loss_fuc_cel): CrossEntropyLoss()
        # )
        # finish: 100%|███████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 14.91it/s]
        # Confusion Matrix:
        #  [[ 831.   31.  509.]
        #  [  85.  361.  220.]
        #  [ 426.  126. 3653.]]
        # --------------------------------------------------
        # Label: B-KP
        # TP:831.0 FN:540.0 FP:511.0 TN:4360.0
        # --------------------------------------------------
        #     Accuracy:   0.831624
        #     Misclassification:  0.168376
        #     Precision:  0.619225
        #     Sensitivity/Recall: 0.606127
        #     Specificity:        0.895093
        #     F1 Score:   0.612606
        # --------------------------------------------------
        # Label: I-KP
        # TP:361.0 FN:305.0 FP:157.0 TN:5419.0
        # --------------------------------------------------
        #     Accuracy:   0.925985
        #     Misclassification:  0.074015
        #     Precision:  0.696911
        #     Sensitivity/Recall: 0.542042
        #     Specificity:        0.971844
        #     F1 Score:   0.609797
        # --------------------------------------------------
        # Label: O
        # TP:3653.0 FN:552.0 FP:729.0 TN:1308.0
        # --------------------------------------------------
        #     Accuracy:   0.794777
        #     Misclassification:  0.205223
        #     Precision:  0.833638
        #     Sensitivity/Recall: 0.868728
        #     Specificity:        0.642121
        #     F1 Score:   0.850821
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Avg. Accuracy: 0.85080
        # Avg. Precision: 0.716591
        # Avg. F1-Score: 0.691075
        # Avg. Sensitivity: 0.672299
        # Avg. Specificity: 0.836353
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # SeqEval Metrics:
        #               precision    recall  f1-score   support

        #           KP       0.56      0.55      0.55      1371

        #    micro avg       0.56      0.55      0.55      1371
        #    macro avg       0.56      0.55      0.55      1371
        # weighted avg       0.56      0.55      0.55      1371

        # Precision given by SeqEval: 52.39%
        # Recall given by SeqEval: 55.07%
        # F1-Score given by SeqEval: 53.70%
        # Accuracy given by SeqEval: 77.62%
        # INFO: Total time of model training:5.402S

        # With Only CEM
        # Precision given by SeqEval: 52.30%
        # Recall given by SeqEval: 55.51%
        # F1-Score given by SeqEval: 53.86%
        # Accuracy given by SeqEval: 77.62%

        pass

    elif args.mode=="testgs91":
        s_time = time.time()
        
        ## Pre-Process JSON with Data Features
        ppd.build_testgs_data_set(
            json_data_folder, args.sections, interim_data_folder, args.stem,
            'testgs91.json')
        
        ## Load pre-trained model
        ## Infer/Predict on Test Gold Standard
        model.testgs(interim_data_folder, 'testgs91', corpus_type, args.mode, args.features)
        print("\033[031mINFO: Total time of model training:%sS\033[0m"%round(time.time() - s_time, 3))
        # (pytorch) [rgoli@node1118 NLP_KPIdentify]$ python main.py -m testgs91
        # ====================================================================================================
        # Mode: testgs91
        # Sections Included: ['title', 'abstract']
        # Features: ['CEM', 'WFOF', 'POS', 'LEN', 'TI', 'TR']
        # Vocab Stemming: no
        # Logging Level: warn
        # ====================================================================================================
        # 91-th document processed: 100%|███████████████████████████████████████████████████| 91/91 [00:02<00:00, 42.22it/s]
        # The 91 document calculation TF-IDF completed: 100%|█████████████████████████████| 91/91 [00:00<00:00, 9855.95it/s]
        # The 91 document has been calculated TextRank completed: 100%|████████████████████| 91/91 [00:00<00:00, 772.43it/s]
        # The 91-th document feature is completed: : 91it [00:00, 4935.37it/s]
        # INFO: testgs91 data saving completed!
        # PARAMS: {'word_embedding_dim': 128, 'char_embedding_dim': 32, 'word_max_len': 32, 'sentence_max_len': 256, 'hidden_dim': 128, 'use_crf': 'True', 'lstm_num_layers': 1, 'is_bidirectional': 'True', 'dropout': 0.5, 'tags_path': './interim_data/tags', 'epochs': 20, 'batch_size': 128, 'lr': 0.01, 'weight_decay': 0.0001, 'optim_scheduler_factor': 0.1, 'optim_scheduler_patience': 1, 'optim_scheduler_verbose': 'True'}
        # BiLSTM_CRF(
        #   (word_embedding): Embedding(4548, 128, padding_idx=0)
        #   (char_embedding): CharEncode(
        #     (embedding): Embedding(41, 32, padding_idx=0)
        #     (bilstm): LSTM(32, 32, bidirectional=True)
        #     (out): Linear(in_features=64, out_features=32, bias=True)
        #     (dropout): Dropout(p=0.5, inplace=False)
        #   )
        #   (feature_mapping): Linear(in_features=5, out_features=64, bias=True)
        #   (bilstm): LSTM(224, 128, bidirectional=True)
        #   (hidden2tag): Linear(in_features=352, out_features=5, bias=True)
        #   (crf_layer): CRF()
        #   (dropout): Dropout(p=0.5, inplace=False)
        #   (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (layer_norm1): LayerNorm((256, 224), eps=1e-05, elementwise_affine=True)
        #   (layer_norm2): LayerNorm((256, 352), eps=1e-05, elementwise_affine=True)
        #   (loss_fuc_cel): CrossEntropyLoss()
        # )
        # finish: 100%|███████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 13.98it/s]
        # Confusion Matrix:
        #  [[1787.   83. 1174.]
        #  [ 116.  593.  366.]
        #  [ 805.  214. 7190.]]
        # --------------------------------------------------
        # Label: B-KP
        # TP:1787.0 FN:1257.0 FP:921.0 TN:8363.0
        # --------------------------------------------------
        #     Accuracy:   0.823329
        #     Misclassification:  0.176671
        #     Precision:  0.659897
        #     Sensitivity/Recall: 0.587057
        #     Specificity:        0.900797
        #     F1 Score:   0.621349
        # --------------------------------------------------
        # Label: I-KP
        # TP:593.0 FN:482.0 FP:297.0 TN:10956.0
        # --------------------------------------------------
        #     Accuracy:   0.936811
        #     Misclassification:  0.063189
        #     Precision:  0.666292
        #     Sensitivity/Recall: 0.551628
        #     Specificity:        0.973607
        #     F1 Score:   0.603562
        # --------------------------------------------------
        # Label: O
        # TP:7190.0 FN:1019.0 FP:1540.0 TN:2579.0
        # --------------------------------------------------
        #     Accuracy:   0.792424
        #     Misclassification:  0.207576
        #     Precision:  0.823597
        #     Sensitivity/Recall: 0.875868
        #     Specificity:        0.626123
        #     F1 Score:   0.848929
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Avg. Accuracy: 0.85085
        # Avg. Precision: 0.716595
        # Avg. F1-Score: 0.691280
        # Avg. Sensitivity: 0.671517
        # Avg. Specificity: 0.833509
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # SeqEval Metrics:
        #               precision    recall  f1-score   support

        #           KP       0.61      0.54      0.58      3044

        #    micro avg       0.61      0.54      0.58      3044
        #    macro avg       0.61      0.54      0.58      3044
        # weighted avg       0.61      0.54      0.58      3044

        # Precision given by SeqEval: 58.20%
        # Recall given by SeqEval: 55.12%
        # F1-Score given by SeqEval: 56.62%
        # Accuracy given by SeqEval: 77.63%
        # INFO: Total time of model training:7.402S

        ## With Only CEM
        # Precision given by SeqEval: 57.50%
        # Recall given by SeqEval: 53.38%
        # F1-Score given by SeqEval: 55.37%
        # Accuracy given by SeqEval: 76.80%

        pass

    elif args.mode == "predict":
        
        total_data = utils.read_json_array(os.path.join(WORKING_PATH,"dummy.json"))
        for article in total_data:
            ppd.build_infer_data_set(article, args.sections, interim_data_folder, args.stem)
            kps = model.predict_kws(interim_data_folder, corpus_type, args.mode, args.features)
            act_kps = article['actual_keywords']
            act_kps.sort(reverse=True, key=len)
            
            print("GoldStandard Keywords:\n{}\nPredicted Keywords:\n{}\n\n".format(act_kps,kps))

    elif args.mode=='al':
        ## Activate Active Learning
        ## Pick test sample from Test Data Pool
        ## Incremental Train with Active Learning
        pass

    elif args.mode == "test":
        ## Metrics Calculation
        s_time = time.time()
        p, r, f1 = model.test(interim_data_folder, corpus_type, args.mode, args.features)
        print("\033[031mINFO: Total time of model test:%sS\033[0m"%round(time.time() - s_time, 3))

        # (pytorch) [rgoli@node1118 NLP_KPIdentify]$ python main.py -m test
        # ====================================================================================================
        # Mode: test
        # Sections Included: ['title', 'abstract']
        # Features: ['CEM', 'WFOF', 'POS', 'LEN', 'TI', 'TR']
        # Vocab Stemming: no
        # Logging Level: warn
        # ====================================================================================================
        # PARAMS: {'word_embedding_dim': 128, 'char_embedding_dim': 32, 'word_max_len': 32, 'sentence_max_len': 256, 'hidden_dim': 128, 'use_crf': 'True', 'lstm_num_layers': 1, 'is_bidirectional': 'True', 'dropout': 0.5, 'tags_path': './interim_data/tags', 'epochs': 20, 'batch_size': 128, 'lr': 0.01, 'weight_decay': 0.0001, 'optim_scheduler_factor': 0.1, 'optim_scheduler_patience': 1, 'optim_scheduler_verbose': 'True'}
        # BiLSTM_CRF(
        #   (word_embedding): Embedding(4548, 128, padding_idx=0)
        #   (char_embedding): CharEncode(
        #     (embedding): Embedding(41, 32, padding_idx=0)
        #     (bilstm): LSTM(32, 32, bidirectional=True)
        #     (out): Linear(in_features=64, out_features=32, bias=True)
        #     (dropout): Dropout(p=0.5, inplace=False)
        #   )
        #   (feature_mapping): Linear(in_features=5, out_features=64, bias=True)
        #   (bilstm): LSTM(224, 128, bidirectional=True)
        #   (hidden2tag): Linear(in_features=352, out_features=5, bias=True)
        #   (crf_layer): CRF()
        #   (dropout): Dropout(p=0.5, inplace=False)
        #   (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (layer_norm1): LayerNorm((256, 224), eps=1e-05, elementwise_affine=True)
        #   (layer_norm2): LayerNorm((256, 352), eps=1e-05, elementwise_affine=True)
        #   (loss_fuc_cel): CrossEntropyLoss()
        # )
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.81it/s]
        # INFO: p: 57.26, r: 53.93, f1: 55.55
        # Confusion Matrix:
        #  [[ 54694.   1092.  17820.]
        #  [  2054.  19974.   6322.]
        #  [ 14975.   4096. 197460.]]
        # --------------------------------------------------
        # Label: B-KP
        # TP:54694.0 FN:18912.0 FP:17029.0 TN:227852.0
        # --------------------------------------------------
        #     Accuracy:   0.887151
        #     Misclassification:  0.112849
        #     Precision:  0.762573
        #     Sensitivity/Recall: 0.743064
        #     Specificity:        0.930460
        #     F1 Score:   0.752692
        # --------------------------------------------------
        # Label: I-KP
        # TP:19974.0 FN:8376.0 FP:5188.0 TN:284949.0
        # --------------------------------------------------
        #     Accuracy:   0.957411
        #     Misclassification:  0.042589
        #     Precision:  0.793816
        #     Sensitivity/Recall: 0.704550
        #     Specificity:        0.982119
        #     F1 Score:   0.746524
        # --------------------------------------------------
        # Label: O
        # TP:197460.0 FN:19071.0 FP:24142.0 TN:77814.0
        # --------------------------------------------------
        #     Accuracy:   0.864318
        #     Misclassification:  0.135682
        #     Precision:  0.891057
        #     Sensitivity/Recall: 0.911925
        #     Specificity:        0.763212
        #     F1 Score:   0.901370
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Avg. Accuracy: 0.90296
        # Avg. Precision: 0.815815
        # Avg. F1-Score: 0.800195
        # Avg. Sensitivity: 0.786513
        # Avg. Specificity: 0.891930
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # SeqEval Metrics:
        #               precision    recall  f1-score   support

        #           KP       0.73      0.71      0.72     73606

        #    micro avg       0.73      0.71      0.72     73606
        #    macro avg       0.73      0.71      0.72     73606
        # weighted avg       0.73      0.71      0.72     73606

        # Precision given by SeqEval: 70.34%
        # Recall given by SeqEval: 71.32%
        # F1-Score given by SeqEval: 70.82%
        # Accuracy given by SeqEval: 85.44%
        # INFO: Total time of model test:24.148S

        ## With Only CEm
        # Precision given by SeqEval: 69.49%
        # Recall given by SeqEval: 70.03%
        # F1-Score given by SeqEval: 69.76%
        # Accuracy given by SeqEval: 84.93%
        # INFO: Total time of model test:22.516S
        pass

    elif args.mode == "createData":
        ## Read PubMed XML processed Data - For all CDSS
        pubmed_data_folder = os.path.join(WORKING_PATH, config.PUBMED_PROCESSED_DATA)
        fullData = data.load_pubmed_data(pubmed_data_folder)

        ## Remove HoldOut Data from Fulltext
        pmids_HO = utils.read_text(config.HOLDOUT_DATA)
        
        N=len(fullData)
        i=0
        j=0
        while i<N:
            if fullData[i]['pmid'] not in pmids_HO:
                fullData[j]=fullData[i]
                j+=1
            i+=1
        fullData[j:]=[]
        
        ## Generate JSON Files from XML
        data.generate_synthetic_dataset(fullData, json_data_folder, min_sentences=args.minSentences, filter_kp=args.filterKP)

    elif args.mode == "train":
        ## Build Data Features with JSON Data
        ppd.build_data_sets(json_data_folder, args.sections, interim_data_folder, args.stem)
        
        # start time
        s_time = time.time()
        model.train(interim_data_folder, corpus_type, args.features)
        
        print("\033[031mINFO: Total time of model training:%sS\033[0m"%round(time.time() - s_time, 3))

        # (pytorch) [rgoli@node1118 NLP_KPIdentify]$ python main.py -m train
        # ====================================================================================================
        # Mode: train
        # Sections Included: ['title', 'abstract']
        # Features: ['CEM', 'WFOF', 'POS', 'LEN', 'TI', 'TR']
        # Vocab Stemming: no
        # Logging Level: warn
        # ====================================================================================================
        # 1049-th document processed: 100%|█████████████████████████████| 1049/1049 [00:30<00:00, 34.84it/s]
        # The 1049 document calculation TF-IDF completed: 100%|███████| 1049/1049 [00:00<00:00, 9870.10it/s]
        # The 1049 document has been calculated TextRank completed: 100%|█| 1049/1049 [00:01<00:00, 645.47it
        # The 1049-th document feature is completed: : 1049it [00:00, 3788.70it/s]
        # 2099-th document processed: 100%|█████████████████████████████| 2099/2099 [00:51<00:00, 40.76it/s]
        # The 2099 document calculation TF-IDF completed: 100%|█████████████| 2099/2099 [00:00<00:00, 5265.27it/s]
        # The 2099 document has been calculated TextRank completed: 100%|████| 2099/2099 [00:02<00:00, 722.62it/s]
        # The 2099-th document feature is completed: : 2099it [00:00, 2787.75it/s]
        # INFO: train data saving completed!
        # INFO: test data saving completed!
        # INFO: Dataset partition completed
        # PARAMS: {'word_embedding_dim': 128, 'char_embedding_dim': 32, 'word_max_len': 32, 'sentence_max_len': 256, 'hidden_dim': 128, 'use_crf': 'True', 'lstm_num_layers': 1, 'is_bidirectional': 'True', 'dropout': 0.5, 'tags_path': './interim_data/tags', 'epochs': 20, 'batch_size': 128, 'lr': 0.01, 'weight_decay': 0.0001, 'optim_scheduler_factor': 0.1, 'optim_scheduler_patience': 1, 'optim_scheduler_verbose': 'True'}
        # BiLSTM_CRF(
        #   (word_embedding): Embedding(4548, 128, padding_idx=0)
        #   (char_embedding): CharEncode(
        #     (embedding): Embedding(41, 32, padding_idx=0)
        #     (bilstm): LSTM(32, 32, bidirectional=True)
        #     (out): Linear(in_features=64, out_features=32, bias=True)
        #     (dropout): Dropout(p=0.5, inplace=False)
        #   )
        #   (feature_mapping): Linear(in_features=5, out_features=64, bias=True)
        #   (bilstm): LSTM(224, 128, bidirectional=True)
        #   (hidden2tag): Linear(in_features=352, out_features=5, bias=True)
        #   (crf_layer): CRF()
        #   (dropout): Dropout(p=0.5, inplace=False)
        #   (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (layer_norm1): LayerNorm((256, 224), eps=1e-05, elementwise_affine=True)
        #   (layer_norm2): LayerNorm((256, 352), eps=1e-05, elementwise_affine=True)
        #   (loss_fuc_cel): CrossEntropyLoss()
        # )
        # loss:128.212: 100%|███████████████████████████████████████████████████| 107/107 [00:29<00:00,  3.66it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.17it/s]
        # Epoch: 1, loss: 746.478, p: 56.22, r: 50.49, f1: 53.2
        # loss:125.111: 100%|███████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.82it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.59it/s]
        # Epoch: 2, loss: 219.155, p: 55.96, r: 54.89, f1: 55.42
        # loss:79.982: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.88it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.67it/s]
        # Epoch: 3, loss: 158.904, p: 53.52, r: 56.18, f1: 54.82
        # loss:71.381: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.88it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.18it/s]
        # Epoch: 4, loss: 137.367, p: 55.57, r: 55.28, f1: 55.43
        # loss:81.697: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.89it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:11<00:00, 15.99it/s]
        # Epoch: 5, loss: 130.677, p: 54.63, r: 55.63, f1: 55.12
        # loss:69.45: 100%|█████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.88it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 16.36it/s]
        # Epoch: 6, loss: 128.969, p: 53.77, r: 56.67, f1: 55.18
        # loss:58.274: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.90it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.52it/s]
        # Epoch: 7, loss: 112.482, p: 51.97, r: 57.63, f1: 54.65
        # loss:54.873: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.89it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.68it/s]
        # Epoch: 8, loss: 103.867, p: 57.26, r: 53.93, f1: 55.55
        # loss:53.194: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.90it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.66it/s]
        # Epoch: 9, loss: 113.668, p: 52.33, r: 57.59, f1: 54.83
        # loss:47.825: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.85it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.63it/s]
        # Epoch: 10, loss: 99.136, p: 56.27, r: 53.98, f1: 55.1
        # loss:55.133: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.90it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.70it/s]
        # Epoch: 11, loss: 104.368, p: 51.48, r: 57.84, f1: 54.48
        # loss:61.411: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.92it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.20it/s]
        # Epoch: 12, loss: 91.871, p: 56.34, r: 54.67, f1: 55.5
        # loss:46.951: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.92it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.32it/s]
        # Epoch: 13, loss: 96.366, p: 52.84, r: 56.6, f1: 54.66
        # loss:30.372: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.92it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.35it/s]
        # Epoch: 14, loss: 88.394, p: 53.32, r: 57.17, f1: 55.18
        # loss:59.973: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.93it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.32it/s]
        # Epoch: 15, loss: 83.755, p: 54.14, r: 56.82, f1: 55.45
        # loss:46.323: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.90it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.71it/s]
        # Epoch: 16, loss: 86.125, p: 53.59, r: 56.45, f1: 54.98
        # loss:44.016: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.90it/s]
        # Epoch    17: reducing learning rate of group 0 to 1.0000e-03.
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.68it/s]
        # Epoch: 17, loss: 91.996, p: 53.71, r: 56.83, f1: 55.23
        # loss:41.097: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.93it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.73it/s]
        # Epoch: 18, loss: 85.928, p: 54.91, r: 55.82, f1: 55.36
        # loss:39.615: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.93it/s]
        # Epoch    19: reducing learning rate of group 0 to 1.0000e-04.
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.38it/s]
        # Epoch: 19, loss: 84.722, p: 55.25, r: 55.72, f1: 55.48
        # loss:41.356: 100%|████████████████████████████████████████████████████| 107/107 [00:27<00:00,  3.93it/s]
        # finish: 100%|█████████████████████████████████████████████████████████| 179/179 [00:10<00:00, 17.46it/s]
        # Epoch: 20, loss: 79.254, p: 54.68, r: 56.31, f1: 55.48
        # BEST: best_p: 57.26, best_r: 53.93, best_f1: 55.55
        # INFO: Total time of model training:767.734S

        ### With ONLY Char Embeddings
        # Epoch: 20, loss: 75.08, p: 50.99, r: 57.69, f1: 54.13
        # BEST: best_p: 56.53, best_r: 53.46, best_f1: 54.95
        # INFO: Total time of model training:765.13S:614.414S

    else:
        print("Invalid option selected!!!\n Mode of opertion: train/test/predict/createData")