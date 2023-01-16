# -*- coding: utf-8 -*-
import math
from tqdm import tqdm
from collections import Counter

# Calculate IDF
def cal_idf(datas):
    total_files  = len(datas)
    all_words = set([j for line in datas for j in line])
    temp = {word: 0.0 for word in all_words}
    for word in temp.keys():
        for words in datas:
            if word in words:
                temp[word] += 1.0
    idf = {}
    for k, v in temp.items():
        idf[k] = math.log((total_files + 1) / (v + 1), 2)
    return idf

# Calculate TF-IDF
def tf_idf(datas):
    idf = cal_idf(datas)
    return_datas = []
    temp_len_datas = len(datas)
    with tqdm(enumerate(datas), total=temp_len_datas) as pbar:
        for index,words in pbar:
            # words = datas[index]
            num_words = len(words)
            tf = Counter(words)
            keywords = {}
            for k, v in tf.items():
                keywords[k] = (v / num_words) * idf[k]
            return_datas.append(keywords)
            pbar.set_description("The %s document calculation TF-IDF completed"%(index + 1))
    return return_datas
