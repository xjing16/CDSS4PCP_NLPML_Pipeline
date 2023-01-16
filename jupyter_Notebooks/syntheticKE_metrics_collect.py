import pubmed_parser as pp
import pke
import os
import string
from nltk.corpus import stopwords
import json 
import spacy
import scispacy
from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

dicts_out = pp.parse_medline_xml('../data/pubmed_data/raw_data/44_abstracts_test.xml',
                                 year_info_only=False,
                                 nlm_category=False,
                                 author_list=False,
                                 reference_list=False)

files = os.listdir("/home/rgoli/MetaMap-src/data/gold_standard/AnnotationResults_Abstracts/done")
print(files)

manual_pmids = [file.split('.')[0][4:] for file in files if file!='.ipynb_checkpoints']

pmid_manual_kw={}
for file in files:
    if file=='.ipynb_checkpoints': continue
    with open("/home/rgoli/MetaMap-src/data/gold_standard/AnnotationResults_Abstracts/done/"+file,'r') as fp:
        temp = fp.read().splitlines()
        pmid_manual_kw[temp[0].strip()]=temp[1:]

# print(pmid_manual_kw['11604796'])

comb_arr=[]
pmid_arr =[]
for paper in dicts_out:
    comb_arr.append(paper['title']+' '+paper['abstract'])
    pmid_arr.append(paper['pmid'])
    
# print(pmid_arr[0:2])

#### sciSpacy BERT ####
idx=0
processed_texts = {}
scispacy_kw = {}
for doc in nlp.pipe(comb_arr):
    text = [token.orth_ for token in doc if not token.is_punct | token.is_space] 

    kws = [str(x) for x in doc.ents]
    kws.sort(reverse=True, key=len)
    
    processed_texts[pmid_arr[idx]] = text
    scispacy_kw[pmid_arr[idx]]=kws
    
    idx+=1
    

with open("processedTexts.json", "w") as outfile:
    json.dump(processed_texts, outfile)

with open("scispacy_KW.json", "w") as outfile:
    json.dump(scispacy_kw, outfile)  

#### PositionRank ####
pos = {'NOUN', 'PROPN', 'ADJ'}
grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
positionRank_kws = {}
for idx, doc in enumerate(comb_arr):
    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=doc,language='en',normalization=None)
    extractor.candidate_selection(grammar=grammar,maximum_word_number=3)
    extractor.candidate_weighting(window=10,pos=pos)
    kws = [str(x) for x,y in extractor.get_n_best(n=20)] 
    kws.sort(reverse=True, key=len)
    positionRank_kws[pmid_arr[idx]] = kws
    
with open("positionRank_KW.json", "w") as outfile:
    json.dump(positionRank_kws, outfile)
# print(positionRank_kws)

#### MultiPartite Rank ####
pos = {'NOUN', 'PROPN', 'ADJ'}
stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist += stopwords.words('english')
multiPartiteRank_kws = {}
for idx, doc in enumerate(comb_arr):
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=doc)
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.74,
                                  method='average')
    kws = extractor.get_n_best(n=20)
    kws = [str(x) for x,y in extractor.get_n_best(n=20)] 
    kws.sort(reverse=True, key=len)
    multiPartiteRank_kws[pmid_arr[idx]] = kws
    
with open("multiPartiteRank_KW.json", "w") as outfile:
    json.dump(multiPartiteRank_kws, outfile)
    
#### TopicRank ####
pos = {'NOUN', 'PROPN', 'ADJ'}
stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist += stopwords.words('english')
topicPartiteRank_kws = {}
for idx, doc in enumerate(comb_arr):
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=doc)
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    extractor.candidate_weighting(threshold=0.74, method='average')
    kws = extractor.get_n_best(n=20)
    kws = [str(x) for x,y in extractor.get_n_best(n=20)] 
    kws.sort(reverse=True, key=len)
    topicPartiteRank_kws[pmid_arr[idx]] = kws
    
with open("topicRank_KW.json", "w") as outfile:
    json.dump(topicPartiteRank_kws, outfile)    