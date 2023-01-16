import json 
import os
import spacy
import scispacy
from scispacy.linking import EntityLinker
import pubmed_parser as pp
from tqdm import tqdm

outlier_kws=[
    'http','www','university','department','antibiotic','antimicrobial','institute','ministry', 'pubmed',
    '.gov','.org','.com','.edu','.net',
    'city','disease','injury','trauma','syndrome','country','national','regimen','swelling',
    'surgery', 'medication','infection','stroke','diabetes','bleeding','comorbid','java','python',
    "united states",'united kingdom','india','china','germany','france','ghana','australia','italy','england','japan',
    'english','spanish','french','british','spain'
    ]

def load_pubmed_data(path):
    '''
    Load PubMed XML parsed Data with PMIDS
    '''
    data = pp.parse_medline_xml(path,
                                 year_info_only=False,
                                 nlm_category=False,
                                 author_list=False,
                                 reference_list=False)

    return data

def generate_synthetic_dataset(fullData, json_data_path, predictData=False, min_sentences=3, filter_kp = False, model_name="en_core_sci_lg", additional_data=[]):
    '''
    Create Train & Test Datasets as JSON with Synthetic Keywords
    '''
    print('Spacy Model to generate Synthetic Keywords:')
    print(model_name)
    nlp = spacy.load(model_name)

    if model_name=="en_core_sci_lg":
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    else:
        nlp.add_pipe('sentencizer')

    if filter_kp:
        nlp2 = spacy.load('en_ner_bc5cdr_md')
        nlp2.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    comb_arr=[]
    pmid_arr =[]
    dataset = []
    p_id=0
    
    fullData.extend(additional_data)
    total_len = len(fullData)
    with tqdm(enumerate(fullData),total=total_len) as pbar1:
        for index, paper in pbar1:
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

    # print("Given Full Data: ", total_len)
    # dataset.extend(additional_data)
    # print("+ Additional Data: ", len(additional_data))
    # total_len = len(dataset)
    # print("New DataSet: ", total_len)

    if predictData==True:
        with open(os.path.join(json_data_path, "testgs.json"), "w") as outfile:
            json.dump(dataset, outfile)
        return True
            
    idx=0
    remove_sent_no_outliers = []
    with tqdm(nlp.pipe(comb_arr),total=total_len) as pbar1:
        # for doc in nlp.pipe(comb_arr):
        for doc in pbar1:
            # text = [token.orth_ for token in doc if not token.is_punct | token.is_space] 
            sents = [[token.orth_.lower() for token in sent if not token.is_punct | token.is_space] for sent in doc.sents]

            ## List all the sentences with are minimum threshold
            if len(sents)<min_sentences:
                remove_sent_no_outliers.append(idx)
                idx+=1
                continue

            text = [token.orth_ for token in doc if not token.is_punct | token.is_space]

            kws = [str(x) for x in doc.ents]

            ## Filter KWS for DISEASES/GPE/Institutes
            if filter_kp:
                rule_out_ents = [str(_) for _ in nlp2(comb_arr[idx]).ents]
                rule_out_ents = [_ for _ in rule_out_ents if not _.isupper()]
                if rule_out_ents!=[]:
                    kws = [_ for _ in kws if _ not in rule_out_ents]
                kws = [_ for _ in kws if all([True if n_ not in _.lower() else False for n_ in outlier_kws])]

            kws.sort(reverse=True, key=len)
            
            dataset[idx]["keywords"]=kws
            dataset[idx]['fullText'] = text
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

    dataset_length = len(dataset)
    # with open('CDSS_PMIDs_SentTrim.txt','w') as f:
    #     for data in dataset:
    #         f.write(data["id"]+"\n")
    print("Total Dataset Length: ",dataset_length)
    print("Train Dataset Length: ",dataset_length//3)
    print("Test  Dataset Length: ",dataset_length-dataset_length//3)
    with open(os.path.join(json_data_path, "train.json"), "w") as outfile:
        json.dump(dataset[:dataset_length//3], outfile)

    with open(os.path.join(json_data_path, "test.json"), "w") as outfile:
        json.dump(dataset[dataset_length//3:], outfile)  
    return True


def gen_gs_dataset(gsData, json_data_path):
    '''
    Create Gold Standard dataset as JSON with Keywords
    '''
    
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

    with open(os.path.join(json_data_path, "testgs.json"), "w") as outfile:
        json.dump(dataset, outfile)
        

# if __name__ == "__main__":
#     fullData = load_data()
#     generate_dataset(fullData)
#     return True