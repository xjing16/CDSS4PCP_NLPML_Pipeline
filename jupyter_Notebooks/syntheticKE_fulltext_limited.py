import json 
import spacy
import scispacy
from scispacy.linking import EntityLinker
import pubmed_parser as pp
from tqdm import tqdm

nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

nlp2 = spacy.load('en_ner_bc5cdr_md')
nlp2.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

dicts_out = pp.parse_medline_xml('../data/pubmed_data/processed_data/processed_full_data.xml',
                                 year_info_only=False,
                                 nlm_category=False,
                                 author_list=False,
                                 reference_list=False)

comb_arr=[]
pmid_arr =[]
for paper in dicts_out:
    comb_arr.append(paper['title']+' '+paper['abstract'])
    pmid_arr.append(paper['pmid'])
    
idx=0
processed_texts = {}
scispacy_kw = {}
no_kws_trimmed =[]
no_kws=[]
post_no_kws =[]
outlier_kws=[
    'http','www','university','department','antibiotic','antimicrobial','institute','ministry', 'pubmed',
    '.gov','.org','.com','.edu','.net',
    'city','disease','injury','trauma','syndrome','country','national','regimen','swelling',
    'surgery', 'medication','infection','stroke','diabetes','bleeding','comorbid','java','python',
    "united states",'united kingdom','india','china','germany','france','ghana','australia','italy','england','japan',
    'english','spanish','french','british','spain'
    ]
with tqdm(nlp.pipe(comb_arr),total=3326) as pbar:
    for doc in pbar:
        # sents = [str(x) for x in doc.sents]
        # if len(sents)<3:
        #     continue
        text = [token.orth_ for token in doc if not token.is_punct | token.is_space] 
        rule_out_ents = [str(_) for _ in nlp2(comb_arr[idx]).ents]
        rule_out_ents = [_ for _ in rule_out_ents if not _.isupper()]
        kws = [str(_) for _ in doc.ents]
        # print('-'*50)
        # print(len(kws))
        temp_kws_len = len(kws)
        # print(kws)
        if rule_out_ents==[]:
            kws = [_ for _ in kws]
        else:
            kws = [_ for _ in kws if _ not in rule_out_ents]
        # print(kws)
        kws = [_ for _ in kws if all([True if n_ not in _.lower() else False for n_ in outlier_kws])]
        no_kws_trimmed.append(temp_kws_len-len(kws))
        no_kws.append(temp_kws_len)
        post_no_kws.append(len(kws))
        # print(len(kws))
        # print(rule_out_ents)
        # print(kws)
        # print('-'*50)
        kws.sort(reverse=True, key=len)
        
        processed_texts[pmid_arr[idx]] = text
        scispacy_kw[pmid_arr[idx]]=kws
        
        idx+=1
        pbar.set_description("The %s document is processed" % (idx + 1))
    
with open("processedFullTexts.json", "w") as outfile:
    json.dump(processed_texts, outfile)

with open("syntheticFullText_KW.json", "w") as outfile:
    json.dump(scispacy_kw, outfile)  

print("Total KWS generated:", sum(no_kws))
print("Avg. KWS generated/abstract:", sum(no_kws)/len(no_kws))

print("Total KWS trimmed: ",sum(no_kws_trimmed))
print("Avg. KWS trimmed:", sum(no_kws_trimmed)/len(no_kws_trimmed))

print("Total KWS - post processing: ",sum(post_no_kws))
print("Avg. KWS - post processing:", sum(post_no_kws)/len(post_no_kws))
