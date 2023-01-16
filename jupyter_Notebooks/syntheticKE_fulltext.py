import json 
import spacy
import scispacy
from scispacy.linking import EntityLinker
import pubmed_parser as pp

nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

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
for doc in nlp.pipe(comb_arr):
    text = [token.orth_ for token in doc if not token.is_punct | token.is_space] 

    kws = [str(x) for x in doc.ents]
    kws.sort(reverse=True, key=len)
    
    processed_texts[pmid_arr[idx]] = text
    scispacy_kw[pmid_arr[idx]]=kws
    
    idx+=1
    
with open("processedFullTexts.json", "w") as outfile:
    json.dump(processed_texts, outfile)

with open("syntheticFullText_KW.json", "w") as outfile:
    json.dump(scispacy_kw, outfile)  