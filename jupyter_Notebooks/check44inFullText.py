import json 
import pubmed_parser as pp

dicts_out_full = pp.parse_medline_xml('../data/pubmed_data/processed_data/processed_full_data.xml',
                                 year_info_only=False,
                                 nlm_category=False,
                                 author_list=False,
                                 reference_list=False)

dicts_out_44 = pp.parse_medline_xml('../data/pubmed_data/raw_data/44_abstracts_test.xml',
                                 year_info_only=False,
                                 nlm_category=False,
                                 author_list=False,
                                 reference_list=False)

pmids_44 = set([paper['pmid'] for paper in dicts_out_44])
pmids_fulltext = set([paper['pmid'] for paper in dicts_out_full])

intersection = pmids_fulltext.intersection(pmids_44)

print(intersection)
print("Common Articles: {}".format(len(intersection)))

print("Total Articles: {}".format(len(pmids_fulltext)))
print("Manual GS Articles: {}".format(len(pmids_44)))