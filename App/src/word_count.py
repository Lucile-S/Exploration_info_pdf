#!/usr/bin/python
# pdf_word_count.py 
import numpy as np
import pandas as pd
import logging
import datetime
import dateutil.parser
import os 
import re
import glob
from  collections import Counter
import time
from collections.abc import Iterable 

# API query packages
import requests
import feedparser
import xmltodict, json
import xml.etree.ElementTree as ET

# PDF manipulation packages 
from pdfrw import PdfReader
import PyPDF2
import textract
import pdftotext

import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO, BytesIO

import tika 
from tika import parser

import pdfplumber

# NLP packages
## NLTK
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
## Spacy
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
nlp = spacy.load('en')


def get_num_pages(pdf_file):
    """
    Using py2pdf (faster than pdfminer)
    with pdfminer : print(len(list(extract_pages(pdf_file))))
    """
    #open allows you to read the file.
    pdfFileObj = open(pdf_file,'rb')
    #The pdfReader variable is a readable object that will be parsed.
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # number of pages
    num_pages = pdfReader.numPages
    return num_pages

##############################
### Custom tokenizer Spacy ###
##############################
"""
To prevent splitting of url and intra hyphen word 
# Ref : https://stackoverflow.com/questions/48112057/customized-tag-and-lemmas-for-urls-using-spacy
# Ref : https://support.prodi.gy/t/how-to-tell-spacy-not-to-split-any-intra-hyphen-words/1456
# Ref : https://stackoverflow.com/questions/56439423/spacy-parenthesis-tokenization-pairs-of-lrb-rrb-not-tokenized-correctly
"""
def custom_tokenizer(nlp):
    #infixes = r'''\b\)\b''' + r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]'''
    #infixes = tuple(r"[.\,\?\:\;\...\‘\’\`\“\”\"\'~]") + tuple([r"\b\)\b"])
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''') 
    #infix_re = compile_infix_regex(nlp.Defaults.infixes)
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    simple_url_re = re.compile(r'''^https?://''')
    re_compile_list = ['^https?://', '^AAV.*?$', '^AAV.*\s*.*?\)$']
    AAV_re = re.compile('|'.join(re_compile_list))

    return Tokenizer(nlp.vocab, token_match=AAV_re.match,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                rules=nlp.Defaults.tokenizer_exceptions
                                )

def custom_tokenizer2(nlp):
    
    my_prefix = r'[0-9]\.'
    
    #all_prefixes_re = spacy.util.compile_prefix_regex(tuple(list(nlp.Defaults.prefixes) + [my_prefix]))
    prefixes_re = spacy.util.compile_prefix_regex(['\.\.\.+', '[!&:,]'])
    
    # Handle ( that doesn't have proper spacing around it
    #custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[!&:,()]']
    custom_infixes = ['\.\.\.+', '[!&:,]']
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    #infix_re = spacy.util.compile_infix_regex(custom_infixes)
    #infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))

    #suffix_re = spacy.util.compile_suffix_regex(tuple(list(nlp.Defaults.suffixes) + custom_suffixes)  )
    custom_suffixes =['\.\.\.+', '[!&:,]']
    #custom_suffixes =['[!&:,]']
    suffix_re = spacy.util.compile_suffix_regex(custom_suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefixes_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                rules=nlp.Defaults.tokenizer_exceptions
                                )
    # return Tokenizer(nlp.vocab, rules=nlp.Defaults.tokenizer_exceptions,
    #  prefix_search = all_prefixes_re.search, 
    #  infix_finditer = infix_re.finditer, 
    #  suffix_search = suffix_re.search,
    #  token_match=None)

nlp.tokenizer = custom_tokenizer(nlp)


def extract_text_tika(pdf_file):
    pdf= parser.from_file(pdf_file)
    text = pdf['content']
    # convert to string
    # safe_text = str(text).encode('utf-8', errors  = 'ignore')
    # print(text)
    # print('########')
    # print(safe_text)
    return text


def extract_text_plomber(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        pages = pdf.pages
        for i in range(len(pages)): 
            print(pages[i].extract_text())


def extract_text_miner(pdf_file):
    """
    pip install pdfminer.six
    Using pdfminer : https://stackoverflow.com/questions/26494211/extracting-text-from-a-pdf-file-using-pdfminer-in-python/26495057
    https://pypi.org/project/pdfminer.six/
    """
    text = extract_text(pdf_file)
    print(text)
    return text


def getWordCount(pdf_file, nlp, mode='tika'):
    nlp.tokenizer = custom_tokenizer(nlp)
    """
    Using spacy tokenizer 
    """
    if mode == 'miner': 
        text = extract_text_miner(pdf_file)
    else : 
        text = extract_text_tika(pdf_file)
    clean_text = text.replace('\n',' ').replace('\x0c',' ').replace('\t',' ').strip()
    print(clean_text)
    # The word_tokenize() function will break our text phrases into individual words.
    tokens =  [token.text for token in nlp(clean_text)]
    # remove ponctuations
    punctuations = [';',':','[',']',',', ' ','  ', '.','-','–','©',')','(','—','   ']
    # stopwords list
    stop_words = stopwords.words('english')
    #only returns a list of words that are NOT IN stop_words and NOT IN punctuations.
    clean_tokens = [word for word in tokens if not word in STOP_WORDS and not word in punctuations]
    #print(clean_tokens)
    print(clean_tokens)
    count = print(len(clean_tokens))
    return clean_tokens, count

def count_word_Abstract(pmcid):
    query_url =  f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&tool=my_tool&email=my_email@example.com&retmode=XML'
    response = requests.get(query_url)
    root = ET.fromstring(response.content)
    abstract= ''.join([text.replace("<\\/?[bi]>", "") for text in root.find("./article/front/article-meta/abstract/p").itertext()])
    print(abstract)
    count_word_abstract = len(abstract.split(" "))
    print(abstract.split())
    print(count_word_abstract)

def count_word_xml_body(pmcid):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    query_url =  f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&tool=my_tool&email=my_email@example.com&retmode=XML'
    response = requests.get(query_url)
    root = ET.fromstring(response.content)
    body =''.join([_RE_COMBINE_WHITESPACE.sub(" ", text) for text in root.find("./article/body").itertext()])
    print(body)
     # The word_tokenize() function will break our text phrases into individual words.
    tokens = [token.text for token in nlp(body)]
    print(tokens)
    # remove ponctuations
    punctuations = [';',':','[',']',',', ' ','  ', '.','-','–','©',')','(']
    # stopwords list
    stop_words = stopwords.words('english')
    #only returns a list of words that are NOT IN stop_words and NOT IN punctuations.
    clean_tokens = [word for word in tokens if not word in STOP_WORDS and not word in punctuations]
    #print(clean_tokens)
    #print(clean_tokens)
    count = print(len(clean_tokens))
    return clean_tokens, count
  

# from xml.etree import ElementTree as ET

# xml = '<a>one two three<b>four five<c>Six Seven</c></b></a>'
# tree = ET.fromstring(xml)
# total = sum(len(text.split()) for text in tree.itertext())
# # 7


# def AAV_extract(pdf_text, mode='tika'):
#     # AAV2/8, # AAV-GFP
#     AAV_regex_1 = r'(:?^AAV[\d-\/]*[-\w]*\s)'
#     AAV_regex_2 = r'(:?^AAV[\d-\/]*\(*[\d\w,\)]*\)*\s*[-\d\w\s,]*\)\s)'
#     if mode == 'miner': 
#         text = extract_text_miner(pdf_file)
#     else : 
#         text = extract_text_tika(pdf_file)
#     print(text.split())

# def pdf_to_json(pdf_path, mode='tika'):
#     if mode == 'miner': 
#         text = extract_text_miner(pdf_file)
#     else : 
#         text = extract_text_tika(pdf_file)
#     _dict={}
#     content = text.splitlines()
#     for line in content:
#         if ':' not in line:
#             continue
#         key, value = line.split(':')
#         _dict[key.strip()] = value.strip()
#     print(_dict)
#     return _dict

    # print(json.dumps(text, indent=4))


# def get_data(page_content):
#     _dict = {}
#     page_content_list = page_content.splitlines()
#     for line in page_content_list:
#         if ':' not in line:
#             continue
#         key, value = line.split(':')
#         _dict[key.strip()] = value.strip()
#     return _dict

# page_data = get_data(page_content)
# json_data = json.dumps(page_data, indent=4)
# print(json_data)

if __name__ == "__main__":
    
    dir = os.getcwd()

    pdf_dir = os.path.join(dir, '../publications')
    # pmid_pdf_table = pd.read_csv(pdf_dir +'/'+  'IDs_table.csv')
    # print(pmid_pdf_table.head())

    pdf_file='/home/lucile/Extraction_info_pdf/src/../publications/2017_Tropism of engineered and evolved recombinant AAVserotypes in therd1mouse andex vivoprimate retina.pdf'
    
    nb_pages = get_num_pages(pdf_file)
    print(nb_pages)
    #count_word_Abstract("PMC5746594")
    #count_word_xml_body("PMC5746594")
    getWordCount(pdf_file, nlp)
    #pdf_to_xml(pdf_file, list(range(1,nb_pages)))


    

    # for pdf in pmid_pdf_table['file']:
    #     pdf_file =  os.path.join(pdf_dir, pdf)
    # 
    # start_time = time.time()
    # #getWordCount(pdf_file)
    # #get_XML('PMC5746594')
    # count_word_Abstract('PMC5746594')
    # #AAV_extract(pdf_file)
    # #extract_text_tika('PMC5746594') 
    # print("--- %s seconds --- tika ---" % (time.time() - start_time))
    # print("#########")
    #pdf_to_json(pdf_file)
    # start_time = time.time()
    # extract_text_miner(pdf_file) 
    # print("--- %s seconds --- miner ---" % (time.time() - start_time))
    # start_time = time.time()
    # extract_text_plomber(pdf_file) 
    # print("--- %s seconds --- plomber ---" % (time.time() - start_time))


    # # Add word count into the dataframe for each publications
    # #for publication_path in publication_paths:
    # #print('---NLTK----')
    # #getWordCount(publication_paths[0])
    # # print('---Spacy----')
    # # getWordCount2(publication_paths[0])
    # print('-----Test3------------')
    # getWordCount5(publication_paths[0])



