#!/usr/bin/python
# text_mining.py 
import numpy as np
import pandas as pd
import dateutil.parser
import os 
import re
import glob
from  collections import Counter
from collections import defaultdict
from bs4 import BeautifulSoup

# API query packages
import xmltodict, json
import xml.etree.ElementTree as ET

# Homemade scripts
from utils import *
from pubmed import *

# packages to access pdf 
import tika 
from tika import parser

import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from io import StringIO, BytesIO

#NLP packages
# NLTK
# import nltk
# nltk.download('stopwords')
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords 

## Spacy
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
nlp = spacy.load('en')


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
    re_compile_list = [r'^https?://', r'^AAV.*?$', r'^AAV.*\s*.*?\)$']
    AAV_re = re.compile('|'.join(re_compile_list))

    return Tokenizer(nlp.vocab, token_match=AAV_re.match,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                rules=nlp.Defaults.tokenizer_exceptions
                                )

def text_tokenizer(text, nlp=spacy.load('en')):
    nlp.tokenizer = custom_tokenizer(nlp)
    """
    Using spacy tokenizer 
    """
    #print(text)
    clean_text = text.replace('\n',' ').replace('\x0c',' ').replace('\r',' ').replace('\n\n',' ').replace('\t',' ').strip()
    #print(clean_text)
    # The word_tokenize() function will break our text phrases into individual words.
    tokens =  [token.text for token in nlp(clean_text)]
    # remove ponctuations
    punctuations = [';',':','[',']',',', ' ','  ', '.','-','–','©',')','(','—','   ']
    # stopwords list
    stop_words = STOP_WORDS
    #only returns a list of words that are NOT IN stop_words and NOT IN punctuations.
    clean_tokens = [word for word in tokens if not word in STOP_WORDS and not word in punctuations]
    #print(clean_tokens)
    return clean_text, clean_tokens

def find_AAV_terms(text):
    """
    returns a dictionnay 
    """
    #regex_aav = r"AAV(\d*\/|\d*-)*\w*(\([\w, -]*\))*(-\w+)*"
    AAV_terms = defaultdict(int)
    regex_AAV =r'''(AAV(\d*\/|\d*-)*\w*(\([\w, -]*\))*(-\w+)*)'''
    #ref_reg = r"^\d+( *\**\.)* ([A-Z]\S* ([A-Z]\S* )*[A-Z]\w*(, )*)+"
    for match in re.findall(regex_AAV, text):
        if len(match[0]) < 45:
            AAV_terms[match[0]] += 1
    #AAV_terms.append({k: v for k,v in d.items()})
    return AAV_terms 

def find_AAV_related_publications(xml_pdf):
    related_references = []
    regex_ref  = r'''^[ ]?\d+( *\**\.)* ([A-Z]\S* ([A-Z]\S* )*[A-Z]\w*(, )*)+'''
    regex_AAV = r'''AAV(\d*\/|\d*-)*\w*(\([\w, -]*\))*(-\w+)*'''
    #print(regex_AAV)
    for el in xml_text.find_all('p'):
        #print(el.text)
        if re.search(regex_ref, el.text,re.M) and re.search(regex_AAV,el.text,re.M | re.I):
            # print(el.text.strip())
            # print('-----')
            ref = re.split(r'^[0-9]{1,2}[.]{0,1}', el.text.replace('\n',' ').replace('**.', "").replace('*.', ""))[1].strip()
            related_references.append(ref)
    return related_references

def find_AAV_related_publications2(xml_pdf):
    related_references = []
    regex_ref  = r'''^\d+( *\**\.)* ([A-Z]\S* ([A-Z]\S* )*[A-Z]\w*(, )*)+'''
    regex_AAV = r'''AAV(\d*\/|\d*-)*\w*(\([\w, -]*\))*(-\w+)*'''
    # keep only references that match regex_ref and that contain "AAV" term
    for text in splitted_text:
        if re.search(regex_ref, text,re.M) and re.search(regex_AAV, text,re.M | re.I):
            related_references.append(text)
    return related_references

def pdf_to_xml(pdf_path):
    raw = parser.from_file(pdf_path, xmlContent=True)
    soup = BeautifulSoup(raw['content'], 'lxml')
    return soup


def find_AAV_term_related_publications_xml(xml_text, AAV_term):
    AAV_term_related_references = []
    regex_ref  = r'''^[ ]?\d+( *\**\.)* ([A-Z]\S* ([A-Z]\S* )*[A-Z]\w*(, )*)+'''
    regex_AAV = re.split(r"^(AAV\d{0,1})", AAV_term)[1]
    #print(regex_AAV)
    for el in xml_text.find_all('p'):
        #print(el.text)
        if re.search(regex_ref, el.text,re.M) and re.search(regex_AAV,el.text,re.M | re.I):
            # print(el.text.strip())
            # print('-----')
            ref = re.split(r'^[0-9]{1,2}[.]{0,1}', el.text.replace('\n',' ').replace('**.', "").replace('*.', ""))[1].strip()
            AAV_term_related_references.append(ref)
    return AAV_term_related_references

def find_AAV_term_related_publications(splitted_text, AAV_term):
    AAV_term_related_references = []
    regex_ref  = r'''^\d+( *\**\.)* ([A-Z]\S* ([A-Z]\S* )*[A-Z]\w*(, )*)+'''
    regex_AAV = re.split(r"^(AAV\d{0,1})", AAV_term)[1]
    # keep only references that match regex_ref and that contain a specific AAV term
    for text in splitted_text:
        #print(text)
        if re.search(regex_ref, text,re.M) and re.search(regex_AAV, text,re.M | re.I):
            ref = re.split(r'^[0-9]{1,2}[.]{0,1}', text.replace('\n',' ').replace('**.', "").replace('*.', ""))[1].strip()
            # print(text)
            AAV_term_related_references.append(ref)
    return AAV_term_related_references

#####################################
#          Function Test            #
#####################################


if __name__ == "__main__":
    # current directory 
    dir = os.getcwd()

    # list of pdf publications present in the './publications' folder
    pdf_dir = os.path.join(dir, '../publications')
    pdf_paths = glob.glob(pdf_dir+'/*.pdf')

    # pdf path for test 
    #pdf_path="/home/lucile/Extraction_info_pdf/publications/2020_Pre-arrayed Pan-AAV Peptide Display Libraries for Rapid Single-Round Screening.pdf"
    #pdf_path= "/home/lucile/Extraction_info_pdf/publications/2019_Using a barcoded AAV capsid library to select for clinically relevant gene therapy vectors.pdf"
    #pdf_path2= "/home/lucile/Extraction_info_pdf/publications/2018_Adeno-Associated Virus Vectors- Rational Design Strategies for Capsid Engineering.pdf"
    #pdf_path ="/home/lucile/Extraction_info_pdf/publications/2017_Tropism of engineered and evolved recombinant AAVserotypes in therd1mouse andex vivoprimate retina.pdf"

    # loop over pdf publications
    for pdf_path in pdf_paths:
        print(pdf_path)
        page_count = pdf_page_count(pdf_path)
        splitted_text, text, _ = pdf_to_text_pdfminer(pdf_path,0,page_count,retstr = StringIO())
        #splitted_text, btext = pdf_to_text_pdfminer(pdf_path,0,page_count,retstr = BytesIO())
        AAV_terms = [ key for key in find_AAV_terms(text).keys()]
        xml_text = pdf_to_xml(pdf_path)
        for AAV_term in AAV_terms: 
            print(AAV_term)
            print('-----next AAV term------')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('--------next Publication------')
            



