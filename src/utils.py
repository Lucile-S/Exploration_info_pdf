#!/usr/bin/python
# utils.py 
import os
import glob
import re
import numpy as np
import pandas as pd
import datetime
import dateutil.parser

# PDF manipulation packages  
import tika 
from tika import parser

from pdfrw import PdfReader
import PyPDF2
#import pdftotext
import fitz

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

# API query packages
import requests
import feedparser
import xmltodict, json
import xml.etree.ElementTree as ET
import logging


def pdf_filename(pdf_path):
    File_name = os.path.basename(pdf_path)
    #Title = File_name.split('_')[1].replace('.pdf','')  
    return File_name


def pdf_page_count2(pdf_path):
    """
    Extract pdf page count using fitz/PyMupdf packages 
    Ref :https://pymupdf.readthedocs.io/en/latest/index.html
    Note : Tika package don't offer this option
    Note : error for some publication
    """
    pdf = fitz.open(pdf_path)
    page_count = pdf.pageCount
    return page_count

def pdf_page_count(pdf_file):
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


def extract_metadata_fitz(pdf_path):
    """
    Extract metadata from pdf using fitz/PyMupdf packages 
    Ref :https://pymupdf.readthedocs.io/en/latest/index.html
    Note : For some publications Fitz just extracts the first author whereas Tika package extracts the list of authors
    """
    pdf = fitz.open(pdf_path)
    metadata = pdf.metadata
    #print(metadata)
    if metadata:
        Title = metadata['title'] if 'title' in metadata.keys()  else None 
        Authors = metadata['author']  if 'author' in metadata.keys()  else None  # it just gives the first author for some publications
        Journal = str(metadata['subject']).split(',')[0] if 'subject' in metadata.keys() else None 
        Year  = re.findall(r'\d{4}', metadata['creationDate'])[0] if 'creationDate' in metadata.keys() else None
        Keywords = metadata['keywords'] if 'keywords' in metadata.keys()  else None
        doi = re.search(r'(doi:[a-z0-9.\/]*)', str(metadata['subject'])) if 'subject' in metadata.keys() else None 
        if doi:
            doi = doi.group(0)
        metadata_fitz ={'Title':Title, 'doi':doi, 'Authors':Authors, 'Journal':Journal, 'Year':Year, 'Keywords':Keywords}
        #print(metadata_fitz)
    return metadata_fitz


def try_fitz(var, key, metadata):
    """
    If the 'var' have not been extracted from the pdf using tika package try with fitz package.
    If the variable is still not found return 'var' = empty list [] and print a warning message
    """
    try :
        if var is None:
            var = metadata['key']
            #print(var)
        if var is None:
            raise Exception     
    except Exception:
        print(f"No {key} retrieved from the pdf using fitz or tika packages")
    if var is None:
        var=[]
    return var

def pdf_to_text_pdfminer(pdf_path, page_nb, max_page, retstr= StringIO()):
    manager = PDFResourceManager()
    #retstr = StringIO()
    layout = LAParams(all_texts=False, detect_vertical=True)
    device = TextConverter(manager, retstr, laparams=layout)
    filepath = open(pdf_path, 'rb')
    interpreter = PDFPageInterpreter(manager, device)
    for page in PDFPage.get_pages(filepath, pagenos=page_nb, maxpages=max_page):
        interpreter.process_page(page)
    text = retstr.getvalue()
    #print(text)
    filepath.close()
    device.close()
    retstr.close()
    splitted_text= []
    splitted_for_abstract =[]

    for line in text.split('\n\n '):
        line2 = line.replace('\n\n','').replace('\n','').strip()
        if line2 != '':
            splitted_text.append(line2)

    for line in text.split('\n\n'):
        line2 = line.replace('\n','').strip()
        if line2 != '':
            splitted_for_abstract.append(line2)


    return splitted_text, text, splitted_for_abstract

def splitted_text(pdf_path):
    pdf = parser.from_file(pdf_path)
    lines = text.splitlines()
    # for line in lines:
    #     splitted_text.append(line)
    
    # text2= text.replace('\n\n',"")
    # for line in text2.split('\n'):
    #     line2 = line.replace('\n','').strip()
    #     if line2 != '':
    #         splitted_text.append(line2)


def get_pdf_abstract(pdf_path):
    # get text from the first page 
    splitted_text = pdf_to_text_pdfminer(pdf_path,0,1, StringIO())[2]
    #print(splitted_text)
    Abstract = None
    if len(splitted_text[4]) > 400:
        Abstract =splitted_text[4] # ok pour 1
    elif len(splitted_text[5]) > 400 :
        Abstract= splitted_text[5]
    elif len(splitted_text[6]) > 400 :
        Abstract = splitted_text[6]
    else:
        Abstract =None
    return str(Abstract)

def get_medata_pdfminer(pdf_path):
    """
    This function extracts the journal, authors and doi from the pdf first page using pdfminer package 
    via the pdf_to_text_pdfminer() function defined above and regex commandes
    Warning  : It doesn't work for every pdf 
    """
    try : 
        first_page = pdf_to_text_pdfminer(pdf_path, 0, 1, StringIO())[1].replace('\n\n'," ")
    except TypeError :
        first_page = pdf_to_text_pdfminer(pdf_path, 0, 1, BytesIO())[1].replace('\n\n'," ")
    #print(first_page)

    # get journal
    regex_journal=r'''([:]\s+[A-Z]{1,}[^\d]{0,25}[.])'''
    Journal= re.search(regex_journal, first_page)
    if Journal :
        Journal = Journal.group(0).replace('\n'," ").replace(':','').replace(".","").strip()

    # get authors
    #print(first_page)
    regex_authors=r'''(\b[A-Z]{1}[a-z]*[-\s][A-Z]{0,1}.?\s?[A-Z][a-z]*\s?[A-Z]{0,1}\w+[0-9,]\b)'''
    authors= re.findall(regex_authors, first_page)
    Authors = [re.sub("\d+", "", author).replace(',','') for author in authors]
   

    #get doi
    regex_doi=r'''(\b10[.][0-9]{4,}\/[a-z]{1,}[.][0-9a-z]*[.]\d*[.]?[0-9]{0,2}[.]?[0-9]{0,3}\b)'''
    doi= re.search(regex_doi, first_page).group(0)

    metadata = {'Journal':Journal, 'Authors':Authors, 'doi':doi}
    return metadata

def try_regex(var, key, metadata):
    """
    If the 'var' have not been extracted from the pdf using tika and fitz packages try with regex command.
    If the variable is still not found return 'var' = empty list [] and print a warning message
    """
    try:
        if not var:
            var = metadata['key']
            print(f"{key} retrieved using pdfminer with regex")
        if not var:
            raise Exception     
    except Exception:
        print(f"No {key} retrieved from the pdf using pdfminer with regex")
    if not var:
        var=[]
    return var


def extract_metadata_pdf(pdf_path):
    """
    This function extracts metadata from pdf
    first using tika package, then using fitz package as an alternative if an information is None
    """
    pdf = parser.from_file(pdf_path)
    metadata =  pdf['metadata']
    #print(metadata)
    if metadata:
        Title = metadata['dc:title'] if 'dc:title' in metadata.keys()  else None 
        doi =  re.findall(r'(doi:[a-z0-9.\/]*)', metadata['cp:subject'])[0] if 'cp:subject' in metadata.keys()  else None 
        Authors = metadata['Author'] if 'Author' in metadata.keys()  else None 
        Journal =  metadata['cp:subject'].split(',')[0] if 'cp:subject' in metadata.keys()  else None 
        Keywords = metadata['Keywords'] if 'Keywords' in metadata.keys()  else None
        Year = metadata['Creation-Date'].split('-')[0] if 'Creation-Date' in metadata.keys() else None
    
    # using fitz 
    metadata_fitz = extract_metadata_fitz(pdf_path)
    #metadata_fitz_keys = metadata_fitz.keys()

    Title = try_fitz(Title, 'Title', metadata_fitz)
    Authors = try_fitz(Authors, 'Authors', metadata_fitz)
    Year = try_fitz(Year, 'Year', metadata_fitz)
    doi = try_fitz(doi, 'doi', metadata_fitz)
    Journal = try_fitz(Journal, 'Journal', metadata_fitz)
    Keywords = try_fitz(Keywords, 'Keywords', metadata_fitz)

    # Finally try with regex
    metadata_pdfminer = get_medata_pdfminer(pdf_path)
  
    # try doi with regex 
    try:
        if not doi:
            doi = metadata_pdfminer['doi']
            print(f"doi retrieved using pdfminer with regex")
        if not doi:
            raise Exception     
    except Exception:
        print(f"No doi retrieved from the pdf using pdfminer with regex")
        doi=[]

    # try  Journal with regex 
    try:
        if not Journal:
            Journal = metadata_pdfminer['Journal']
            print(f"Journal retrieved using pdfminer with regex")
        if not Journal:
            raise Exception     
    except Exception:
        print(f"No Journal retrieved from the pdf using pdfminer with regex")
        Journal=[]

    
    # try Authors with regex 
    try:
        if not Authors:
            Authors = metadata_pdfminer['Authors']
            print(f"Authors retrieved using pdfminer with regex")
        if not Authors:
            raise Exception     
    except Exception:
        print(f"No Authors retrieved from the pdf using pdfminer with regex")
        Authors=[]

    # get the Abstract  (pdfminer and regex)
    Abstract = get_pdf_abstract(pdf_path).replace('\n', " ")

    #store value in a dictionnary METADATA 
    #File_name = pdf_filename(pdf_path)
    Pages = pdf_page_count(pdf_path)
    METADATA  = {'Year': Year, 'Authors':Authors, 'Title':Title, 'Journal':Journal, 'doi':doi, 'Keywords':Keywords, 'Pages':Pages, 'Abstract':Abstract}
    #print(METADATA)
    return METADATA
    
if __name__ == "__main__":

    #pdf_path="/home/lucile/Extraction_info_pdf/publications/2020_Pre-arrayed Pan-AAV Peptide Display Libraries for Rapid Single-Round Screening.pdf"
    pdf_path= "/home/lucile/Extraction_info_pdf/publications/2019_Using a barcoded AAV capsid library to select for clinically relevant gene therapy vectors.pdf"
    #pdf_path2= "/home/lucile/Extraction_info_pdf/publications/2018_Adeno-Associated Virus Vectors- Rational Design Strategies for Capsid Engineering.pdf"


    dir = os.getcwd()
    # pdf list from './publications' folder
    pdf_dir = os.path.join(dir, '../publications')
    pdf_path_list = glob.glob(pdf_dir+'/*.pdf')
    for pdf_path in pdf_path_list:
        print(pdf_path)
        print('--Abstract--')
        A = get_pdf_abstract(pdf_path)
        print(A)
        print('---METADATA---')
        METADATA = extract_metadata_pdf(pdf_path)
        #print(METADATA)
        for key, value in METADATA.items():
            print(key,':', value)
        print('---next pdf ---')
  




