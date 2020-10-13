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

#log file
import logging

def pdf_filename(pdf_path):
    File_name = os.path.basename(pdf_path)
    #Title = File_name.split('_')[1].replace('.pdf','')  
    return File_name

def pdf_page_count(pdf_path):
    """
    Extract pdf page count using using py2pdf or fitz/PyMupdf packages  (faster than pdfminer)
    with pdfminer : print(len(list(extract_pages(pdf_file))))
    Ref :https://pymupdf.readthedocs.io/en/latest/index.html
    Note : Tika package don't offer this option
    Note : error for some publications
    """
    try : 
        #open allows you to read the file.
        pdfFileObj = open(pdf_path,'rb')
        #The pdfReader variable is a readable object that will be parsed.
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        # number of pages
        page_count = pdfReader.numPages
    except :
        pdf = fitz.open(pdf_path)
        page_count = pdf.pageCount
    return page_count

def pdf_to_text_pdfminer(pdf_path, page_nb, max_page, retstr= StringIO()):
    manager = PDFResourceManager()
    #retstr = StringIO()
    layout = LAParams(all_texts=False, detect_vertical=True)
    device = TextConverter(manager, retstr, laparams=layout, codec = 'utf-8')
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

    # Remove some extra line break and spaces
    text =text.replace('-\n\n','').replace('-\n','').replace(', \n',' ').replace('\n\n,','').replace(',\n\nand',' ').strip()
    text=re.sub(r'(?<=[a-z,])\n\n(?=[a-z]+)',' ', text)

    for paragraph in text.split('\n\n'):
        paragraph = paragraph.replace('\n\n','').replace(' \n',' ')
        # print(paragraph)
        # print('-----')
       
        #paragraph = paragraph.replace('\n\n','').replace(' \n',' ').replace('\n',' ')
        #.replace('   ',' ').replace('  ',' ').strip()
        paragraph = re.sub(r'\s+',' ', paragraph).strip()
        if paragraph != '':
            splitted_text.append(paragraph)

    # for line in text.split('\n\n '):
    #     line2 = line.replace('\n\n','').replace('\n','').strip()
    #     if line2 != '':
    #         splitted_text.append(line2)

    # for line in text.split('\n\n'):
    #     line2 = line.replace('\n','').strip()
    #     if line2 != '':
    #         splitted_for_abstract.append(line2)

    splitted_for_abstract= splitted_text
    return splitted_text, text, splitted_for_abstract


def get_pdf_Title_and_Authors(pdf_path):
    """
    Get the title by splitting into lines the first page of the pdf (using pdfminer package) and keeping only the ten first lines
    Then (in order to separate from journal name, doi or author list) applying filters in the number of words and digits because usually title length is between 50-150 characters
    and doesn't contain a lot of digit. Also some regex substitutions are used for special case encountered in a pool of publications. 
    """
    _ , _ ,splitted_text= pdf_to_text_pdfminer(pdf_path,0,1, StringIO())
    #print(splitted_text)
    result=[]
    result_index =[]
    
    for i in range(0,10):
        s = splitted_text[i].replace('  ','')
        numbers = sum(c.isdigit() for c in s)
        spaces  = sum(c.isspace() for c in s)
        if 45 < len(s) < 200 and numbers < 5 and spaces < 30:
            result.append(s) 
            result_index.append(i)
    try:
        combined_pat = r'|'.join(('Article','ORIGINAL', 'ARTICLE' ,'Review','JAMA Ophthalmology \| ',
        '([0-9,]+\s?[A-Z][a-z]*\s+[A-Z][a-z]*\s*){1,}'))
        #https://stackoverflow.com/questions/36589797/split-multiple-joined-words-with-upper-and-lower-case
        Title = re.sub(combined_pat, '',result[0].strip())
        # special case regex 
        Title = re.sub('\B(?=[A-Z][a-z])',' ', Title)
        Title = re.sub('([A-Z][a-z]*\s[A-Z]?[.]?\s?[A-Z][a-z]*[&]\s[A-Z][a-z]*\s[A-Z]?[.]?\s?[A-Z][a-z]*)','', Title)    
        

        location_words = ['Department', 'Section','National','Institut','University','Foundation','Medical', 'School']
        split_at =[word for word in location_words if word in splitted_text[result_index[0]+1]]
        #print(splitted_text[result_index[0]+1]) 
        """
        Generally Authors is right after the title, so we select them by adding one to the index of the title line.
        not working if authors and title in the same line
        """
        regex_authors=r'''(\b[A-Z]{1}[a-z]*[-\s][A-Z]{0,1}.?-?\s?[A-Z][a-z]*-?\s?[A-Z]{0,1}\w*\b)'''

        if split_at:
            authors = splitted_text[result_index[0]+1].split(split_at[0])[0]
        else:
            authors = splitted_text[result_index[0]+1]
    
        authors= re.findall(regex_authors, authors)
        Authors = [re.sub("\d+|ID", "", author).replace('†','').replace('*','') for author in authors]
        
        return Title.strip(), Authors   
    except:
        Title = ""
        Authors =""
        print('Title and Authors not retrieve with regex')
        logging.info('Title and Authors not retrieve with regex')
        return Title, Authors

def get_pdf_abstract(pdf_path):
    """
    Get the Abstract by splitting into paragraph the first page of the pdf (using pdfminer package) and looping over them.
    Then, minimum character count conditions is applied + in order to separate abstract from author correspondance a another one is applied using  the @ character.
    """
    # get text from the first page 
    splitted_text = pdf_to_text_pdfminer(pdf_path,0,1, StringIO())[2]
    #print(splitted_text)
    possible_Abstracts = []
    for i in range(2,len(splitted_text)):
        #print(splitted_text[i])
        if len(splitted_text[i]) > 500 and len(re.findall("@",splitted_text[i])) < 1  and len(re.findall(r"[;,]",splitted_text[i])) < 20 : 
            possible_Abstracts.append(splitted_text[i])
    to_remove = r'|'.join(('Abstract: ','Abstract ', 'ABSTRACT ', 'RESULTS '))
    try: 
        Abstract=re.sub(to_remove,'', possible_Abstracts[0].strip())
    except:
        Abstract = None
        print('No abstract retrieved with regex')
        logging.info('No abstract retrieved with regex')
    return str(Abstract)


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
        Journal = str(metadata['subject']).split(',')[0] if 'subject' in metadata.keys()  else None 
        Year  = re.findall(r'\d{4}', metadata['creationDate'])[0] if 'creationDate' in metadata.keys() else None
        Keywords = metadata['keywords'] if 'keywords' in metadata.keys()  else None
        doi = re.search(r'(doi:[a-z0-9.\/]*)', str(metadata['subject'])) if 'subject' in metadata.keys() else None 
        if doi:
            doi = doi.group(0)
        metadata_fitz ={'Title':Title, 'DOI':doi, 'Authors':Authors, 'Journal':Journal, 'Year':Year, 'Keywords':Keywords}
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
        logging.info(f"No {key} retrieved from the pdf using fitz or tika packages")
    if var is None:
        var=""
    return var



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
    print(first_page)
    regex_authors=r'''(\b[A-Z]{1}[a-z]*[-\s][A-Z]{0,1}.?\s?[A-Z][a-z]*\s?[A-Z]{0,1}\w*[0-9,]\b)'''
    authors= re.findall(regex_authors, first_page)
    Authors = [re.sub("\d+", "", author).replace(',','') for author in authors]
   
    #get doi
    #regex_doi=r'''(\b10[.][0-9]{4,}\/[a-z]{1,}[.][0-9a-z]*[.]\d*[.]?[0-9]{0,2}[.]?[0-9]{0,3}\b)'''
    try :
        regex_doi=r'''(\b10[.][0-9]{4,}\/[ﬁa-zA-Z]{1,}[.]?[0-9a-zA-Z]*[.]?\d*[.]?[0-9]{0,2}[.]?[0-9]{0,3}([-]?[0-9]{0,4}){0,5}\b)'''
        doi= re.search(regex_doi, first_page).group(0)
    except:
        doi =""
    
    metadata = {'Journal':Journal, 'Authors':Authors, 'DOI':doi}
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
            logging.info(f"{key} retrieved using pdfminer with regex")
        if not var:
            raise Exception     
    except Exception:
        print(f"No {key} retrieved from the pdf using pdfminer with regex")
        logging.info(f"No {key} retrieved from the pdf using pdfminer with regex")
    if not var:
        var=""
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
        try :
            doi =  re.findall(r'(doi:[a-z0-9.\/-]*)', metadata['cp:subject'])[0] if 'cp:subject' in metadata.keys() else None
            print('xxxx',metadata['cp:subject'])
        except :
            doi = ""
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

    # try title with regex:
    if not Title:
        Title= get_pdf_Title_and_Authors(pdf_path)[0]
        print('Title tentitavelly retrieve with regex')
        logging.info('Title tentitavelly retrieve with regex')
    if not Authors:
        Authors= get_pdf_Title_and_Authors(pdf_path)[1]
        print('Authors tentitavelly retrieve with regex')
        logging.info('Authors tentitavelly retrieve with regex')

    # try doi with regex 
    try:
        if not doi:
            doi = metadata_pdfminer['DOI']
            print(f"doi retrieved using pdfminer with regex")
            logging.info(f"doi retrieved using pdfminer with regex")
        if not doi:
            raise Exception     
    except Exception:
        print(f"No doi retrieved from the pdf using pdfminer with regex")
        logging.info(f"No doi retrieved from the pdf using pdfminer with regex")
        doi=""

    # try  Journal with regex 
    try:
        if not Journal:
            Journal = metadata_pdfminer['Journal']
            print(f"Journal retrieved using pdfminer with regex")
            logging.info(f"Journal retrieved using pdfminer with regex")
        if not Journal:
            raise Exception     
    except Exception:
        print(f"No Journal retrieved from the pdf using pdfminer with regex")
        logging.info(f"No Journal retrieved from the pdf using pdfminer with regex")
        Journal=""    
    # try Authors with regex 
    try:
        if not Authors:
            Authors = metadata_pdfminer['Authors']
            print(f"Authors retrieved using pdfminer with regex")
            logging.info(f"Authors retrieved using pdfminer with regex")
        if not Authors:
            raise Exception     
    except Exception:
        print(f"No Authors retrieved from the pdf using pdfminer with regex")
        logging.info(f"No Authors retrieved from the pdf using pdfminer with regex")
        Authors=""

    # get the Abstract  (pdfminer and regex)
    Abstract = get_pdf_abstract(pdf_path)
    
    # Store value in a dictionnary METADATA 
    # File_name = pdf_filename(pdf_path)
    Pages = pdf_page_count(pdf_path)
    METADATA  = {'Year': Year, 'Authors':Authors, 'Title':Title, 'Journal':Journal, 'DOI':doi, 'Keywords':Keywords, 'Pages':Pages, 'Abstract':Abstract}
    #print(METADATA)
    return METADATA

##################################################
#                      Test                      #
##################################################
    
if __name__ == "__main__":

    dir = os.getcwd()
    ## pdf list from './publications' folder
    pdf_dir = os.path.join(dir, '../publications')
    pdf_path_list = glob.glob(pdf_dir+'/*.pdf')

    # pdf to test 
    #pdf_path="/home/lucile/Extraction_info_pdf/publications/2020_Pre-arrayed Pan-AAV Peptide Display Libraries for Rapid Single-Round Screening.pdf"
    #pdf_path= "/home/lucile/Extraction_info_pdf/publications/2019_Using a barcoded AAV capsid library to select for clinically relevant gene therapy vectors.pdf"
    #pdf_path2= "/home/lucile/Extraction_info_pdf/publications/2018_Adeno-Associated Virus Vectors- Rational Design Strategies for Capsid Engineering.pdf"
    # pdf_path3="/home/lucile/Extraction_info_pdf/publications_2/1-s2.0-S0731708520313674-main.pdf"
    # pdf_path4 ="/home/lucile/Extraction_info_pdf/publications_2/fimmu-11-01135.pdf"
    # pdf_path5='/home/lucile/Extraction_info_pdf/publications_2/jimd.12316.pdf'
    # pdf_path6='/home/lucile/Extraction_info_pdf/src/../publications_2/acn3.51165.pdf'
    pdf_path_list=['/home/lucile/WhiteLab_project/Extraction_info_pdf/publications/s41467-019-11687-8.pdf']
    # pdf =get_medata_pdfminer(pdf_path4)
    # nb_pages = pdf_page_count(pdf_path4)
    # print(nb_pages)
    # print(pdf_to_text_pdfminer(pdf_path6,0,1, BytesIO()))

    ## get_pdf_Title(pdf_path3)
    for pdf_path in pdf_path_list:
        print(pdf_path)
        #pdf_to_text_pdfminer(pdf_path,0,1, StringIO())
        # print(pdf_page_count(pdf_path))
        # print('--Title--')
        # print(get_pdf_Title_and_Authors(pdf_path)[0])
        # # print('--Authors--')
        # print(get_pdf_Title_and_Authors(pdf_path)[1])
        # print('--Abstract--')
        # print(get_pdf_abstract(pdf_path))
        print('---METADATA---')
        METADATA = extract_metadata_pdf(pdf_path)
        print('---DOI---')
        print(METADATA['DOI'])
        # for key, value in METADATA.items():
        #     print(key,':', value)
        #     print('------')
        print('------------------NEXT PDF -----------------')
  






