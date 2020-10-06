#!/usr/bin/python
#pubmed.py 
"""
From a directory containing several publications in pdf format,
the script creates a csv file with the 'pmid','pmcid','Year','Authors','Title', 'Journal', 'DOI','keywords', 'Abstract' for every publications.
Those informations (Metadata) are retrieved using Pubmed API.
"""
import os
import glob
import re
import numpy as np
import pandas as pd

from utils import *

# API query packages
import requests
import xmltodict, json
import xml.etree.ElementTree as ET

###########################
# ID related functions    #
###########################

def get_pmid(title:str):
    """
    This function fetches the publication pmid by querying use pubmed API using publication title
    """
    query_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={title}&retmode=json&field=title"
    try: 
        response = requests.get(query_url)
        result = response.json()
        pmid = result['esearchresult']['idlist'][0] # the output is a list with a unique element into it
        if response.status_code != 200:
            raise Exception 
        else:
            return pmid
    except Exception:
        return []
        print(f"Http request failed... status code : {response.status_code}")



def convert_doi_to_pmid_and_pmcid(doi:str):
    """
    This function convert DOI to pmid and pmcid
    """
    query_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids={doi}&format=json"
    response = requests.get(query_url)
    result = response.json()
    try : 
        pmid = result['records'][0]['pmid']
        pmcid = result['records'][0]['pmcid']
        return pmid, pmcid
    except KeyError:
        print('error')
        return [],[]


def get_pmcid_and_doi(pmid:str):
    """
    This function converts pmid to pmcid : e.g. 23193287 -->  PMC3531190
    """
    query_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids={pmid}&format=json"
    response = requests.get(query_url)
    result = response.json()
    pmcid = result['records'][0]['pmcid']
    doi= result['records'][0]['doi']
    return pmcid, doi

#################################
# METADATA related functions    #
#################################

def get_metadata_pubmed(pmid):
    """
    This function fetches publication metadata (title, doi, authors, year of publication, keywords, journal) from xml content using pubmed API 
    """
    def get_data(root, path_to_data, metadata_root="./PubmedArticle/MedlineCitation/"):
        try:
            data = root.find(metadata_root + path_to_data).text
        except :
            data = None
        return data

    # init a list that will contain metadata of the publication
    publication_metadata = {}

    # store pubmed id
    publication_metadata['PMID']=pmid
    #print(pmid)

    # store pmc id
    pmcid, doi = get_pmcid_and_doi(pmid)
    publication_metadata['PMCID']=pmcid

    # fetch XML abstract from pubmed
    publication_url =  f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=XML&rettype=abstract"
    response = requests.get(publication_url)
    root = ET.fromstring(response.content)

    # root of the XML content we are interested in
    metadata_root = "./PubmedArticle/MedlineCitation/"

    # title
    title = get_data(root, "Article/ArticleTitle")
    publication_metadata['Title']=title
    #print(title)

    # year of publication 
    try: 
        year = int(get_data(root, "Article/ArticleDate/Year"))
        publication_metadata['Year']=year 
        #print(year)
    except:
        year = int(get_data(root, "DateCompleted/Year"))
        publication_metadata['Year']=year 
        #print(year)
    
    # journal of publication 
    journal = get_data(root, "/MedlineJournalInfo/MedlineTA")
    publication_metadata['Journal']=journal
    #print(journal)

    # DOI - if present
    ids_path = "./PubmedArticle/PubmedData/ArticleIdList/ArticleId"
    for id in root.findall(ids_path):
        if id.get("IdType") == "doi":
            doi = id.text
    publication_metadata['DOI'] = doi
    #print(doi)

    # authors
    author_last_names = root.findall(metadata_root + "/AuthorList/Author/LastName")
    author_fore_names = root.findall(metadata_root + "/AuthorList/Author/ForeName")
    authors =[]    
    for last_name,fore_name in zip(author_last_names,author_fore_names):
        author = ' '.join([fore_name.text,last_name.text])
        authors.append(author)

    publication_metadata['Authors'] = authors
    # print(authors)

    # keywords - not always present
    keywords = []
    try :
        for keyword in root.findall(metadata_root  + "/KeywordList/Keyword"):
            keyword_name= keyword.text
            keywords.append(keyword_name)
        #keywords = ', '.join(keywords)
    except :
        for keyword in root.findall(metadata_root  + "/MeshHeadingList/MeshHeading"):
            keyword_name = keyword.find("DescriptorName").text
            keywords.append(keyword_name)
        #keywords = ', '.join(keywords)
    #print(keywords)
 
    publication_metadata['Keywords']=keywords

    # abstract
    abstract= get_data(root, "Article/Abstract/AbstractText")
    publication_metadata['Abstract']=abstract

    return publication_metadata

################################
### FORMAT related functions ###
################################

def get_xml(pmcid):
    query_url =  f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&tool=my_tool&email=my_email@example.com&retmode=XML'
    response = requests.get(query_url)
    root = ET.fromstring(response.content)
    return root

def xml_to_json(pmcid, json_directory=None):
    # fetch XML content
    query_url =  f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&tool=my_tool&email=my_email@example.com'
    response = requests.get(query_url)
    obj = xmltodict.parse(response.content,attr_prefix='')
    json_obj = json.dumps(obj, ensure_ascii = False,indent=1)
    if json_directory :
        with open(json_directory, "w") as json_file:
            json_file.write(json_obj)
    return json_obj

def save_json(json_obj):
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    # current directory 
    dir = os.getcwd()
    # list of pdf publications present in the './publications' folder
    pdf_dir = os.path.join(dir, '../publications')
    pdf_paths = glob.glob(pdf_dir+'/*.pdf')
    # create a dataframe to store all results 
    column_names=['Publi_ID','PMID','PMCID','Year','Authors','Title', 'Journal', 'DOI','Keywords', 'Abstract']
    df  = pd.DataFrame(columns = column_names)
    IDs_table = pd.DataFrame()
    # loop over pdf publications
    seq_id = 1
    for pdf_path in pdf_paths:
        # retrieve title and doi from the pdf file
        METADATA_pdf = extract_metadata_pdf(pdf_path)
        Title = METADATA_pdf['Title']
        doi = METADATA_pdf['doi']
        if not Title:
            Title = os.path.basename(pdf_path).split('_')[1].replace('.pdf','')
        
        # get pmid from the pubmed API using title
        try : 
            pmid = get_pmid(Title)
            pmcid, doi  = get_pmcid_and_doi(pmid)
        except:
            pmid, pmcid = get_pmid_and_pmcid(doi)
    
        #get metadata : title, authors, year, doi...
        METADATA = get_metadata_pubmed(pmid)

   