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
import datetime
import dateutil.parser

from pdfrw import PdfReader

# API query packages
import requests
import feedparser
import xmltodict, json
import xml.etree.ElementTree as ET
import logging

def get_pdf_title_and_doi(publication_path):
    """
    This function return title and doi extracted from a pdf publication.
    If the title can not be extract from pdf metadata,
    it is extracted from the pdf file name that is supposed to follow the pattern <publication_date>_<title>.pdf
    If the DOI is not present, the function will return None.
    Then this infomation can be used to query pubmed API for this particular publication. 
    """
    reader = PdfReader(publication_path)

    if reader.Info.Title:
        title =  reader.Info.Title.strip('()').replace("\\", "") # remove brackets surrounding the title text
    else:
        title = os.path.basename(publication_path).split('_')[1].replace('.pdf','')  

    if reader.Info.Subject:
        doi = re.findall(r'(doi:[a-z0-9.\/]*)', reader.Info.Subject)[0]
    else:
        doi = None 
        print(f"No DOI retrive from pdf reader for '{title}'")
    return title, doi

def get_pmid(title:str):
    """
    This function fetches the publication pmid by querying use pubmed API using publication title
    """
    query_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={title}&retmode=json&field=title"
    response = requests.get(query_url)
    result = response.json()
    pmid = result['esearchresult']['idlist'][0] # the output is a list with a unique element into it
    if response.status_code != 200:
        logger.error("HTTP Error {} in query".format(result.get('status', 'no status')))
        return []
    return pmid

def get_pmid_and_pmcid(doi:str):
    """
    This function convert DOI to pmid and pmcid
    """
    query_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids={doi}&format=json"
    response = requests.get(query_url)
    result = response.json()
    pmid = result['records'][0]['pmid']
    pmcid = result['records'][0]['pmcid']
    return pmid, pmcid


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


def get_metadata(pmid):
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
    publication_metadata['pmid']=pmid
    print(pmid)

    # store pmc id
    pmcid, doi = get_pmcid_and_doi(pmid)
    publication_metadata['pmcid']=pmcid

    # fetch XML abstract from pubmed
    publication_url =  f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=XML&rettype=abstract"
    response = requests.get(publication_url)
    root = ET.fromstring(response.content)

    # root of the XML content we are interested in
    metadata_root = "./PubmedArticle/MedlineCitation/"

    # title
    title = get_data(root, "Article/ArticleTitle")
    publication_metadata['Title']=title
    print(title)

    # abstract
    abstract= get_data(root, "Article/Abstract/AbstractText")
    publication_metadata['Abstract']=abstract

    # year of publication 
    try: 
        year = int(get_data(root, "Article/ArticleDate/Year"))
        publication_metadata['Year']=year 
        print(year)
    except:
        year = int(get_data(root, "DateCompleted/Year"))
        publication_metadata['Year']=year 
        print(year)
    
    # journal of publication 
    journal = get_data(root, "/MedlineJournalInfo/MedlineTA")
    publication_metadata['Journal']=journal
    print(journal)

    # DOI - if present
    ids_path = "./PubmedArticle/PubmedData/ArticleIdList/ArticleId"
    for id in root.findall(ids_path):
        if id.get("IdType") == "doi":
            doi = id.text
    publication_metadata['DOI'] = doi
    print(doi)

    # authors
    author_last_names = root.findall(metadata_root + "/AuthorList/Author/LastName")
    author_fore_names = root.findall(metadata_root + "/AuthorList/Author/ForeName")
    authors =[]    
    for last_name,fore_name in zip(author_last_names,author_fore_names):
        author = ' '.join([fore_name.text,last_name.text])
        authors.append(author)

    publication_metadata['Authors'] = authors
    print(authors)

    # keywords - special case
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
    print(keywords)
    publication_metadata['keywords']=keywords

    return publication_metadata

##############
### FORMAT ###
##############

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
        json.dump(data, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    # current directory 
    dir = os.getcwd()
    # list of pdf publications present in the './publications' folder
    publication_dir = os.path.join(dir, '../publications')
    publication_paths = glob.glob(publication_dir+'/*.pdf')
    # create a dataframe to store all results 
    column_names=['Publi_ID','pmid','pmcid','Year','Authors','Title', 'Journal', 'DOI','keywords', 'Abstract']
    df  = pd.DataFrame(columns = column_names)
    IDs_table = pd.DataFrame()
    # loop over pdf publications
    seq_id = 1
    for publication_path in publication_paths:
        # retrieve title and doi from the pdf file
        title, doi = get_pdf_title_and_doi(publication_path)
        # get pmid from the pubmed API using title
        try : 
            pmid = get_pmid(title)
            pmcid, doi  = get_pmcid_and_doi(pmid)
        except:
            pmid, pmcid = get_pmid_and_pmcid(doi)
        print(pmid,pmcid,doi)
        #get metadata : title, authors, year, doi...
        metadata = get_metadata(pmid)
        # create Publi_ID 
        Publi_ID = "Pub_{:06d}".format(seq_id)
        metadata['Publi_ID']=Publi_ID
        seq_id += 1
        # Add as row to a Dataframe df
        df = df.append(metadata, ignore_index=True)
        IDs_table  = IDs_table.append({'DOI': doi, 'PMCID': pmcid, 'PMID': pmid, 'Publi_ID':Publi_ID, 'pdf_name': os.path.basename(publication_path)},ignore_index=True)
        print('--------------------------------')
    # save dataframes as csv files
    csv_name = 'Publication_Informations.csv'
    df.to_csv(publication_dir + '/../' + 'csv_name', index=False)
    IDs_table.to_csv(publication_dir + '/../'+'IDs_table.csv', index=False)
    # checking
    print(df.head())
    print(IDs_table.head())
        #print(xml_to_json('PMC5746594'))
