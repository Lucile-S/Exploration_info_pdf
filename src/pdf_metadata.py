"""
From a directory containing several publications in pdf format,
the script creates a csv file with the 'pmid','pmcid','Year','Authors','Title', 'Journal', 'DOI','keywords', 'Abstract' for every publications.
Those informations (Metadata) are retrieved using Pubmed API.
"""

import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import requests
import feedparser
import logging
import datetime
import dateutil.parser
import os 
from pdfrw import PdfReader
import os
import glob
import re

def get_pdf_info(publication_path):
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

def get_publication_pmid(search:str):
    """
    This function fetches the publication pmid by querying use pubmed API  using publication title
    """
    base_url= "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    query_url = base_url +  f"db=pubmed&term={search}&retmode=json&field=title"
    response = requests.get(query_url)
    result = response.json()
    pmid = result['esearchresult']['idlist'][0] # the output is a list with a unique element into it
    if response.status_code != 200:
        logger.error("HTTP Error {} in query".format(result.get('status', 'no status')))
        return []
    return pmid

def get_pmcid(pmid:str):
    """
    This function converts pmid  to pmcid : e.g. 23193287 -->  PMC3531190
    """
    query_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids={pmid}&format=json"
    response = requests.get(query_url)
    result = response.json()
    pmcid = result['records'][0]['pmcid']
    return pmcid 

def get_pdf_content_from_pmc(pmcid:str):
    # fetch XML content
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    query_url = base_url + f'db=pmc&id={pmcid}&tool=my_tool&email=my_email@example.com'
    response = requests.get(query_url)
    #print(response)
    root = ET.fromstring(response.content)
    #print(root.text)
    #return content 
 

def get_publication_metadata(pmid):
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
    publication_metadata['pmcid']=get_pmcid(pmid)

    # fetch XML abstract from pubmed
    base_url =  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    publication_url = base_url + f"db=pubmed&id={pmid}&retmode=XML&rettype=abstract"
    response = requests.get(publication_url)
    root = ET.fromstring(response.content)

    # root of the XML content we are interested in
    matadata_root = "./PubmedArticle/MedlineCitation/"

    # title
    title = get_data(root, "Article/ArticleTitle")
    publication_metadata['Title']=title
    print(title)

    # abstract
    abstract= get_data(root, "Article/Abstract/AbstractText")
    publication_metadata['Abstract']=abstract

    # year of publication 
    try: 
        year = int(get_data(root, "DateCompleted/Year"))
        publication_metadata['Year']=year 
        print(year)
    except:
        year = int(get_data(root, "DateRevised/Year"))
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
    
    author_last_names = root.findall(matadata_root + "/AuthorList/Author/LastName")
    author_fore_names = root.findall(matadata_root + "/AuthorList/Author/ForeName")
    authors =[]    
    for last_name,fore_name in zip(author_last_names,author_fore_names):
        author = ' '.join([fore_name.text,last_name.text])
        authors.append(author)

    publication_metadata['Authors'] = authors
    print(authors)

    # keywords - special case
    keywords = []
    try :
        for keyword in root.findall(metadata_root  + "/MeshHeadingList/MeshHeading"):
            keyword_name = keyword.find("DescriptorName").text
            keywords.append(keyword_name)
        keywords = ', '.join(keywords)
    except :
        pass
    print(keywords)
    publication_metadata['keywords']=keywords

    return publication_metadata



if __name__ == "__main__":
    dir = os.getcwd()
    # list of pdf publications present in the './publications' folder
    publication_dir = os.path.join(dir, '../publications')
    publication_paths = glob.glob(publication_dir+'/*.pdf')

    # create a dataframe to store all results 
    column_names=['pmid','pmcid','Year','Authors','Title', 'Journal', 'DOI','keywords', 'Abstract']
    df  = pd.DataFrame(columns = column_names)
    # loop over pdf publications
    for publication_path in publication_paths:
        # retrieve title and doi from the pdf file
        title, doi = get_pdf_info(publication_path)
        # get pmid from the pubmed API using title
        pmid = get_publication_pmid(title)
        # get metadata : title, authors, year, doi...
        metadata = get_publication_metadata(pmid)
        # Add as row to a Dataframe df
        df = df.append(metadata, ignore_index=True)

    # save as csv
    csv_name = 'Publication_Informations.csv'
    df.to_csv(publication_dir + 'csv_name', index=False)
    print(df.head())


#     # save to csv 
#     df.to_csv('data/articles_pubmed_1.csv', index=False)

#     return df



# info_pdf={}
# for publication_path in publication_paths: 
#     title, doi = get_pdf_info(publication_path)
#     # add to a dictionnary with key=title and value=doi
#     info_pdf[title]=doi


# pmid = get_publication_pmid('Development of AAV Variants with Human Hepatocyte Tropism and Neutralizing Antibody Escape Capacity')
# print(pmid)

# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=32637455

# deg 

#28872643