import pandas as pd
import numpy as np
from Bio import Entrez
import re
import os
import glob
# API query packages
import requests
import xmltodict, json
import xml.etree.ElementTree as ET
import logging
LOG_FILENAME = "scraping_abstract_pubmed.log"
logging.basicConfig(filename='./log/'+ LOG_FILENAME, level=logging.INFO)

# you need to install Biopython:
# pip install biopython
# Ref : https://medium.com/@kliang933/scraping-big-data-from-public-research-repositories-e-g-pubmed-arxiv-2-488666f6f29b
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/12/searching-pubmed-with-python/


def search(query, retmax):
    Entrez.email = 'your.email@example.com'
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax=retmax,
                            retmode='xml', 
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'your.email@example.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results


def get_pmcid_and_doi(pmid:str):
    """
    This function converts pmid to pmcid : e.g. 23193287 -->  PMC3531190
    """
    query_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email_2@example.com&ids={pmid}&format=json"
    response = requests.get(query_url)
    result = response.json()
    try:
        pmcid = result['records'][0]['pmcid']
    except:
        pmcid = ""
    try :
        doi= result['records'][0]['doi']
    except: 
        doi = ""
    return pmcid, doi


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
    if not doi:
        try:
            doi_path = "./PubmedArticle/PubmedData/ArticleIdList/ArticleId"
            for article_id in root.findall(doi_path):
                if article_id.attrib['IdType'] == 'doi':
                    doi = article_id.text
                else :
                    doi = ""
        except:
            doi= ""


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
        if get_data(root, "Article/Journal/JournalIssue/PubDate/Year")  :
            year = int(get_data(root, "Article/Journal/JournalIssue/PubDate/Year"))
            publication_metadata['Year']=year 
        if get_data(root, "DateCompleted/Year") :
            year = int(get_data(root, "DateCompleted/Year"))
            publication_metadata['Year']=year
        if  get_data(root, "Article/Journal/JournalIssue/PubDate/MedlineDate"): 
            year = int( re.sub(r'[^\d]','',get_data(root, "Article/Journal/JournalIssue/PubDate/MedlineDate")))
            publication_metadata['Year']=year
        else :
            year =""
        #print(year)
    
    # journal of publication 
    if  get_data(root, "/MedlineJournalInfo/MedlineTA"):
        journal = get_data(root, "/MedlineJournalInfo/MedlineTA")
        publication_metadata['Journal']=journal
    else :
        journal = get_data(root, "Article/Journal/Title")
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

if __name__ == '__main__':
    # define path:
    dir = os.getcwd()
    save_dir = os.path.join(dir,'scrapped_Abstracts')

    # list of topics for which we want to retrieve article abstracts
    # topic_list_1 = ['adeno associated virus AAV', 'immunology','gene therapy','tropism', 'variant']
    # csv_names_1 =['AAV', 'immunology','gene_therapy','tropism', 'variant']
    topic_list_2 = ['genome engineering','regulatory element','sequence', 'transfection','epigenetics','post-translational modification']
    csv_names_2 =['genome_engineering','regulatory_element','sequence', 'transfection','epigenetics','post_translational_modification']
    topic_dict = dict(zip(topic_list_2, csv_names_2))
   

    # Pubmed API queries
    for topic in topic_dict.keys():
    #for topic in topics:
        print(f'{topic} query in progress....')
        logging.info(f'Abstract retrieved for {topic}')

        results = search(f'{topic}', 350) # 250 is the number of publications to retrieve
        pmids = results['IdList']
        try :
            df = pd.read_csv(f'./{topic_dict[topic]}_publications_pubmed.csv')
        except :
            df = pd.DataFrame(columns=['PMID', 'PMCID', 'Title', 'Year', 'Journal', 'DOI', 'Authors', 'Keywords', 'Abstract'])
        
        for pmid in pmids:
            if int(pmid) in df.values:
                print('Publication already retrieved')
            else:
                print(pmid)
                try : 
                    info = get_metadata_pubmed(pmid)
                except :
                    logging.info('Network is unreachable')

                #print(info)
                if  info['Abstract']: 
                    df=df.append(info, ignore_index=True)
                else:
                    print(f"No Abstract for this publication : {pmid}")
                    logging.info(f"No Abstract for this publication : {pmid}")
        print('------------------NEXT TOPIC--------------------') 
        
        #pmids.to_csv('./AAV_pmids.csv', index=False)
        
        df.to_csv(f'{save_dir}/{topic_dict[topic]}_publications_pubmed.csv', index=False)
        print(df.head())
        print(df.shape)

    # quick look of the results 
    
    csv_files = glob.glob(f'{dir}/*.csv', recursive=True)
    for file in csv_files:
        print(file)
        df = pd.read_csv(file)
        print(df.shape)
        print('--------')


    

    