"""
From a directory containing several publications in pdf format,
this script creates a csv file with the 'pmid','pmcid','Year','Authors','Title', 'Journal', 'DOI','keywords', 'Abstract' for every publications.
Those informations (Metadata) are first retrieved from the pdf itself and alternatively using Pubmed API. 
If an information could not be retrieved neither from the pdf nor from pubmed, None or a empty list is returned.
"""
import os
import re
import glob
from io import StringIO, BytesIO
import numpy as np
import pandas as pd
from collections import defaultdict
# progress bar
from tqdm import tqdm

# Homemade scripts
from utils import *
from pubmed import *
from text_mining import *
from csv_to_json import save_to_json

# PDF parser 
from tika import parser
from pdfrw import PdfReader

# format 
import json
from bs4 import BeautifulSoup

# log file
import logging


def extract_info(pdf_path, seq_id=1): 
    #-------------------#
    # Publi_ID creation #
    #-------------------#
    Publi_ID = "Pub_{:06d}".format(seq_id)
    
    #---------------------------------------------------------------------------------------------------------------#
    #  Metadata retrieve from the pdf - Authors, Title, Journal, Year, Abstract, Keywords, Journal, Pages, doi      #    
    #---------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    #  Metadata from pubmed  : Authors, Title, Journal, Year, Abstract, Keywords, Journal, Pages, doi      #    
    #------------------------------------------------------------------------------------------------------#
    """
    If the publication is availbale on pubmed, get the metadata from there using the pmid
    Else we keep those retrieved from the pdf 
    """
    METADATA_pdf = extract_metadata_pdf(pdf_path)
    Title = METADATA_pdf['Title']
    if not Title:
        Title = os.path.basename(pdf_path).split('_')[1].replace('.pdf','')
    DOI = METADATA_pdf['DOI']
    # IDs (pmid, pmcid) retrieving from pubmed API query using the publication title
    # get the pmid via pubmed API query using publication title 
    pmid = get_pmid(Title) 
    pmcid = get_pmcid_and_doi(pmid)[0]
    # if it failed try  DOI convertion using pubmed API 
    try :
        if not pmid:
            pmid, pmcid = convert_doi_to_pmid_and_pmcid(DOI)
        if not pmid:
            raise Exception
    except Exception:
        print("Publication not available on Pubmed or Title/DOI failed to be retrieved from the pdf.")
        logging.info('Publication not available on Pubmed or Title/DOI failed to be retrieved from the pdf.')
    # Store all informations retrieved in METADATA dictionnary 
    if pmid:
        METADATA = get_metadata_pubmed(pmid) 
        print('Publication available on PUBMED !')
        logging.info('Publication available on PUBMED !')
    else : 
        METADATA = METADATA_pdf  

    # Add Publi_ID and page count 
    METADATA['Publi_ID']=Publi_ID
    page_count = METADATA_pdf['Pages']
    METADATA['Pages']= page_count

    # Checking 
    # print("----PUBMED METADATA-----")
    # print(METADATA)
    # print("----NEXT PUBLI-----")

    #-------------------#
    # Total Word Count  #
    # (text_mining.py)  #
    #-------------------#
    # Functions defined in text_mining.py
    splitted_text, text, _ = pdf_to_text_pdfminer(pdf_path, 0 ,page_count, retstr = StringIO())
    clean_text, clean_tokens = text_tokenizer(text)
    total_word_count = len(clean_tokens)
    METADATA['Total_word_count']= total_word_count 

    #--------------------------------------------------------------------------#
    #     AAV terms  : count, Frequency and Related-publication references     #
    #--------------------------------------------------------------------------#

    # ADD the list of AAV terms found in the publication to the METADATA dictionnary 
    AAV_terms = find_AAV_terms(text) #  return an  dictionnary {'AAV_term1': count_1, 'AAV_term2': count_2,...} defined in word_count.py
    METADATA['AAV_terms'] = [key for key in AAV_terms.keys()]

    # TOTAL "AAV" count and frequency  
    AAV_count = sum([count for count in AAV_terms.values()])
    METADATA['AAV_count'] = AAV_count
    METADATA['Frequency'] =  AAV_count/total_word_count

    #  Loop aver all AAV terms to get their specific count, frequency and related references
    ## ADD the list of all AAV-related publication references  find in the publication to the METADATA dictionnary 
    Linked_references = []
    for AAV_term, AAV_count in AAV_terms.items():
        Count =  AAV_count
        Frequency = AAV_count/total_word_count

        # need to convert the pdf text to xml to split into its different lines  correctly without if not -->  refrences breaking
        xml_text = pdf_to_xml(pdf_path)

        # list of related-publication references specific to a AAV_term 
        try :
            related_references = find_AAV_term_related_publications_xml(xml_text, AAV_term)
        except:
            related_references = find_AAV_term_related_publications(splitted_text ,AAV_term)

        # get the list of all AAV-related publication references  find in the publication to the METADATA dictionnary 
        Linked_references += related_references
        # store into a dictionnary 
        #AAV_term_dict = {'Linked_references': Linked_references, 'AAV_term':AAV_term ,'AAV_term_count': AAV_count, 'Frequency_AAV_term':Frequency, 'Linked_AAV_references': related_references}
        AAV_term_dict = {'AAV_term':AAV_term ,'AAV_term_count': AAV_count, 'Frequency_AAV_term':Frequency, 'Linked_AAV_references': related_references}
        # Combine the 2 dictionnaries and add to a list
        AAV_data.append({**METADATA, **AAV_term_dict})
    METADATA['Linked_references']=Linked_references

    # IDs dictionnary:
    IDs = {'DOI': DOI, 'PMCID': pmcid, 'PMID': pmid, 'Publi_ID':Publi_ID,'Title':Title, 'PDF_name': File_name}

    return METADATA, AAV_data, IDs

#-----------------------------------------------------------------------------------------------------------------#
#-------------------------------------------- MAIN----------------------------------------------------------------#

if __name__ == "__main__":

    #--------------------#
    # Define directories #
    #--------------------#

    ## current directory
    dir = os.getcwd()

    ## project directory
    project_dir =  f'{dir}/../'

    ##  pdf list from './publications' folder
    pdf_dir = os.path.join(dir, '../publications')
    pdf_path_list = glob.glob(pdf_dir+'/*.pdf')

    ## output directory 
    save_dir = os.path.join(dir, '../output')
    ## log file
    
    LOG_FILENAME = "pdf_infos.log"
    log_path =os.path.join(dir, '..','output', 'log', LOG_FILENAME)
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('pdfminer').setLevel(logging.ERROR)


    #-----------------#
    # Initialization  #
    #-----------------#

    # csv output file names
    csv_name_Metadata = 'Publication_Metadata'
    csv_name_Infos = 'Publication_Informations'

    ## Create a dataframe df to store all results or open a existing one to add new data 
    try:
        df = pd.read_csv(project_dir + csv_name_Metadata + '.csv')
        IDs_table = pd.read_csv(project_dir + 'IDs_table.csv')
        AAV_df =pd.read_csv(project_dir + csv_name_Infos + '.csv')
        if df is not None:
            print('CSV files exits already, only new data will be append to them.')
            logging.info('CSV files exits already, only new data will be append to them.')
            
    except :
        df_column_names=['Publi_ID', 'Year', 'Authors', 'Title', 'Journal', 'DOI', 'Keywords','Pages','Abstract','Total_word_count','AAV_count','Frequency','Linked_references','AAV_terms']
        df  = pd.DataFrame(columns = df_column_names)
        ## Create a table of standard identifiers, i.e. PMCID, PMID, DOI,Publi_ID for each publications 
        IDs_table = pd.DataFrame(columns = ['DOI', 'PMCID', 'PMID', 'Publi_ID', 'Title','PDF_name'])
        ## Create a dataframe df to store all results  specific to a AAV term
        AAV_df_column_names = ['Publi_ID', 'Year', 'Authors', 'Title', 'Journal', 'DOI', 'Keywords','Pages','Abstract', 'Total_word_count','AAV_count','Frequency','AAV_terms','AAV_term','AAV_term_count','Frequency_AAV_term', 'Linked_AAV_references']
        AAV_df = pd.DataFrame(columns = AAV_df_column_names)
        

    # intermediaire storage 
    AAV_data = []
    AAV_dict = {}

    # variable Initialization
    if IDs_table['Publi_ID'].empty == False:
        seq_id = int(IDs_table['Publi_ID'].values[-1].split('_')[1]) # get the last Publi_ID of the list and extract the last number 
    else :
        seq_id =1

    #-----------------------#
    # Metadata retrieving   #
    #-----------------------#

    # Loop over all pdf files
    for pdf_path in tqdm(pdf_path_list):
        # File_name
        File_name = os.path.basename(pdf_path)
        print(File_name)
        
        if File_name in IDs_table['PDF_name'].values:
            print(f'Informations already extracted for this publication: {File_name}')
            logging.info(f'Informations already extracted for this publication: {File_name}')
            print('-----------NEXT PUBLICATION---------------------')
            logging.info('-----------NEXT PUBLICATION---------------------')

        else : 
            print(f'New publication : {File_name}')
            logging.info(f'New publication : {File_name}')
            METADATA, AAV_data, IDs =  extract_info(pdf_path, seq_id)
            logging.info('Information extraction : Metadata and AAV-related data')
                
            # Dataframe filling (add dictonnary as  row to dataframe)
            df = df.append(METADATA, ignore_index=True)
            IDs_table  = IDs_table.append(IDs,ignore_index=True)
            AAV_df = AAV_df.append(AAV_data, ignore_index=True)
            
            seq_id += 1
  
            print('-----------NEXT PUBLICATION---------------------')
            logging.info('-----------NEXT PUBLICATION---------------------')
    logging.info('--------------------Saving---------------------------')

    #-------------------------------#
    # save dataframes as csv files  #
    #-------------------------------#
    

    #df.drop(['PMID','PMCID'], axis=1, errors='ignore') # remove those columns if exist  before saving
    df.to_csv(save_dir + '/' + csv_name_Metadata + '.csv', index=False)
    IDs_table.to_csv(save_dir+ '/' + 'IDs_table.csv', index=False)
    AAV_df.to_csv( save_dir + '/' + csv_name_Infos + '.csv', index=False)
    print('csv files saved in {save_dir}')
    logging.info('csv files saved in {save_dir}')
    
    #-----------------------------------------------#
    #          Convert  to json                     #
    #     function defined in csv_to_json.py        #
    #-----------------------------------------------#

     # convert IDs_table to json
    json_output_0 = save_dir + '/' + 'IDs_table' + '.json'
    save_to_json(IDs_table,json_output_0)

 
    # convert df to json
    json_output_1 = save_dir + '/' +  csv_name_Metadata + '.json'
    save_to_json(df,json_output_1)

    # convert AAV_df to json 
    json_output_2 = save_dir + '/' +  csv_name_Infos + '.json'
    save_to_json(AAV_df,json_output_2)

    logging.info(f'json files saved in {save_dir}')
        
    # Checking  by printing dataframe first lignes
    print('---ID_tables---')
    print(IDs_table.head())
    print('---Publication_Metadata---')
    print(df.head())
    print('---Publication_Informations---')
    print(AAV_df.head())

 

      


