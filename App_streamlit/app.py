# -*- coding: utf-8 -*-
# homemade package
import sys
sys.path.append('../src')
from pubmed import *
from utils import *
from csv_to_json import *
from text_mining import *
from pdf_infos import extract_info


# Api package 
import streamlit as st

# basic packages
import numpy as np
import pandas as pd
import logging
import datetime
import dateutil.parser
import os 
import re
import glob
from  collections import Counter
from os import walk
from pathlib import Path



def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:orange ; padding:15px">
    <h2 style = "color:white; text_align:center;"> {main_txt} </h2>
    <p style = "color:white; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def display_header(header):
    """
    function to display minor headers at user interface main pannel 
    Parameters
    ----------
    header: str -> the major text to be displayed
    """
    #view clean data
    html_temp = f"""<h4 style = "color:grey;text_align:center;"> {header} </h5>"""
    st.markdown(html_temp, unsafe_allow_html = True)


# page setting
display_app_header(main_txt='AAV-related PDF Information Extraction Tool',sub_txt='Find pdf Metadata and AAV terms and store the information retrieved to csv and json files : "IDs_table", "Publication_Metadata" and "Publication_Informations"')
st.set_option('deprecation.showfileUploaderEncoding', False)


# get the current directory 
dir = os.getcwd()
pdf_dir = os.path.join(dir, '../publications')





def folder_selector(dir):
    st.sidebar.title('PDF Selector')
    # current directory path 
    path = Path(dir)
    # list of folders in the current directory parent
    dir_names= os.listdir(path.parent)
    # get absolute path 
    parent_path = path.parent.resolve()
    # pdf containg folder choosing 
    select_box_options = ['<selected>']
    select_box_options += dir_names
    selected_dir = st.sidebar.selectbox('Select a folder that contains your pdf files...',select_box_options,0)
    #get absolute path 
    selected_dir_path = os.path.join(parent_path, selected_dir)
    st.sidebar.write('The folder selected is :', selected_dir_path)
    return selected_dir_path

def upload_single_pdf(folder_path='.'):
    uploaded_pdf = st.sidebar.file_uploader('...or choose a pdf file to upload', type="pdf", key=0)
    #st.sidebar.write('The pdf uploaded is :', uploaded_pdf)
    return uploaded_pdf

def IDs_table_selector(folder_path='.'):
    st.sidebar.title('IDs Table Selector')
    uploaded_IDs_table = st.sidebar.file_uploader('If you have already an IDs correspondance table "IDs_table.csv" in which you would like to add to add new publication into, select it:', type=["csv","xlsx"], key=1)
    #st.sidebar.write('The IDs table uploaded is :', uploaded_IDs_table)
    return uploaded_IDs_table


def df_selector(folder_path='.'):
    st.sidebar.title('Publication Metadata file Selector')
    uploaded_df = st.sidebar.file_uploader('If you have already a "Publication_Metadata.csv" file in which you would like to add new publication into, select it:', type=["csv","xlsx"], key=2)
    #st.sidebar.write('The Publication Infos file uploaded is :', uploaded_df)
    return uploaded_df

def AAV_df_selector(folder_path='.'):
    st.sidebar.title('Publication Infos file Selector')
    uploaded_df_AAV = st.sidebar.file_uploader('If you have already a "Publication_Informations.csv" in which you would like to new publication into, select it:', type=["csv","xlsx"], key=2)
    #st.sidebar.write('The Publication Infos file uploaded is :', uploaded_df)
    return uploaded_df_AAV



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
  
    #If the publication is availbale on pubmed, get the metadata from there using the pmid
    #Else we keep those retrieved from the pdf 

    METADATA_pdf = extract_metadata_pdf(pdf_path)
    Title = METADATA_pdf['Title']
    if not Title:
        Title = os.path.basename(pdf_path).split('_')[1].replace('.pdf','')
    DOI = METADATA_pdf['doi']
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
        st.write("Publication not available on Pubmed or Title/DOI failed to be retrieved from the pdf. Please rename the pdf like the following exemple : '2020_Development of AAV Variants with Human Hepatocyte Tropism and Neutralizing Antibody Escape Capacity' ")

    # Store all informations retrieved in METADATA dictionnary 
    if pmid:
        METADATA = get_metadata_pubmed(pmid) 
        st.write('Publication available on PUBMED !')
    else : 
        METADATA = METADATA_pdf  

    # Add Publi_ID, IDs and page count 
    METADATA['PMID']=pmid
    METADATA['PMCID']=pmcid
    page_count = METADATA_pdf['Pages']
    METADATA['Pages']= page_count

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
            related_references = find_AAV_term_related_publications(xml_text, AAV_term)
        except:
            related_references = find_AAV_term_related_publications2(splitted_text ,AAV_term)

        # get the list of all AAV-related publication references  find in the publication to the METADATA dictionnary 
        Linked_references += related_references
        # store into a dictionnary 
        #AAV_term_dict = {'Linked_references': Linked_references, 'AAV_term':AAV_term ,'AAV_term_count': AAV_count, 'Frequency_AAV_term':Frequency, 'Linked_AAV_references': related_references}
        AAV_term_dict = {'AAV_term':AAV_term ,'AAV_term_count': AAV_count, 'Frequency_AAV_term':Frequency, 'Linked_AAV_references': related_references}
        # Combine the 2 dictionnaries and add to a list
        AAV_data.append({**METADATA, **AAV_term_dict})

    # IDs dictionnary:
    IDs = {'DOI': DOI, 'PMCID': pmcid, 'PMID': pmid, 'Publi_ID':Publi_ID,'Title':Title, 'PDF_name': File_name}

    return METADATA, AAV_data, IDs


def save_file(IDs_table,df,AAV_df):
    #save csv files
    csv_name = 'Publication_Metadata'
    csv_name_2 = 'Publication_Informations'
    save_dir = pdf_dir + '/../'
    #st.header('CSV file creation')
   
    #-------------------------------#
    # save dataframes as csv files  #
    #-------------------------------#

    #df.drop(['PMID','PMCID'], axis=1, errors='ignore') # remove those columns if exist  before saving
    df.to_csv(save_dir + csv_name + '.csv', index=False)
    IDs_table.to_csv(save_dir+'IDs_table.csv', index=False)
    AAV_df.to_csv( save_dir + csv_name_2 + '.csv', index=False)
    #st.write(f'csv files saved in {save_dir + csv_name}')

def convert_and_save_to_json(df,AAV_df): 
    csv_name = 'Publication_Metadata'
    csv_name_2 = 'Publication_Informations'
    pdf_dir = os.path.join(dir, '../publications')
    save_dir = pdf_dir + '/../'
 
    # convert df to json
    json_output_1 = save_dir +  csv_name + '.json'
    save_to_json(df,json_output_1)

    # convert AAV_df to json 
    json_output_2 = save_dir +  csv_name_2 + '.json'
    save_to_json(AAV_df,json_output_2)


################################
###############################

selected_dir_path=folder_selector(dir=os.getcwd())
uploaded_pdf=upload_single_pdf()
uploaded_IDs_table = IDs_table_selector()
uploaded_df = df_selector()
uploaded_AAV_df = AAV_df_selector()



# ID TABLE 
if uploaded_IDs_table is not None:
    IDs_table_file = IDs_table_selector()
    IDs_table = pd.read_csv(IDs_table_file)
else:
    IDs_table = pd.DataFrame(columns = ['DOI', 'PMCID', 'PMID', 'Publi_ID', 'Title','PDF_name'])

# df 
df_column_names=['Publi_ID', 'Year', 'Authors', 'Title', 'Journal', 'DOI','PMID','PMCID', 'Keywords','Pages','Abstract','Total_word_count','AAV_count','Frequency','Linked_references','AAV_terms']
if uploaded_df is not None:
    uploaded_df_file = df_selector()
    df = pd.read_csv(uploaded_df_file)
else:
    df = pd.DataFrame(columns = df_column_names)


# AAV_df 
AAV_df_column_names = ['Publi_ID', 'Year', 'Authors', 'Title', 'Journal', 'DOI','PMID','PMCID', 'Keywords','Pages','Abstract', 'Total_word_count','AAV_count','Frequency','AAV_terms','AAV_term','AAV_term_count','Frequency_AAV_term', 'Linked_AAV_references']
if uploaded_AAV_df is not None:
    uploaded_AAV_df_file = df_selector()
    AAV_df = pd.read_csv(uploaded_df_file)
else:
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
#       INFOS           #
#-----------------------#

if selected_dir_path != '<selected>':
    pdf_paths = glob.glob(selected_dir_path +'/*.pdf')
    # loop over pdf files 
    for pdf_path in pdf_paths:
        st.subheader("PDF file uploaded :")
        File_name = os.path.basename(pdf_path)
        st.write(f'<font color=green>{File_name}</font>', unsafe_allow_html=True)

        if File_name in IDs_table['PDF_name'].values:
            st.write(f'Informations already extracted for this publication: {File_name}')
            st.write(f" DOI: {IDs_table['DOI']} ; PMID: {IDs_table['PMID']} ; PMCID: {IDs_table['PMCID']}")
            st.write('-------------NEXT PUBLICATION------------------')
        else : 
    
            st.subheader("Fetching and collectiong PDF metadata and AAV informations ...")
            METADATA, AAV_data, IDs =  extract_info(pdf_path, seq_id)
            Title =  METADATA['Title']
            st.write(f'Title : {Title}')
            st.write(f" DOI: {METADATA['DOI']} ; IDs_table: {METADATA['PMID']} ; PMCID: {METADATA['PMCID']}")

            # Dataframe filling (add dictonnary as  row to dataframe)
            df = df.append(METADATA, ignore_index=True)
            IDs_table  = IDs_table.append(IDs,ignore_index=True)
            AAV_df = AAV_df.append(AAV_data, ignore_index=True)
            seq_id += 1
            st.write('**--------------------NEXT PUBLICATION---------------------**')

    if df.empty == False :   
        st.subheader('Save to csv and convert to json...')
        save_file(IDs_table,df,AAV_df)
        convert_and_save_to_json(df,AAV_df)  
        st.write('Done!')
        # checking
        st.subheader('Quick view of the files')
        st.write('-- IDs table --')
        st.write(IDs_table.head())
        st.write('-- Publication Metadata --')
        st.write(df.head())
        st.write('-- Publication Informations --')
        st.write(AAV_df.head())




if uploaded_pdf != None:
    st.header("PDF files uploaded : ")
    st.write(f'<font color=green>{uploaded_pdf}</font>', unsafe_allow_html=True)

    if File_name in IDs_table['PDF_name'].values:
        st.write(f'Informations already extracted for this publication: {File_name}')
        st.write(f" DOI: {IDs_table['DOI']} ; PMID: {IDs_table['PMID']} ; PMCID: {IDs_table['PMCID']}")
    else : 
        st.write(f'{File_name}')
        st.subheader("fetching and collectiong PDF metadata and AAV informations ...")
        METADATA, AAV_data, IDs =  extract_info(pdf_path, seq_id)
        Title =  METADATA['Title']
        st.write(f'Title : {Title}')
        st.write(f" DOI: {METADATA['DOI']} ; IDs_table: {METADATA['PMID']} ; PMCID: {METADATA['PMCID']}")
        # Dataframe filling (add dictonnary as  row to dataframe)
        df = df.append(METADATA, ignore_index=True)
        IDs_table  = IDs_table.append(IDs,ignore_index=True)
        AAV_df = AAV_df.append(AAV_data, ignore_index=True)
        seq_id += 1
        st.write('-----------NEXT PUBLICATION-----------------')
        #save csv files
        save_file(IDs_table,df,AAV_df)
        # Convert  to json
        convert_and_save_to_json(df,AAV_df)
        # checking
        st.write('-- IDs table --')
        st.write(IDs_table.head())
        st.write('-- Publication Metadata --')
        st.write(df.head())
        st.write('-- Publication Informations --')
        st.write(AAV_df.head())




