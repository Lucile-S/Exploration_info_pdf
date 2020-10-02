# -*- coding: utf-8 -*-
# homemade package
from pubmed import *

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
import codecs
from pdfminer.high_level import extract_text
from pathlib import Path

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
nlp = English()


# get the current directory 


# page setting
st.title("PDF information extraction and json convertion (using Pubmed API)")
st.set_option('deprecation.showfileUploaderEncoding', False)

dir = os.getcwd()

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
    uploaded_IDs_table = st.sidebar.file_uploader('If you have a IDs correspondance table (e.g. IDs_table.csv) in which you would like to add publication into, select it:', type=["csv","xlsx"], key=1)
    #st.sidebar.write('The IDs table uploaded is :', uploaded_IDs_table)
    return uploaded_IDs_table

def df_selector(folder_path='.'):
    st.sidebar.title('Publication Infos file Selector')
    uploaded_df = st.sidebar.file_uploader('If you have a Publication Info file (e.g. Publication_Informations.csv) in which you would like to add publication info into, select it:', type=["csv","xlsx"], key=2)
    #st.sidebar.write('The Publication Infos file uploaded is :', uploaded_df)
    return uploaded_df


def metadata():
    title, doi = get_pdf_title_and_doi(pdf_path)
    # if doi is None:
    #     st.write(f":warning: No DOI retrieves for {title} from the pdf")
    try : 
        pmid = get_pmid(title)
        pmcid, doi  = get_pmcid_and_doi(pmid)
    except:
        pmid, pmcid = get_pmid_and_pmcid(doi)
    st.write(title)
    st.write(f" DOI: {doi} ; PMID: {pmid} ; PMCID: {pmcid}")
    # Add as row to a Dataframe df if ID not present already
    if pmid not in IDs_table['pmid'].values: 
        IDs_table  = IDs_table.append({'DOI': doi, 'PMCID': pmcid, 'PMID': pmid, 'pdf_name': os.path.basename(pdf_path)},ignore_index=True)

    st.markdown("fetching and collectiong pdf metadata (title, ids, authors, year of publication, keywords, journal, abstract) ...")
    metadata = get_metadata(pmid)
    # Add as row to a Dataframe df if ID not present already
    if metadata['pmid'] not in df['pmid'].values: 
        df = df.append(metadata, ignore_index=True)
    st.write("--------")

def save_file(IDs_table,df):
    #save csv files
    st.text("**Saving csv files in the project directory**:")
    #st.markdown("Creation or Update of 'IDs_table.csv' the correspondance table between IDs (doi, pmid, pmcid) ")
    IDs_table.to_csv(dir + '/../'+ 'IDs_table.csv', index=False)
    #st.markdown("Creation or update of 'Publication_Informations.csv' file")
    df.to_csv(dir + '/../' + 'Publication_Informations.csv', index=False)
    st.markdown("Save csv files done...")
    st.write(df.head())

if __name__ == "__main__":
    selected_dir_path=folder_selector(dir=os.getcwd())
    uploaded_pdf=upload_single_pdf()
    uploaded_IDs_table = IDs_table_selector()
    uploaded_df = df_selector()

    if uploaded_IDs_table is not None:
        IDs_table_file = IDs_table_selector()
        IDs_table = pd.read_csv(IDs_table_file)
    else:
        IDs_table = pd.DataFrame(columns = ['DOI','PMCID','PMCID','pdf_name'])

    column_names=['pmid','pmcid','Year','Authors','Title', 'Journal', 'DOI','keywords', 'Abstract']

    if uploaded_df is not None:
        uploaded_df_file = df_selector()
        df = pd.read_csv(uploaded_df_file)
    else:
        df  = pd.DataFrame(columns = column_names)


    if selected_dir_path != '<selected>':
        pdf_paths = glob.glob(selected_dir_path +'/*.pdf')
        st.subheader("PDF files uploaded : ")
        # loop over pdf files 
        for pdf_path in pdf_paths:
            title, doi = get_pdf_title_and_doi(pdf_path)
            # if doi is None:
            #     st.write(f":warning: No DOI retrieves for {title} from the pdf")
            try : 
                pmid = get_pmid(title)
                pmcid, doi  = get_pmcid_and_doi(pmid)
            except:
                pmid, pmcid = get_pmid_and_pmcid(doi)
            st.write(title)
            st.write(f" DOI: {doi} ; PMID: {pmid} ; PMCID: {pmcid}")
            # Add as row to a Dataframe df if ID not present already
            if pmid not in IDs_table['pmid'].values: 
                IDs_table  = IDs_table.append({'DOI': doi, 'PMCID': pmcid, 'PMID': pmid, 'pdf_name': os.path.basename(pdf_path)},ignore_index=True)

            st.markdown("fetching and collectiong pdf metadata (title, ids, authors, year of publication, keywords, journal, abstract) ...")
            metadata = get_metadata(pmid)
            # Add as row to a Dataframe df if ID not present already
            if metadata['pmid'] not in df['pmid'].values: 
                df = df.append(metadata, ignore_index=True)
            st.write("--------")
        save_file(IDs_table,df)

    if uploaded_pdf != None:
        title, doi = get_pdf_title_and_doi(pdf_path)
        # if doi is None:
        #     st.write(f":warning: No DOI retrieves for {title} from the pdf")
        try : 
            pmid = get_pmid(title)
            pmcid, doi  = get_pmcid_and_doi(pmid)
        except:
            pmid, pmcid = get_pmid_and_pmcid(doi)
        st.write(title)
        st.write(f" DOI: {doi} ; PMID: {pmid} ; PMCID: {pmcid}")
        # Add as row to a Dataframe df if ID not present already
        if pmid not in IDs_table['pmid'].values: 
            IDs_table  = IDs_table.append({'DOI': doi, 'PMCID': pmcid, 'PMID': pmid, 'pdf_name': os.path.basename(pdf_path)},ignore_index=True)

        st.markdown("fetching and collectiong pdf metadata (title, ids, authors, year of publication, keywords, journal, abstract) ...")
        metadata = get_metadata(pmid)
        # Add as row to a Dataframe df if ID not present already
        if metadata['pmid'] not in df['pmid'].values: 
            df = df.append(metadata, ignore_index=True)
        st.write("--------")
        #save csv files
        save_file(IDs_table,df)
 





