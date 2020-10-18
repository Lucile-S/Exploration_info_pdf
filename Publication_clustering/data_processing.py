# data_processing.py 
import numpy as np
import pandas as pd
import os 
import re
import glob
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

#NLP packages
import string 
import stop_words

## NLTK
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.stem import PorterStemmer

## Spacy
import spacy
import scispacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS  as stop_words_spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex




import logging
LOG_FILENAME = "Absracts_preprocessing.log"
logging.basicConfig(filename='./log/'+ LOG_FILENAME, level=logging.INFO)


# pip install scispacy
# pip install <Model URL> :  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_md-0.2.5.tar.gz
# For the parser, we will use en_core_sci_lg. This is a model for processing biomedical, scientific or clinical text.
import en_core_sci_sm
nlp = en_core_sci_sm.load(disable=["tagger", "ner"])

# Initialize the stopwords
stop_words_ =stop_words.get_stop_words('fr')
stop_words_nltk = stopwords.words('english')
custom_stop_words = []
stopwords= list( 
    set( 
        list(stop_words_nltk)  + stop_words_ + list(stop_words_spacy) + custom_stop_words
    )
)

# Initialize ponctuation
punctuations = string.punctuation +  "«" + "»"+ "’" + '—'

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


def tokenize_and_lemmatize(text,  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # regex specific to AAV variants 
    text =re.sub('-|\(|\)',' ', text)
    # tokens + lower case
    tokens = [token.text.lower().strip() for token in nlp(text)]
    ## remove number
    # tokens = [word for word in tokens if word.isalpha()]
    # remove stop words and punctuations and keep words >= 1 letter
    tokens = [ token for token in tokens if token not in stopwords and token not in punctuations and len(token) > 1]
    #print(tokens)
    # lemmatization
    tokens=[nlp(token)[0].lemma_  if nlp(token)[0].lemma_ != "-PRON-" else token for token in tokens]
    #print(tokens)
    # join the tokens
    tokenized_text = " ".join([token for token in tokens])
    return tokenized_text 


#-----------------------------------------------------------#
#                            MAIN                           #
#-----------------------------------------------------------#

if __name__ == "__main__":

    #-----------------------------------------------------------#
    #                 Dataframe creation                        #
    #-----------------------------------------------------------#

    # define paths
    dir = os.getcwd()
    Abstract_dir = os.path.join(dir,'Abstracts')
   
    # load csv files containing publication abstracts retrieved from pubmed API 
    csv_files = glob.glob(f'{Abstract_dir}/*pubmed.csv', recursive=True)

    #############################################
    # Combine csv files into a unique dataframe #
    #############################################
    appended_data = []
    categories =[]
    for file in csv_files:
        print(file)
        logging.info(f'---{file}---')
        # retrieve category from the base name
        category = re.sub('_publications_pubmed.csv','', os.path.basename(file))
        print(f'Category : {category}')
        logging.info(f'Category : {category}')
        data = pd.read_csv(file)
        data['Category']=category
        print(data.shape)
        logging.info(f'Shape : {data.shape}')
        # store DataFrame in list
        appended_data.append(data)
        categories.append(category)

    all_data_df = pd.concat(appended_data)
    print(f'All data dataframe combined shape : {all_data_df.shape}')
    logging.info(f'All data dataframe combined shape : {all_data_df.shape}')
    print(all_data_df.head())

    # Handle Possible Duplicates 
    all_data_df = all_data_df.drop_duplicates(keep="first")
    print(f'Duplicate removed dataframe shape : {all_data_df.shape}')
    logging.info(f'Duplicate removed dataframe shape : {all_data_df.shape}')

    # Remove publications with no abstracts
    missing_abstracts = all_data_df['Abstract'].isnull().sum()
    print(f'There are {missing_abstracts} missing abstracts')
    all_data_df =all_data_df.dropna(how='any', subset=['Abstract'])
    print(f'Dataframe shape after removed rows with missing abstracts : {all_data_df.shape}')
    logging.info(f'Dataframe shape after removed rows with missing abstracts : {all_data_df.shape}')


    try:
        # To append  more data into an existing file
        df = pd.read_csv(f'{Abstract_dir}/Processed_Abstracts.csv')
        for index, row in all_data_df.iterrows():
            if row['PMID'] not in df.values:
                row_data = all_data_df.iloc[index].to_dict('records')
                abstract = all_data_df.iloc[index]['Abstract']
                Abstract_word_count = len(abstract.strip().split())
                AAV_terms = [key for key in find_AAV_terms(abstract).keys()] 
                AAV_count = sum([value for value in find_AAV_terms(abstract).values()]) 
                Processed_abstract = tokenize_and_lemmatize(abstract)
                supp_data = {'Abstract_word_count':Abstract_word_count,'AAV_terms':AAV_terms,'AAV_count':AAV_count,'Processed_abstract':Processed_abstract}
                df.append({**row_data, **supp_data})
                df = df.dropna(how='any', subset=['Processed_abstract'])
    except : 
        df = all_data_df.copy()
        ########################################
        #            Word Count                #
        ########################################

        # Abstract Word count 
        df['Abstract_word_count'] = df['Abstract'].apply(lambda x: len(x.strip().split())) 

        #-----------------------------------------------------------#
        #                  Preprocessing                            #
        #-----------------------------------------------------------#

        # AAV_term finding 
        df['AAV_terms'] = df['Abstract'].apply(lambda x:  [key for key in find_AAV_terms(x).keys()] )
        df['AAV_count'] = df['Abstract'].apply(lambda x:  sum([value for value in find_AAV_terms(x).values()]) )
     
        # tokenize et lemmatize 
        tqdm.pandas()
        df["Processed_abstract"] = df["Abstract"].progress_apply(tokenize_and_lemmatize)
        df = df.dropna(how='any', subset=['Processed_abstract'])

    # quick look
    print(df.head())
    print('dataframe shape: ', df.shape)
    
    # distribution of Abstract word count
    #sns.displot(df['Abstract_word_count'])
    print(df['Abstract_word_count'].describe())
    
    # check for no processed abstract
    print("Missing Processed Abstract rows:")
    print(df[df['Processed_abstract'].isnull()])
    df = df.dropna(how='any', subset=['Processed_abstract'])
    logging.info(f'Dataframe shape after removed no processed abstracts : {df.shape}')

    # save to csv 
    df.to_csv('{Abstract_dir}/Processed_Abstracts.csv', index=False)

