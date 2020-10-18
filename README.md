# Project - Part 1 : PDF information extraction
The aim of this project is to extract informations from PDF of research publications related to **AAV virus (adeno-associated virus)**.

Those informations are then incluted into a csv file and converted to json format in order to be transferred to a **neo4j database**.

<p align="center">
  <img src="Img/context.JPG" width="650" title="Context">
</p>
<p align="center">
  <img src="Img/context2.JPG" width="650" title="Context2">
</p>


# Get Started 
The use of venv is recommended for creating virtual environment with all necessary packages listed in `requirements.txt`

```
python -m venv venv/ 
source ./venv/bin/activate # OSX - bash/zsh
.\venv\Scripts\activate # Windows - Powershell
pip install -r requirements.txt
```

# Get pdf informations
Run `run pdf_infos.py` script will create:
- IDs_table.csv (reference correspondence table) - columns: DOI, PMCID, PMID, Publi_ID, Title, PDF_name.

- Publication_Metadata.csv/.json (a row for each publication) - columns: Publi_ID, Year, Authors, Title, Journal, DOI, PMID, PMCID, Keywords, Pages, Abstract, Total_word_count, AAV_count, Frequency, Linked_references, AAV_terms.

- Publication_Informations.csv/.json (a row for each AAV term found in the publications) - columns :  Publi_ID, Year, Authors, Title, Journal, DOI, PMID, PMCID, Keywords, Pages, Abstract, Total_word_count, AAV_count, Frequency, Linked_references, AAV_terms, AAV_term, AAV_term_count, Frequency_AAV_term, Linked_AAV_references.

Those files contain informations (Metadata and AAV-related informations) about the pdf files present in the ./publications folder.


# Streamlit PDF Information Extraction Application

Run ` streamlit run extraction_info_pdf.py` (pdf_infos.py corresponding application) to open the PDF information extraction tool. 
<p align="center">
  <img src="Img/app_2.JPG" width="650">
</p>


<p align="center">
  <img src="Img/app_3.JPG" width="650">
</p>

- Select a folder containing pdf files or select an unique pdf.
- Indicate if you want to upload existing csv files (IDs_table.csv, Publication_Metadata.csv, Publication_Informations.csv) in order to add new data to them.
- Choose to import json data to a Neo4j database by adding a configuration file (`neoj4_database.ini`, file example in src/ folder).
- Metadata (Title, Authors, DOI, PMID, PMCID, Keywords, Abstract, Journal, Year, Pdf Word Count) are retrieved using pdf manipulation packages (Fitz, pdfminer, Tika), regex patterns and pubmed API.
- AAV terms, frequency and their linked publication references are retrieved using pdf manipulation packages (Fitz, pdfminer, Tika) and regex patterns
- Retrieved informations are stored into csv files : 
  - IDs_table.csv, 
  - Publication_Metadata.csv,
  - and Publication_Informations.csv.
- Json files are generated from those csv files.
- Csv and json files are saved into the extraction_info_pdf_output/ folder
- Data are pushed to Neo4j. 


<p align="center">
  <img src="Img/capture_Neo4j_test_example.jpg" width="650" title="Neo4j Example">
</p>



#  Project - Part 2 : Genomic Medicine Literature Clustering Clustering 
The aim of the second part of this project is to clustering similar research article abstracts together in order to see if it's possible to simplify the search for related publications. 

the steps for doing that are the following:
1. fetch some abstracts,
1. Preprocessing of each abstract,
1. represent each abstracts as a vector,
1. perform k-means clustering,
1. perform t-SNE dimensionality reduction for visualization,
1. and perform topic modelling to find the keywords of each cluster.

## 1. Fetch publication abstracts 
Publication abstracts from different Genomic Medicine related topics were downloaded using the Pubmed Api. 
Selected query topics are:
- Adeno-Associated-Virus
- Epigenetics
- Gene Therapy
- Genome Engineering
- Immunology
- Post-Translational Modification
- Regulatory Element
- Sequence
- Transfection
- Tropism
- Variant


For each query topic a csv file was created to store the abstract and the metadata of each publications. 
Those csv files can be find in the `Abstracts/` folder.

Around 4800 abstracts were collected in total by running `Publication_clustering/scraping_abstract_pubmed.py`.

## 1. Preprocessing
- Combine all csv files into a unique dataframe,
- Remove duplicate and missing abstracts,
- remove punctuations and stopwords,
- tokenize and lemmatize,
- save as `Processed_Abstracts.csv`

The preprocessing was performed by running `Publication_clustering/data_processing.py`.

## 1. Represent each abstracts as a vector
Each abstract was transformed into a feature vector using Term Frequency–inverse Document Frequency (TF-IDF). TF-IDF evaluates how relevant a word is to a document in a collection of documents.
The clustering was based off the TF-IDF matrix  where each row is a vector representation of a publication abstract. The maximum number of features was limited to the top 512 features.

## 1. K-means clustering
The TF-IDF matrix was used as input for the k-means algorithm.
First, to determine the best k number of clusters, the silhouette score and the sum of squared distances from each point to its assigned center were computed at different k values.

<p align="center">
  <img src="Img/Silhouette_Method_For_Optimal_k.png" width="650" title="How many clusters">
</p>

For 12 query topics, the optical k determined was **14**. 

<p align="center">
  <img src="Img/nb_of_publications_per_cluster.png" width="650" title="Nb of publications per cluster">
</p>


<p align="center">
  <img src="Img/category_per_cluster.png" width="650" title="Category per cluster">
</p>












