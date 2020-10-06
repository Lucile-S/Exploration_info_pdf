# Project 
The aim of this project is to extract informations from PDF format scientific publications related to AAV virus (adeno-associated virus).

Those informations are then incluted into a csv file and converted to json format in order to be transferred to a Neo4j database.

# Get Started 
The use of venv is recommended for creating virtual environment with all necessary packages listed in `requirements.txt`

```
python -m venv venv/ 
source ./venv/bin/activate # OSX - bash/zsh
.\venv\Scripts\activate # Windows - Powershell
pip install -r requirements.txt
```

# Get pdf informations
Run `python pdf_infos.py` will create a IDs_table.csv, Publication_Metadata.csv/.json and Publication_Informations.csv/.json containing informations about the pdf files present in the ./publications folder


# Scripts description :
