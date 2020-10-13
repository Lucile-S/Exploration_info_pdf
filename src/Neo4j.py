#Neo4j.py
# package installation : pip install neo4j ;  pip install preprocessing; pip install py2neo
# ref : https://towardsdatascience.com/neo4j-cypher-python-7a919a372be7
from neo4j import __version__ as neo4j_version
from neo4j import GraphDatabase
import configparser
import json
import os
import glob
import logging

# check If Neo4j package is working
print(neo4j_version)

# parameters
config = configparser.ConfigParser()
config.read(neo4j_database.ini)
uri  = config['myneo4j']['uri']
user = config['myneo4j']['user']
pwd = config['myneo4j']['passwd']

# Define a connection class to connect to the graph database
class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd), encrypted=False)
            print('Connection done')
            logging.info('Connection done')
        except Exception as e:
            print("Failed to create the driver:", e)
            logging.info("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, json, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query, json=json))
            print("Query done")
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

def import_json(json_path, label):
    file_name = os.path.basename(json_path)
    with open(json_path,'r') as file:
        json_file = json.load(file)
        query ="""WITH $json as data
        UNWIND data as v
        CREATE(n: %s) SET n = v"""% label
        try : 
            conn.query(query,json_file)
            print(f'{file_name} importation succeeded with {label} as label')
            logging.info(f'{file_name} importation succeeded with {label} as label')
        except:
            print(f'{file_name} importation failed')
            logging.info(f'{file_name} importation failed')

if __name__ == '__main__':

    # define paths
    dir = os.getcwd()
    json_dir = os.path.join(dir,'..','output')

    # json file list
    json_files = glob.glob(json_dir+'/*json')
    #print(json_files)

    # Create an instance of connection with the parameters defined before.
    conn = Neo4jConnection(uri=uri, user=user, pwd=pwd)

    # Create a query  : import json file and create nodes labeled as <label>
    json_path = json_files[0]
    label = "PublicationIDTest"

    # import data to neo4j
    import_json(json_path, label)



