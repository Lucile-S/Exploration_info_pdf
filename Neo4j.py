#Neo4j.py
# package installation : pip install neo4j ;  pip install preprocessing; pip install py2neo
# ref : https://towardsdatascience.com/neo4j-cypher-python-7a919a372be7
from neo4j import __version__ as neo4j_version
from neo4j import GraphDatabase
import databaseconfig as cfg
import json

# check If Neo4j package is working
print(neo4j_version)

# parameters
uri = cfg.myneo4j["uri"]
user =   cfg.myneo4j["user"]
pwd =  cfg.myneo4j["passwd"]
role = cfg.myneo4j["role"]
db = cfg.myneo4j["database"]

# Firstly, we need to define a connection class to connect to the graph database.
class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response


# define paths
dir = os.getcwd()

json_1 = json.load(dir+'/Publication_Metadata.json')
json_2 = json.load(dir+'/Publication_Informations.json')

# Create an instance of connection with the parameters defined before.
conn = Neo4jConnection(uri=uri, user=user, pwd=pwd)

#create a query  : import json file and create nodes labeled as publication
with open(dir+'/Publication_Metadata.json') as data_file:
    json = json.load(data_file)
    query = """
        with {json} as data YIELD value as v
        CREATE(pub:publication) SET pub = v
  
        """
""" In neoJ4 desKtop
call apoc.load.json("file:///Publication_Metadata.json") YIELD value as v
CREATE(pub:publication) SET pub = v
"""
