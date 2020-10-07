import os
import re
import glob

import numpy as np
import pandas as pd
# format 
import json


# #-------------------------------#
# #    Convert to json           #
# #-------------------------------#

def save_to_json(df,json_output):
    try : 
        json_df = df.to_json(json_output, orient='records', indent = 2 ,force_ascii=False)
        print('JSON conversion and file saving succeeded')
    except Exception as e:
        print("Conversion to json failed")
        print(e)

    try : 
        with open(json_output, "r") as f:
            content = json.load(f)
        with open(json_output, "w", encoding='utf-8') as fout:
            json.dump(content, fout,  ensure_ascii=False, indent=4 )
    except Exception as e:
        print("Conversion to json failed")
        print(e)

if __name__ == "__main__":
    # define current directory 
    dir = os.getcwd()

    # define csv and saving directory 
    save_dir = dir + '/../'
    csv_name_1 = 'Publication_Metadata.csv'
    csv_name_2 = 'Publication_Informations.csv'


    df1 = pd.read_csv(save_dir + csv_name_1 +'.csv')
    df2 = pd.read_csv(save_dir + csv_name_2 +'.csv')
    # function defined in csv_to_json.py 
    json_output_1 = save_dir +  csv_name_1 + '.json'
    save_to_json(df1,json_output_1)

    json_output_2 = save_dir +  csv_name_2 + '.json'
    save_to_json(df2,json_output_2)
