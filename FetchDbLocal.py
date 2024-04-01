import requests
import re
import os
import pandas as pd
from IPython.display import display

df = []
column_widths = [(0,5),(6,14),(15,19),(20,28),(29,33),(41,48),(49,56),(57,64),(65,72),(73,80),(81,88),(89,96),(97,103)]
column_names = ["WBANNO","UTC_DATE","UTC_TIME","LST_DATE","LST_TIME","LONGITUDE","LATTITUDE","T_CALC","T_HR_AVG","T_MAX","T_MIN","P_CALC","SOLARAD"]

def add_text_to_db(text):
	new_df = pd.read_fwf(text, colspecs=column_widths, names=column_names)
	df.append(new_df)

def parse_directory(dir):
    for file in os.listdir(dir):
        print(f"parsing {file}")
        add_text_to_db(dir+file)


if __name__ == "__main__":
    startYear = 2023
    endYear = 2024
    for i in range(startYear,endYear):
    	print("Parsing "+str(i))
    	parse_directory("db/"+str(i)+"/")
    pd.concat(df).to_csv("db/minidb.csv")
