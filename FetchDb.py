import requests
import re
import os
import pandas as pd
from IPython.display import display

df = []
column_widths = [(0,4),(6,13),(15,18),(20,27),(29,32),(41,47),(49,55),(57,63),(65,71),(73,79),(81,87),(89,95),(97,102)]
column_names = ["WBANNO","UTC_DATE","UTC_TIME","LST_DATE","LST_TIME","LONGITUDE","LATTITUDE","T_CALC","T_HR_AVG","T_MAX","T_MIN","P_CALC","SOLARAD"]

def add_text_to_db(text):
	new_df = pd.read_fwf(text, colspecs=column_widths, names=column_names)
	df.append(new_df)

def download_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from URL: {e}")
        return None

def retrieve_data_from_directory(directory_url):
    try:
        response_text = download_text_from_url(directory_url)
        if response_text:
            # Use a simple regex to extract filenames from table data
            files = re.findall(r'td><a href="([^"]+)">', response_text)
            # Filter out directories (those ending with '/')
            files = [file for file in files if not file.endswith('/')]
            return files
        else:
            print("Empty response received.")
            return None
    except Exception as e:
        print(f"Error fetching directory listing: {e}")
        return None

def main(directory_url,year):
    directory_contents = retrieve_data_from_directory(directory_url)
    if directory_contents:
        for item in directory_contents:
            item_url = directory_url + "/" + item
            if item.endswith('.txt'):
                print(f"Processing text file: {item}")
                text_data = download_text_from_url(item_url)
                if text_data:
                	# Recreate database locally
                    file_path = "db/"+str(year)
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    with open(file_path+"/"+str(item), 'w') as file:
                        file.write(text_data)
                    add_text_to_db(file_path+"/"+str(item))
                else:
                    print(f"Failed to download text data from file: {item}")
            elif '.' not in item:  # If it's a directory (assuming no dots in directory names)
                print(f"Descending into directory: {item}")
                main(item_url)
            else:
                print(f"Ignoring non-text file: {item}")


if __name__ == "__main__":
    startYear = 2013
    endYear = 2024
    for i in range(startYear,endYear):
    	print("Parsing "+str(i))
    	main("https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/"+str(i)+"/",i)
    pd.concat(df).to_csv("db.csv")
