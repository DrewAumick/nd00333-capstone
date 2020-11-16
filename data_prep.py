import numpy as np
import pandas as pd
from azureml.core.workspace import Workspace
from azureml.core import Dataset
from azureml.core.datastore import Datastore
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.preprocessing import LabelEncoder

from kaggle.api.kaggle_api_extended import KaggleApi
kaggle_api = KaggleApi()
kaggle_api.authenticate()

# Searches the workspace for a dataset of the given key. If it doesn't find it, it creates the dataset from a CSV file
def get_dataset(ws, key="Malware Dataset"):   
    if key in ws.datasets.keys(): 
            dataset = ws.datasets[key] 
    else:
            dataset = create_dataset(ws)
            
    return dataset

# Download data from kaggle, create the Malware Dataset, and register to the Workspace
def create_dataset(ws):  
    kaggle_api.dataset_download_file('divg07/malware-analysis-dataset','data.csv')

    data = pd.read_csv(
            './data.csv.zip',
            compression='zip',
            sep='|'
        )

    # Clean dataset 
    data = clean_data(data)

    # Register Dataset in Workspace
    datastore = Datastore(ws)
    name = "Malware Dataset"
    description_text = "Malware DataSet for Udacity Capstone Project"
    dataset = TabularDatasetFactory.register_pandas_dataframe(data,
                               datastore,
                               name,
                               description=description_text)
    
    return dataset
    
# Clean the data    
def clean_data(df):
    LE = LabelEncoder()
    df['Name'] = LE.fit_transform(df['Name'])    
    df['md5'] = LE.fit_transform(df['md5'])
        
    return df

def main():
    ws = Workspace.from_config()
    
    df = get_DDoS_dataset(ws)

if __name__ == '__main__':
    main()