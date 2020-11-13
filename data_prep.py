import numpy as np
import pandas as pd
from azureml.core.workspace import Workspace
from azureml.core import Dataset
from azureml.core.datastore import Datastore
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.preprocessing import LabelEncoder

# Searches the workspace for a dataset of the given key. If it doesn't find it, it creates the dataset from a CSV file
def get_DDoS_dataset(ws, key="DDoS Dataset"):   
    if key in ws.datasets.keys(): 
            dataset = ws.datasets[key] 
    else:
            dataset = create_DDoS_datasets(ws)
            
    return dataset

# Download data from kaggle, create the DDoS Dataset, clean it, and register both the base version and the cleaned version to Workspace
def create_DDoS_datasets(ws):  
    dtypes = {
        'Src IP': 'category',
        'Src Port': 'uint16',
        'Dst IP': 'category',
        'Dst Port': 'uint16',
        'Protocol': 'category',
        'Tot Fwd Pkts': 'uint32',
        'Tot Bwd Pkts': 'uint32',
        'Flow IAT Min': 'float32',
        'Init Bwd Win Byts': 'uint32',
        'Fwd Seg Size Min': 'uint32',
        'Label': 'category'
    }

    data = pd.read_csv(
            './final_dataset.csv',
            dtype=dtypes,
            parse_dates=['Timestamp'],
            usecols=[*dtypes.keys(), 'Timestamp'],
            engine='c',
            low_memory=True,
            na_values=np.inf
        )

    # Clean dataset 
    data = clean_data(data)

    # There are over 12 million rows in this orignal dataset. For this project, that much is crashing the VMs, so only sampling 25% 
    data = data.sample(frac=0.25)

    # Register Base Dataset in Workspace
    datastore = Datastore(ws)
    name = "DDoS Dataset"
    description_text = "DDoS DataSet for Udacity Capstone Project"
    dataset = TabularDatasetFactory.register_pandas_dataframe(data,
                               datastore,
                               name,
                               description=description_text)
    
    return dataset
    
# Clean the data    
def clean_data(df):
    LE = LabelEncoder()
    df['Src IP'] = LE.fit_transform(df['Src IP'])    
    df['Dst IP'] = LE.fit_transform(df['Dst IP'])
    
    df['Timestamp'] = df['Timestamp'].apply(lambda s: s.value)    
        
    # Drop columns that don't impact classification 
    #colsToDrop = np.array(['Flow Byts/s', 'Flow Pkts/s','Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])
    
    # Drop columns where missing values are more than 50% 
    missing = df.isna().sum()
    missing = pd.DataFrame({'count': missing, '% of total': missing/len(df)*100}, index=df.columns)
    colsToDrop = missing[missing['% of total'] >= 50].index.values

    # Drop rows where a column missing values are no more than 5%
    dropnaCols = missing[(missing['% of total'] > 0) & (missing['% of total'] <= 5)].index.values
    
    #dropnaCols = np.union1d(dropnaCols, ['Flow Byts/s', 'Flow Pkts/s'])
    df.dropna(subset=dropnaCols, inplace=True)

    df.drop(columns=colsToDrop, inplace=True)
    
    # convert the label to 1 for ddos or 0 for benign
    df["Label"] = df.Label.apply(lambda s: 1 if s == "ddos" else 0)  

    #df = df.fillna(0)  
        
    return df

def main():
    ws = Workspace.from_config()
    
    df = get_DDoS_dataset(ws)

if __name__ == '__main__':
    main()