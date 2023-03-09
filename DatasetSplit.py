import os
import pandas as pd
import shutil

# Only consider the classes: healthy (0) and COVID-19 (1)
ORIGINAL_PATH = './coughvid_20211012/spec'
SPLIT_PATH = './dataset_baseline'

def split_tool(metadata, dataset_path, dataset_original_path):
    '''
    metadata: DataFrame of the dataset e.g. metadata_train
    dataset_path: Root directory for the dataset after spliting e.g. ./dataset_baseline/train
    dataset_original_path: 
    '''
    healthy_data = metadata[metadata['status'] == 'healthy']
    COVID19_data = metadata[metadata['status'] == 'COVID-19']
    healthy_uuids = list(healthy_data['uuid'])
    for uuid in healthy_uuids:
        source_file_path = os.path.join(dataset_original_path, uuid + '.png')
        destination_file_path = os.path.join(dataset_path, '0', uuid + '.png')
        shutil.copyfile(source_file_path, destination_file_path)
    covid_uuids = list(COVID19_data['uuid'])
    for uuid in covid_uuids:
        source_file_path = os.path.join(dataset_original_path, uuid + '.png')
        destination_file_path = os.path.join(dataset_path, '1', uuid + '.png')
        shutil.copyfile(source_file_path, destination_file_path)

if __name__ == '__main__':
    dataset_original_path = ORIGINAL_PATH # e.g. ./coughvid_20211012/spec
    dataset_split_path = SPLIT_PATH # e.g. ./dataset_baseline
    # get the csv files for training, validation and test sets
    metadata_train = pd.read_csv(os.path.join(dataset_split_path, 'train.csv'))
    metadata_val = pd.read_csv(os.path.join(dataset_split_path, 'val.csv'))
    metadata_test = pd.read_csv(os.path.join(dataset_split_path, 'test.csv'))
    split_tool(metadata_train, os.path.join(dataset_split_path, 'train'), dataset_original_path)
    split_tool(metadata_val, os.path.join(dataset_split_path, 'val'), dataset_original_path)
    split_tool(metadata_test, os.path.join(dataset_split_path, 'test'), dataset_original_path)

    