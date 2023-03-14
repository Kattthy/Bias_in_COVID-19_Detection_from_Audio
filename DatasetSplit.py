import os
import pandas as pd
import shutil
import glob
from tqdm import tqdm

# Only consider the classes: healthy (0) and COVID-19 (1)
ORIGINAL_PATH = './coughvid_20211012/spec_chunk'
SPLIT_PATH = './dataset_baseline_chunk'

def split_tool(metadata, dataset_path, dataset_original_path):
    '''
    metadata: DataFrame of the dataset e.g. metadata_train
    dataset_path: Root directory for the dataset after spliting e.g. ./dataset_baseline/train
    dataset_original_path: './coughvid_20211012/spec_chunk'
    '''
    healthy_data = metadata[metadata['status'] == 'healthy']
    COVID19_data = metadata[metadata['status'] == 'COVID-19']
    Symptomatic_data = metadata[metadata['status'] == 'symptomatic']
    healthy_uuids = list(healthy_data['uuid'])
    for uuid in tqdm(healthy_uuids):
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            # source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '0', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)
    covid_uuids = list(COVID19_data['uuid'])
    for uuid in tqdm(covid_uuids):
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '1', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)
    symptomatic_uuids = list(Symptomatic_data['uuid'])
    for uuid in tqdm(symptomatic_uuids):
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '2', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)

def split_gender_tool(metadata, dataset_root_path, dataset_original_path):
    '''
    metadata: DataFrame of the dataset e.g. metadata_train
    dataset_root_path: Root directory for the dataset after spliting e.g. ./dataset_baseline
    dataset_original_path: 
    '''
    healthy_data = metadata[metadata['status'] == 'healthy']
    COVID19_data = metadata[metadata['status'] == 'COVID-19']
    Symptomatic_data = metadata[metadata['status'] == 'symptomatic']
    healthy_male_data = healthy_data[healthy_data['gender'] == 'male']
    healthy_female_data = healthy_data[healthy_data['gender'] == 'female']
    COVID_male_data = COVID19_data[COVID19_data['gender'] == 'male']
    COVID_female_data = COVID19_data[COVID19_data['gender'] == 'female']
    Symptomatic_male_data = Symptomatic_data[Symptomatic_data['gender'] == 'male']
    Symptomatic_female_data = Symptomatic_data[Symptomatic_data['gender'] == 'female']

    dataset_path = os.path.join(dataset_root_path, 'test-male')
    
    healthy_male_uuids = list(healthy_male_data['uuid'])
    for uuid in healthy_male_uuids:
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '0', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)
    
    COVID_male_uuids = list(COVID_male_data['uuid'])
    for uuid in COVID_male_uuids:
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '1', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)
    
    Symptomatic_male_uuids = list(Symptomatic_male_data['uuid'])
    for uuid in Symptomatic_male_uuids:
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '2', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)
    
    dataset_path = os.path.join(dataset_root_path, 'test-female')
    
    healthy_female_uuids = list(healthy_female_data['uuid'])
    for uuid in healthy_female_uuids:
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '0', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)
    
    COVID_female_uuids = list(COVID_female_data['uuid'])
    for uuid in COVID_female_uuids:
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '1', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)

    Symptomatic_female_uuids = list(Symptomatic_female_data['uuid'])
    for uuid in Symptomatic_female_uuids:
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            source_file_path = fileName
            destination_file_path = os.path.join(dataset_path, '2', fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)
    
    # for uuid in healthy_male_uuids:
    #     matching_files = glob.glob(dataset_original_path + uuid + "*")
    #     for fileName in matching_files:
    #         source_file_path = os.path.join(dataset_original_path, fileName)
    #         destination_file_path = os.path.join(dataset_path, '0', fileName)
    #         shutil.copyfile(source_file_path, destination_file_path)
    # covid_uuids = list(COVID19_data['uuid'])
    # for uuid in covid_uuids:
    #     matching_files = glob.glob(dataset_original_path + uuid + "*")
    #     for fileName in matching_files:
    #         source_file_path = os.path.join(dataset_original_path, fileName)
    #         destination_file_path = os.path.join(dataset_path, '1', fileName)
    #         shutil.copyfile(source_file_path, destination_file_path)

def copy_files(uuids, target_directory, dataset_original_path):
    count = 0
    for uuid in uuids:
        matching_files = glob.glob(dataset_original_path + '/' + uuid + "*")
        for fileName in matching_files:
            count += 1
            source_file_path = fileName
            destination_file_path = os.path.join(target_directory, fileName.split('/')[-1])
            shutil.copyfile(source_file_path, destination_file_path)

    return count
def split_age_tool(metadata, dataset_root_path, dataset_original_path):
    # <=20 20-40 40-60 >60 => 0,1,2,3
    healthy_data = metadata[metadata['status'] == 'healthy']
    COVID19_data = metadata[metadata['status'] == 'COVID-19']
    Symptomatic_data = metadata[metadata['status'] == 'symptomatic']

    dataset_under20_path = os.path.join(dataset_root_path, 'test-under20')
    dataset_20to40_path = os.path.join(dataset_root_path, 'test-20to40')
    dataset_40to60_path = os.path.join(dataset_root_path, 'test-40to60')
    dataset_over60_path = os.path.join(dataset_root_path, 'test-over60')

    healthy_under20_data = healthy_data[healthy_data['age'] == '<=20']
    healthy_under20_uuids = list(healthy_under20_data['uuid'])
    copy_files(healthy_under20_uuids, os.path.join(dataset_under20_path, '0'), dataset_original_path)

    healthy_20to40_data = healthy_data[healthy_data['age'] == '20-40']
    healthy_20to40_uuids = list(healthy_20to40_data['uuid'])
    copy_files(healthy_20to40_uuids, os.path.join(dataset_20to40_path, '0'), dataset_original_path)

    healthy_40to60_data = healthy_data[healthy_data['age'] == '40-60']
    healthy_40to60_uuids = list(healthy_40to60_data['uuid'])
    copy_files(healthy_40to60_uuids, os.path.join(dataset_40to60_path, '0'), dataset_original_path)

    healthy_over60_data = healthy_data[healthy_data['age'] == '>60']
    healthy_over60_uuids = list(healthy_over60_data['uuid'])
    copy_files(healthy_over60_uuids, os.path.join(dataset_over60_path, '0'), dataset_original_path)
    
    COVID_under20_data = COVID19_data[COVID19_data['age'] == '<=20']
    COVID_under20_uuids = list(COVID_under20_data['uuid'])
    copy_files(COVID_under20_uuids, os.path.join(dataset_under20_path, '1'), dataset_original_path)

    COVID_20to40_data = COVID19_data[COVID19_data['age'] == '20-40']
    COVID_20to40_uuids = list(COVID_20to40_data['uuid'])
    copy_files(COVID_20to40_uuids, os.path.join(dataset_20to40_path, '1'), dataset_original_path)

    COVID_40to60_data = COVID19_data[COVID19_data['age'] == '40-60']
    COVID_40to60_uuids = list(COVID_40to60_data['uuid'])
    copy_files(COVID_40to60_uuids, os.path.join(dataset_40to60_path, '1'), dataset_original_path)

    COVID_over60_data = COVID19_data[COVID19_data['age'] == '>60']
    COVID_over60_uuids = list(COVID_over60_data['uuid'])
    copy_files(COVID_over60_uuids, os.path.join(dataset_over60_path, '1'), dataset_original_path)

    Symptomatic_under20_data = Symptomatic_data[Symptomatic_data['age'] == '<=20']
    Symptomatic_under20_uuids = list(Symptomatic_under20_data['uuid'])
    copy_files(Symptomatic_under20_uuids, os.path.join(dataset_under20_path, '2'), dataset_original_path)

    Symptomatic_20to40_data = Symptomatic_data[Symptomatic_data['age'] == '20-40']
    Symptomatic_20to40_uuids = list(Symptomatic_20to40_data['uuid'])
    copy_files(Symptomatic_20to40_uuids, os.path.join(dataset_20to40_path, '2'), dataset_original_path)

    Symptomatic_40to60_data = Symptomatic_data[Symptomatic_data['age'] == '40-60']
    Symptomatic_40to60_uuids = list(Symptomatic_40to60_data['uuid'])
    copy_files(Symptomatic_40to60_uuids, os.path.join(dataset_40to60_path, '2'), dataset_original_path)

    Symptomatic_over60_data = Symptomatic_data[Symptomatic_data['age'] == '>60']
    Symptomatic_over60_uuids = list(Symptomatic_over60_data['uuid'])
    copy_files(Symptomatic_over60_uuids, os.path.join(dataset_over60_path, '2'), dataset_original_path)
    

if __name__ == '__main__':
    dataset_original_path = ORIGINAL_PATH # e.g. ./coughvid_20211012/spec
    dataset_split_path = SPLIT_PATH # e.g. ./dataset_baseline
    # get the csv files for training, validation and test sets
    metadata_train = pd.read_csv(os.path.join(dataset_split_path, 'train.csv'))
    metadata_val = pd.read_csv(os.path.join(dataset_split_path, 'val.csv'))
    metadata_test = pd.read_csv(os.path.join(dataset_split_path, 'test.csv'))
    # split_tool(metadata_train, os.path.join(dataset_split_path, 'train'), dataset_original_path)
    # split_tool(metadata_val, os.path.join(dataset_split_path, 'val'), dataset_original_path)
    # split_tool(metadata_test, os.path.join(dataset_split_path, 'test'), dataset_original_path)

    split_age_tool(metadata_test, dataset_split_path, dataset_original_path)
    split_gender_tool(metadata_test, dataset_split_path, dataset_original_path)

    