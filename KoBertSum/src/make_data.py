import os
import sys
import re
#import MeCab

import json
import numpy as np 
import pandas as pd
from tqdm import tqdm
import argparse
import pickle

PROBLEM = 'ext'

## 사용할 path 정의
PROJECT_DIR = os.getcwd()

DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 

def preprocessing(text, tokenizer=None):
    if tokenizer is not None:
        text = tokenizer(text)
        text = ' '.join(text)

    return text

def create_json_files(df, data_type='train', target_summary_sent=None, path=''):
    NUM_DOCS_IN_ONE_FILE = 1000
    start_idx_list = list(range(0, len(df), NUM_DOCS_IN_ONE_FILE))

    for start_idx in tqdm(start_idx_list):
        end_idx = start_idx + NUM_DOCS_IN_ONE_FILE
        if end_idx > len(df):
            end_idx = len(df)  # -1로 하니 안됨...

        #정렬을 위해 앞에 0 채워주기
        length = len(str(len(df)))
        start_idx_str = (length - len(str(start_idx)))*'0' + str(start_idx)
        end_idx_str = (length - len(str(end_idx-1)))*'0' + str(end_idx-1)

        file_name = os.path.join(f'{path}/{data_type}_{target_summary_sent}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json') if target_summary_sent is not None \
                    else os.path.join(f'{path}/{data_type}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json')
        
        json_list = []
        for i, row in df.iloc[start_idx:end_idx].iterrows():
            original_sents_list = [preprocessing(original_sent).split()  # , korean_tokenizer
                                    for original_sent in row['article_original']]

            summary_sents_list = []
            if target_summary_sent is not None:
                if target_summary_sent == 'ext':
                    summary_sents = row['extractive_sents']
                summary_sents_list = [preprocessing(original_sent).split() # , korean_tokenizer
                                        for original_sent in summary_sents]

            json_list.append({'src': original_sents_list,
                              'tgt': summary_sents_list
            })

        json_string = json.dumps(json_list, indent=4, ensure_ascii=False)
        with open(file_name, 'w') as json_file:
            json_file.write(json_string)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default=None, type=str, choices=['df', 'train_bert', 'test_bert'])
    parser.add_argument("-target_summary_sent", default='abs', type=str)
    parser.add_argument("-n_cpus", default='2', type=str)

    args = parser.parse_args()

    # python make_data.py -make df
    # Convert raw data to df
    if args.task == 'df': # and valid_df
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

        # import data
        with open(f'{RAW_DATA_DIR}/kobertsum_train_0615.jsonl', 'r') as json_file:
            train_json_list = list(json_file)
            
        # import data
        with open(f'{RAW_DATA_DIR}/kobertsum_valid_0615.jsonl', 'r') as json_file:
            valid_json_list = list(json_file)
        
        with open(f'{RAW_DATA_DIR}/kobertsum_test_0615.jsonl', 'r') as json_file:
            test_json_list = list(json_file)

        trains = []
        for json_str in train_json_list:
            line = json.loads(json_str)
            trains.append(line)
        
        valids = []
        for json_str in valid_json_list:
            line = json.loads(json_str)
            valids.append(line)
            
        tests = []
        for json_str in test_json_list:
            line = json.loads(json_str)
            tests.append(line)

        # Convert raw data to df
        train_df = pd.DataFrame(trains)
        train_df['extractive_sents'] = train_df.apply(lambda row: list(np.array(row['article_original'])[row['extractive']]) , axis=1)
        
        valid_df = pd.DataFrame(valids)
        valid_df['extractive_sents'] = valid_df.apply(lambda row: list(np.array(row['article_original'])[row['extractive']]) , axis=1)
        
        test_df = pd.DataFrame(tests)

        # save df
        train_df.to_pickle(f"{RAW_DATA_DIR}/train_df.pickle")
        valid_df.to_pickle(f"{RAW_DATA_DIR}/valid_df.pickle")
        test_df.to_pickle(f"{RAW_DATA_DIR}/test_df.pickle")
        print(f'train_df({len(train_df)}) is exported')
        print(f'valid_df({len(valid_df)}) is exported')
        print(f'test_df({len(test_df)}) is exported')
        
    # python make_data.py -make bert -by abs
    # Make bert input file for train and valid from df file
    elif args.task  == 'train_bert':
        os.makedirs(JSON_DATA_DIR, exist_ok=True)
        os.makedirs(BERT_DATA_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        for data_type in ['train', 'valid']:
            df = pd.read_pickle(f"{RAW_DATA_DIR}/{data_type}_df.pickle")

            ## make json file
            # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
            json_data_dir = f"{JSON_DATA_DIR}/{data_type}_{args.target_summary_sent}"
            if os.path.exists(json_data_dir):
                os.system(f"rm {json_data_dir}/*")
            else:
                os.mkdir(json_data_dir)

            create_json_files(df, data_type=data_type, target_summary_sent=args.target_summary_sent, path=JSON_DATA_DIR)
           
            ## Convert json to bert.pt files
            bert_data_dir = f"{BERT_DATA_DIR}/{data_type}_{args.target_summary_sent}"
            if os.path.exists(bert_data_dir):
                os.system(f"rm {bert_data_dir}/*")
            else:
                os.mkdir(bert_data_dir)
            
            os.system(f"python preprocess.py"
                + f" -mode format_to_bert -dataset {data_type}"
                + f" -raw_path {json_data_dir}"
                + f" -save_path {bert_data_dir}"
                + f" -log_file {LOG_PREPO_FILE}"
                + f" -lower -n_cpus {args.n_cpus}")


    # python make_data.py -task test_bert
    # Make bert input file for test from df file
    elif args.task  == 'test_bert':
        os.makedirs(JSON_DATA_DIR, exist_ok=True)
        os.makedirs(BERT_DATA_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        test_df = pd.read_pickle(f"{RAW_DATA_DIR}/test_df.pickle")

        ## make json file
        # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
        json_data_dir = f"{JSON_DATA_DIR}/test"
        if os.path.exists(json_data_dir):
            os.system(f"rm {json_data_dir}/*")
        else:
            os.mkdir(json_data_dir)

        create_json_files(test_df, data_type='test', path=JSON_DATA_DIR)
        
        ## Convert json to bert.pt files
        bert_data_dir = f"{BERT_DATA_DIR}/test"
        if os.path.exists(bert_data_dir):
            os.system(f"rm {bert_data_dir}/*")
        else:
            os.mkdir(bert_data_dir)
        
        os.system(f"python preprocess.py"
            + f" -mode format_to_bert -dataset test"
            + f" -raw_path {json_data_dir}"
            + f" -save_path {bert_data_dir}"
            + f" -log_file {LOG_PREPO_FILE}"
            + f" -lower -n_cpus {args.n_cpus}")