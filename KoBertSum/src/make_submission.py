import json
import numpy as np
import pandas as pd
import time
import re
import sys
import os

PROBLEM = 'ext'

## 사용할 path 정의
PROJECT_DIR = os.getcwd()
print(PROJECT_DIR)

DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 

MODEL_DIR = f'{PROJECT_DIR}/{PROBLEM}/models' 
RESULT_DIR = f'{PROJECT_DIR}/{PROBLEM}/results' 

if __name__ == '__main__':
    # test set
    with open(RAW_DATA_DIR + '/kobertsum_test_0615.jsonl', 'r') as json_file:
        json_list = list(json_file)

    tests = []
    for json_str in json_list:
        line = json.loads(json_str)
        tests.append(line)
    test_df = pd.DataFrame(tests)

    # 추론결과
    print(sys.argv[1])
    with open(RESULT_DIR + '/' + sys.argv[1], 'r') as file:
        lines = file.readlines()
    # print(lines)
    test_pred_list = []
    for line in lines:
        sum_sents_text, sum_sents_idxes = line.rsplit(r'[', maxsplit=1)
        sum_sents_text = sum_sents_text.replace('<q>', '\n')
        sum_sents_idx_list = [ int(str.strip(i)) for i in sum_sents_idxes[:-2].split(', ')]
        test_pred_list.append({'sum_sents_tokenized': sum_sents_text, 
                            'sum_sents_idxes': sum_sents_idx_list
                            })

    result_df = pd.merge(test_df, pd.DataFrame(test_pred_list), how="left", left_index=True, right_index=True)
    result_df.to_csv(f'{RESULT_DIR}/kobertsum_{sys.argv[1]}.csv', index=False, encoding="utf-8")