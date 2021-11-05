""" 此pyhon文件集中了所有的功能 """
import os
import sys

import numpy as np
import pandas as pd
import re
import csv
from LZM.hes_diag_filter import HES_diagnosis
from LZM.ukbb_field_extract import Field_extract_for_self_report

""" 通过指定的icd9，icd10，self-report获取eid """


def Function_one():
    global ukb_self_report_cancer, ukb_self_report_non_cancer
    field_20001_path = '../data/field_extraction/field_20001.csv'
    field_20002_path = '../data/field_extraction/field_20002.csv'
    save_path = '../data/eid_filter/eid_filter.csv'

    data = {}
    # 默认查询冠心病
    icd10_list = [r'I20.*', r'I21.*', r'I22.*', r'I23.*', r'I241', r'I252']
    icd9_list = [r'410.*', r'4110.*', r'412.*', r'42979']
    hes_diag_data = HES_diagnosis(icd9_list, icd10_list)
    data = {x: y for x, y in zip(hes_diag_data['ukb_index'], hes_diag_data['eid'])}

    self_report_cancer_list = ['1024']
    self_report_non_cancer_list = ['1356']

    if len(self_report_cancer_list) > 1 or self_report_cancer_list[0] != '':
        if os.path.exists(field_20001_path):
            data = []
            with open(field_20001_path, 'r') as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    if np.mod(i, 5000) == 0:
                        print('Iterated entries 20001:', i)
                    data.append(row)
            ukb_self_report_cancer = pd.DataFrame(data, columns=['eid', '20001'])
        else:
            ukb_self_report_cancer = Field_extract_for_self_report('20001')
            ukb_self_report_cancer = pd.DataFrame(ukb_self_report_cancer)

        for idx in range(ukb_self_report_cancer.shape[0]):
            cancer = ukb_self_report_cancer.loc[idx, '20001']
            for cancer_id in self_report_cancer_list:
                if re.search(cancer_id, cancer):
                    data[idx] = ukb_self_report_cancer.loc[idx, 'eid']
                    break

    if len(self_report_non_cancer_list) > 1 or self_report_non_cancer_list[0] != '':
        if os.path.exists(field_20002_path):
            data = []
            with open(field_20001_path, 'r') as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    if np.mod(i, 5000) == 0:
                        print('Iterated entries 20002:', i)
                    data.append(row)
            ukb_self_report_non_cancer = pd.DataFrame(data, columns=['eid', '20002'])
        else:
            ukb_self_report_non_cancer = Field_extract_for_self_report('20002')
            ukb_self_report_non_cancer = pd.DataFrame(ukb_self_report_non_cancer)

        for idx in range(ukb_self_report_non_cancer.shape[0]):
            non_cancer = ukb_self_report_non_cancer.loc[idx, '20002']
            for cancer_id in self_report_non_cancer_list:
                if re.search(cancer_id, non_cancer):
                    data[idx] = ukb_self_report_non_cancer.loc[idx, 'eid']
                    break
    df = pd.DataFrame({'ukb_index': list(data.keys()), 'eid': list(data.values())})

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ukb_index', 'eid'])
        for i in range(df.shape[0]):
            if np.mod(i, 5000):
                print('Iterated save patients:', i)
            writer.writerow(df.iloc[i])

    if not os.path.exists(field_20001_path):
        with open(field_20001_path, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['eid', '20001'])
            for i in range(ukb_self_report_cancer.shape[0]):
                if np.mod(i, 5000):
                    print('Iterated save 20001:', i)
                writer.writerow(ukb_self_report_cancer.iloc[i])

    if not os.path.exists(field_20002_path):
        with open(field_20002_path, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['eid', '20002'])
            for i in range(ukb_self_report_cancer.shape[0]):
                if np.mod(i, 5000):
                    print('Iterated save 20002:', i)
                writer.writerow(ukb_self_report_non_cancer.iloc[i])


if __name__ == '__main__':
    Function_one()
