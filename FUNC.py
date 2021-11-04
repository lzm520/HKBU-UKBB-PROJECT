""" 此pyhon文件集中了所有的功能 """
import os
import sys

import numpy as np
import pandas as pd
import re
from LZM.hes_diag_filter import HES_diagnosis
from LZM.ukbb_field_extract import Field_extract_for_self_report

""" 通过指定的icd9，icd10，self-report获取eid """
def Function_one():
    data = {}
    hes_diag_data = HES_diagnosis()
    data = {x: y for x, y in zip(hes_diag_data['ukb_index'], hes_diag_data['eid'])}

    self_report_cancer_list = input('请输入想要查询的自报告的癌症疾病id号（逗号,隔开）:')
    self_report_non_cancer_list = input('请输入想要查询的自报告的非癌症疾病id号（逗号,隔开）:')
    self_report_cancer_list = self_report_cancer_list.split(',')
    self_report_non_cancer_list = self_report_non_cancer_list.split(',')

    ukb_self_report_cancer = Field_extract_for_self_report('20001')
    ukb_self_report_non_cancer = Field_extract_for_self_report('20002')

    ukb_self_report_cancer = pd.DataFrame(ukb_self_report_cancer)
    ukb_self_report_non_cancer = pd.DataFrame(ukb_self_report_non_cancer)

    if len(self_report_cancer_list) > 1 or self_report_cancer_list[0] != '':
        for idx in range(ukb_self_report_cancer.shape[0]):
            cancer = ukb_self_report_cancer.loc[idx, '20001']
            for cancer_id in self_report_cancer_list:
                if re.search(cancer_id, cancer):
                    data[idx] = ukb_self_report_cancer.loc[idx, 'eid']
                    break
    if len(self_report_non_cancer_list) > 1 or self_report_non_cancer_list[0] != '':
        for idx in range(ukb_self_report_non_cancer.shape[0]):
            non_cancer = ukb_self_report_non_cancer.loc[idx, '20002']
            for cancer_id in self_report_non_cancer_list:
                if re.search(cancer_id, non_cancer):
                    data[idx] = ukb_self_report_non_cancer.loc[idx, 'eid']
                    break
    df = pd.DataFrame({'ukb_index': list(data.keys()), 'eid': list(data.values())}).sort_values('ukb_index')
    df.index = range(0, len(df))

    save_path = '../data/eid_filter/eid_filter.csv'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(path_or_buf=save_path, index=False)

    pass


if __name__ == '__main__':
    Function_one()
