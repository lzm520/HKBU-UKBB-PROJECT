""" 此pyhon文件集中了所有的功能 """
import sys

import numpy as np
import pandas as pd
import re
from LZM.hes_diag_filter import HES_diagnosis
from LZM.hkbb_field_extract import Field_extract

""" 通过指定的icd9，icd10，self-report获取eid """
def Function_one():
    hes_diag_data = HES_diagnosis()

    self_report_cancer_list = input('请输入想要查询的自报告的癌症疾病id号（逗号,隔开）:')
    self_report_non_cancer_list = input('请输入想要查询的自报告的非癌症疾病id号（逗号,隔开）:')
    self_report_cancer_list = self_report_cancer_list.split(',')
    self_report_non_cancer_list = self_report_non_cancer_list.split(',')

    hkb_self_report_cancer = Field_extract('20001')
    hkb_self_report_non_cancer = Field_extract('20002')

    hkb_self_report_cancer = pd.DataFrame(hkb_self_report_cancer)
    hkb_self_report_non_cancer = pd.DataFrame(hkb_self_report_non_cancer)

    data = {x: y for x, y in zip(hes_diag_data['ukb_index'], hes_diag_data['eid'])}

    if len(self_report_cancer_list) > 1 or self_report_cancer_list[0] != '':
        for idx in range(hkb_self_report_cancer.shape[0]):
            cancer = hkb_self_report_cancer.loc[idx, '20001']
            for cancer_id in self_report_cancer_list:
                if re.search(cancer_id, cancer):
                    data[idx] = hkb_self_report_cancer.loc[idx, 'eid']
                    break
    if len(self_report_non_cancer_list) > 1 or self_report_non_cancer_list[0] != '':
        for idx in range(hkb_self_report_non_cancer.shape[0]):
            non_cancer = hkb_self_report_non_cancer.loc[idx, '20002']
            for cancer_id in self_report_non_cancer_list:
                if re.search(cancer_id, non_cancer):
                    data[idx] = hkb_self_report_non_cancer.loc[idx, 'eid']
                    break
    df = pd.DataFrame({'ukb_index': list(data.keys()), 'eid': list(data.values())})
    df.to_csv(path_or_buf='../data/eid_filter/testing.csv', index=False)
    pass


if __name__ == '__main__':
    Function_one()
