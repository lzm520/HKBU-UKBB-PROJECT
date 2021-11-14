'''
功能：从医疗诊断记录中以某一疾病提取出相对应的病人及诊断记录以及该病人在ubk41910.csv文件中的所处位置
'''
import re
import json
import numpy as np
from collections import defaultdict

import pandas as pd


def HES_diagnosis(icd9_list, icd10_list):
    n_participants = 502506

    hes_diag = dict()

    hesdiag_file = open('../HES/hesin_diag.txt', 'r')
    ind_eid = pd.read_csv('../data/field_extraction/test.csv').to_numpy().reshape([-1])

    count = 0
    for line in hesdiag_file:

        count += 1

        if np.mod(count, 5000) == 0:
            print('Iterated entries hes_diag_filter:', count)

        A = line.strip('\n').split('\t')
        eid = A[0]
        ins_index = A[1]
        level = A[3]
        ICD9 = A[4]
        ICD10 = A[6]
        index = None
        if eid != 'eid':
            index = np.where(ind_eid == np.int32(eid))[0]
        else:
            continue

        if len(index) == 0:
            continue
        else:
            index = index[0]

        '''在此位置设置想要抽取的疾病ICD9和ICD10'''
        if level == '1':
            for icd10 in icd10_list:
                if re.match(icd10, ICD10):
                    if not hes_diag.get(index):
                        hes_diag[index] = {'ukb_index': index, 'eid': eid}
                        break
            for icd9 in icd9_list:
                if re.match(icd9, ICD9):
                    if not hes_diag.get(index):
                        hes_diag[index] = {'ukb_index': index, 'eid': eid}
                        break
    hesdiag_file.close()
    data = pd.DataFrame(hes_diag).transpose()
    data.index = range(0, len(data))
    return data


if __name__ == '__main__':
    # filter diagnosis in HES_diag file 
    pass
