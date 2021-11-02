'''
功能：从医疗诊断记录中以某一疾病提取出相对应的病人及诊断记录以及该病人在ubk41910.csv文件中的所处位置
'''
import re
import json
import numpy as np
from collections import defaultdict


def HES_diagnosis():
    n_participants = 502506

    disease_name = 'Coronary artery disease'
    hes_diag = dict()

    hesdiag_file = open('../HES/hesin_diag.txt', 'r')
    ind_eid = np.load('../data/field_extraction/eids.npy')

    count = 0
    for line in hesdiag_file:

        count += 1

        if np.mod(count, 50000) == 0:
            print('Iterated entries:', count)

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
            flag = False
            diag = defaultdict()
            diag = {'ukb_index': str(index), 'eid': str(eid), 'icd9': '', 'icd10': ''}

            if re.match(r'I20.*', ICD10) or re.match(r'I21.*', ICD10) or re.match(r'I22.*', ICD10) \
                    or re.match(r'I23.*', ICD10) or re.match(r'I241', ICD10) or re.match(r'I252', ICD10):
                diag['icd10'] = str(ICD10)
                diag['Coronary artery disease'] = True
                flag = True

            if re.match(r'410.*', ICD9) or re.match(r'4110.*', ICD9) or re.match(r'412.*', ICD9) \
                    or re.match(r'42979',  ICD9):
                diag['icd9'] = str(ICD9)
                diag['Coronary artery disease'] = True
                flag = True

            if flag:
                if hes_diag.get(diag['eid']):
                    if diag['icd9']:
                        if not re.search(diag['icd9'], hes_diag[diag['eid']]['icd9']):
                            hes_diag[diag['eid']]['icd9'] = hes_diag[diag['eid']]['icd9'] + '&' + diag['icd9']
                    if diag['icd10']:
                        if not re.search(diag['icd10'], hes_diag[diag['eid']]['icd10']):
                            hes_diag[diag['eid']]['icd10'] = hes_diag[diag['eid']]['icd10'] + '&' + diag['icd10']
                else:
                    hes_diag[diag['eid']] = diag

    hesdiag_file.close()

    with open('../data/diag_selection/' + disease_name + ' diag records.json', 'w+', encoding='utf-8') as fp:
        json.dump(hes_diag, fp)
    print('########## ' + disease_name + ' diag records is saved ##########')
    # np.save('/home/comp/ericluzhang/lzm/data/ind_select/hes', hes_matrix)


if __name__ == '__main__':
    # filter diagnosis in HES_diag file 
    HES_diagnosis()
