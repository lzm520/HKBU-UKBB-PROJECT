""" 此pyhon文件集中了所有的功能 """
import os
import sys

import numpy as np
import pandas as pd
import re
import csv
from lxml import etree
from LZM.hes_diag_filter import HES_diagnosis
from LZM.ukbb_field_extract import Field_extract_for_self_report, Field_extraction
from sklearn.impute import SimpleImputer

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


""" 将所有字段都分别抽出来 """
def Function_two():
    cols_id = []
    with open('../data/cols_type.txt', 'r') as fp:
        for line in fp:
            row = line.strip().split('\t')
            cols_id.append(row[1])

    for field_id in cols_id:
        if id in ['20001', '20002', 'eid']:
            continue
        Field_extraction(field_id)
        break


""" 从ukb4190.html文件中将各个字段的字段类型抽取出来 """
def Cols_type_extraction():
    fp = open('../ukb41910.html', 'r')
    outfile = open('../data/cols_type.txt', 'w')
    f = fp.read()
    fp.close()

    html = etree.HTML(f)
    contents_rows = html.xpath('/html/body/table[2]/tr')[2:]
    uids = []
    for i, row in enumerate(contents_rows):
        if np.mod(i, 5000) == 0:
            print('iterated entries:', i)

        row_content = etree.tostring(row, encoding='utf-8').decode('utf-8')
        uid = re.search(r'<a.*?>(.*)</a>', row_content).group(1).split('-')[0]
        if uid in uids:
            continue
        uids.append(uid)
        type = re.search(r'<span.*?>(\w*).*?</span>', row_content).group(1)
        type.strip()
        uid.strip()
        outfile.write(type + '\t' + uid)
        outfile.write('\n')
    outfile.close()


""" 从ukb41910文件中将字段类型为Categorical, Integer, Continuous的字段抽出来 """
def Cols_filter_type():
    outfp = open('../data/cols_filter.txt', 'w')
    with open('../data/cols_type.txt', 'r') as fp:
        for line in fp:
            if line == '\n':
                continue
            row = line.strip().split('\t')
            if row[0] in ['Categorical', 'Integer', 'Continuous']:
                outfp.write(line)
    outfp.close()
    pass


""" 获取过滤出来的字段内容并从中提取出过滤出来的eid的数据，并对数据进行清洗 """
def Clean_field():
    outfile1 = '../data/clean_data/raw_impute_data.txt'
    outfile2 = '../data/clean_data/raw_impute_type.txt'

    ukb_idx = []
    with open('../data/eid_filter/eid_filter.csv', 'r') as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            ukb_idx.append(row[0])
    ukb_idx = np.asarray(ukb_idx, dtype=np.int64)

    # fields_id = []
    # fields_type = []
    # with open('../data/cols_filter.txt', 'r') as fp:
    #     for line in fp:
    #         row = line.strip().split('\t')
    #         fields_type.append(row[0])
    #         fields_id.append(row[1])

    # testing
    fields_id = ['19', '46', '48']
    fields_type = ['Categorical', 'Integer', 'Continuous']

    outfile1 = open(outfile1, 'w')
    outfile2 = open(outfile2, 'w')
    for i, field_id in enumerate(fields_id):
        data = []
        filed_path = '../data/field_extraction/fields/field_' + field_id + '.txt'
        with open(filed_path, 'r') as fp:
            for line in fp:
                if line == '\n':
                    continue
                data.append(line.strip())
        data = np.asarray(data)[ukb_idx]

        if fields_type[i] == 'Integer' or fields_type[i] == 'Continuous':
            missing = 0
            newdata = []
            for d in data:
                if d == 'NAN':
                    missing += 1
                    newdata.append(111111)
                elif float(d) < 0:
                    missing += 1
                    newdata.append(111111)
                else:
                    newdata.append(float(d))
            if missing > 50000:
                continue
            else:
                imputer = SimpleImputer(missing_values=111111, strategy='median', verbose=0, copy=True)
                X = imputer.fit_transform(np.array(newdata).reshape(-1, 1))
                for x in X:
                    outfile1.write(str(x[0]) + ' ')
                outfile1.write('\n')
                outfile2.write(fields_type[i] + ' ' + field_id[i])
                outfile2.write('\n')
        elif 'Categorical' in fields_type[i]:
            missing = 0
            newdata = []
            for d in data:
                if d == 'NAN':
                    missing += 1
                    newdata.append(111111)
                elif float(d) < 0:
                    missing += 1
                    newdata.append(111111)
                else:
                    newdata.append(float(d))
            if missing > 50000:
                continue
            else:
                imputer = SimpleImputer(missing_values=111111, strategy='most_frequent', verbose=0, copy=True)
                X = imputer.fit_transform(np.array(newdata).reshape(-1, 1))
                for x in X:
                    outfile1.write(str(x[0]) + ' ')
                outfile1.write('\n')
                outfile2.write(fields_type[i] + ' ' + field_id[i])
                outfile2.write('\n')


""" 将Category类型的数据转变成01类型的数据 """
def Category_features_transform():
    infile_data = open('../data/clean_data/raw_impute_data.txt', 'r')
    infile_info = open('../data/clean_data/raw_impute_type.txt', 'r')
    fields_id = []
    fields_type = []
    for line in infile_info:
        if line == '\n':
            continue
        A = line.strip().split(' ')
        fields_id.append(A[1])
        fields_type.append(A[0])
    infile_info.close()
    outfile_result = open('../data/clean_data/raw_impute__dummy_data.txt', 'w')
    outfile_label = open('../data/clean_data/raw_impute__dummy_type.txt', 'w')

    index = 0
    for line in infile_data:
        if np.mod(index, 2000) == 0:
            print('Iterated transform:', index)
        if line == '\n':
            continue
        A = line.strip().split(' ')
        mylist = list(set(A))
        if len(mylist) == 1:
            index += 1
            continue
        if 'Categorical' in fields_type[index]:
            if len(mylist) == 2:
                outfile_label.write(fields_id[index] + '\n')
                outfile_result.write(line)
            else:
                df = pd.DataFrame(data=np.asarray(A), columns=['data'])
                just_dummies = pd.get_dummies(df['data'])
                step_1 = pd.concat([df, just_dummies], axis=1)
                step_1.drop(['data'], inplace=True, axis=1)
                step_1 = step_1.applymap(np.int32)
                Y = step_1.columns
                X = np.asarray(step_1).transpose()
                sizelabel = X.shape[0]
                for i in range(sizelabel):
                    outfile_label.write(fields_id[index] + '_' + str(Y[i]) + '\n')
                    outfile_result.write(" ".join(map(str, X[i])))
                    outfile_result.write('\n')
        else:
            outfile_label.write(fields_id[index] + '\n')
            outfile_result.write(line)
        index += 1
    outfile_result.close()
    infile_data.close()
    outfile_label.close()
    pass


if __name__ == '__main__':
    # Function_one()
    # Function_two()
    # Cols_type_extraction()
    # Cols_filter_type()
    # Clean_field()
    Category_features_transform()