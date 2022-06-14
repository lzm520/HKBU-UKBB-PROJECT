""" 此pyhon文件集中了所有的功能 """
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import csv
from python_code.data_preprocessing.hes_diag_filter import HES_diagnosis
from python_code.data_preprocessing.ukbb_field_extract import Field_extract_for_self_report, Field_extraction
from sklearn.impute import SimpleImputer
import statsmodels.api as sm


# 通过指定的icd9，icd10，self-report获取eid
def extract_Eid_According_to_icd9_or_icd10_or_selfReport():
    global ukb_self_report_cancer, ukb_self_report_non_cancer
    field_20001_path = '../../data/field_extraction/field_20001.csv'
    field_20002_path = '../../data/field_extraction/field_20002.csv'
    field_birth_path = '../../data/field_extraction/field_34.npy'
    field_age_at_recruitment_path = '../../data/field_extraction/field_21022.npy'
    save_path = '../../data/eid_filter/eid_filter.csv'
    training_eid_path = '../../data/eid_filter/eid_training.csv'
    evaluation_eid_path = '../../data/eid_filter/eid_evaluation.csv'

    time_threshold = 20150101.0
    # CAD
    # icd10_list = [r'I21.*', r'I22.*', r'I23.*', r'I241', r'I252']
    # icd9_list = [r'410.*', r'4110.*', r'412.*', r'42979']

    # Type2 Diabetes
    icd10_list = [r'E11*']
    icd9_list = []

    # Hypertension
    # icd10_list = [r'I10*', r'O13*']
    # icd9_list = []

    # Extract patients from clinical records
    hes_diag_data = HES_diagnosis(icd9_list, icd10_list)
    data = {x: y for x, y in zip(hes_diag_data['ukb_index'], hes_diag_data['eid'])}

    # user birth year loading
    birth = np.load(field_birth_path)

    # user recruitment age loading
    recruitment_age = np.load(field_age_at_recruitment_path)

    training_data = {}
    evaluation_data = {}
    for i in range(hes_diag_data.shape[0]):
        if np.isnan(hes_diag_data.loc[i, 'disdate']) or (hes_diag_data.loc[i, 'disdate'] - float(birth[hes_diag_data.loc[i, 'ukb_index']])) < float(recruitment_age[hes_diag_data.loc[i, 'ukb_index']]):
            training_data[hes_diag_data.loc[i, 'ukb_index']] = hes_diag_data.loc[i, 'eid']
        else:
            evaluation_data[hes_diag_data.loc[i, 'ukb_index']] = hes_diag_data.loc[i, 'eid']
    # training_data = hes_diag_data.loc[(hes_diag_data['disdate'] <= time_threshold) | (hes_diag_data['disdate'] is None)]
    # training_data = {x: y for x, y in zip(training_data['ukb_index'], training_data['eid'])}
    # evaluation_data = hes_diag_data[hes_diag_data['disdate'] > time_threshold]
    # evaluation_data = {x: y for x, y in zip(evaluation_data['ukb_index'], evaluation_data['eid'])}

    self_report_cancer_list = []

    # CAD
    self_report_non_cancer_list = [r'1092']

    # Type2 Diabetes
    # self_report_non_cancer_list = [r'1248']

    # Hypertension
    # self_report_non_cancer_list = [r'1072', r'1073']

    # Extract patients from self-report records (Field 20001)
    if len(self_report_cancer_list) > 0 and self_report_cancer_list[0] != '':
        if os.path.exists(field_20001_path):
            ukb_self_report_cancer = pd.read_csv(field_20001_path)
            ukb_self_report_cancer = ukb_self_report_cancer.fillna('')
        else:
            ukb_self_report_cancer = Field_extract_for_self_report('20001')
            ukb_self_report_cancer = pd.DataFrame(ukb_self_report_cancer)

        for idx in range(ukb_self_report_cancer.shape[0]):
            cancer = ukb_self_report_cancer.loc[idx, '20001']
            for cancer_id in self_report_cancer_list:
                if re.search(cancer_id, cancer):
                    data[idx] = ukb_self_report_cancer.loc[idx, 'eid']
                    training_data[idx] = ukb_self_report_cancer.loc[idx, 'eid']
                    if evaluation_data.get(idx) is not None:
                        evaluation_data.pop(idx)
                    break

    # Extract patients from self-report records (Field 20002)
    if len(self_report_non_cancer_list) > 0 and self_report_non_cancer_list[0] != '':
        if os.path.exists(field_20002_path):
            ukb_self_report_non_cancer = pd.read_csv(field_20002_path)
            ukb_self_report_non_cancer = ukb_self_report_non_cancer.fillna('')
        else:
            ukb_self_report_non_cancer = Field_extract_for_self_report('20002')
            ukb_self_report_non_cancer = pd.DataFrame(ukb_self_report_non_cancer)

        for idx in range(ukb_self_report_non_cancer.shape[0]):
            non_cancer = ukb_self_report_non_cancer.loc[idx, '20002']
            for cancer_id in self_report_non_cancer_list:
                if re.search(cancer_id, non_cancer):
                    data[idx] = ukb_self_report_non_cancer.loc[idx, 'eid']
                    training_data[idx] = ukb_self_report_non_cancer.loc[idx, 'eid']
                    if evaluation_data.get(idx) is not None:
                        evaluation_data.pop(idx)
                    break
    df = pd.DataFrame({'ukb_index': list(data.keys()), 'eid': list(data.values())})
    training_df = pd.DataFrame({'ukb_index': list(training_data.keys()), 'eid': list(training_data.values())})
    evaluation_df = pd.DataFrame({'ukb_index': list(evaluation_data.keys()), 'eid': list(evaluation_data.values())})

    # Save file
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ukb_index', 'eid'])
        for i in range(df.shape[0]):
            if np.mod(i, 500) == 0:
                print('Iterated save patients:', i)
            writer.writerow(df.iloc[i])

    if not os.path.exists(os.path.dirname(training_eid_path)):
        os.makedirs(os.path.dirname(training_eid_path))
    with open(training_eid_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ukb_index', 'eid'])
        for i in range(training_df.shape[0]):
            if np.mod(i, 500) == 0:
                print('Iterated save training patients:', i)
            writer.writerow(training_df.iloc[i])

    if not os.path.exists(os.path.dirname(evaluation_eid_path)):
        os.makedirs(os.path.dirname(evaluation_eid_path))
    with open(evaluation_eid_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ukb_index', 'eid'])
        for i in range(evaluation_df.shape[0]):
            if np.mod(i, 500) == 0:
                print('Iterated save evaluation patients:', i)
            writer.writerow(evaluation_df.iloc[i])

    if not os.path.exists(field_20001_path):
        with open(field_20001_path, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['eid', '20001'])
            for i in range(ukb_self_report_cancer.shape[0]):
                if np.mod(i, 5000) == 0:
                    print('Iterated save 20001:', i)
                writer.writerow(ukb_self_report_cancer.iloc[i])

    if not os.path.exists(field_20002_path):
        with open(field_20002_path, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['eid', '20002'])
            for i in range(ukb_self_report_cancer.shape[0]):
                if np.mod(i, 5000) == 0:
                    print('Iterated save 20002:', i)
                writer.writerow(ukb_self_report_non_cancer.iloc[i])


# 将所有字段都分别抽出来
def extract_all_features():
    cols_id = []
    with open('../../data/cols_type.txt', 'r') as fp:
        for line in fp:
            if line == '\n':
                continue
            row = line.strip().split('\t')
            cols_id.append(row[1])
    cols_id.remove('20001')
    cols_id.remove('20002')
    A = None
    for _, _, c in os.walk('../../data/field_extraction/fields'):
        A = c
    for f in A:
        cols_id.remove(f[6: -4])
        print('remove field:', f[6: -4])
    Field_extraction(cols_id)


# 判断是不是数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


# 清洗特征——填充和删除缺失值大于missingValue的特征
def clean_field():
    infile = '../data/cols_filter/lifeStyle_and_physical_measures_cols_filter.csv'
    outfile1 = '../data/clean_data/cleaned_lifeStyle_and_physical_measures_data.txt'
    outfile2 = '../data/clean_data/cleaned_lifeStyle_and_physical_measures_type.csv'

    fields_id = []
    fields_type = []
    fields_des = []
    with open(infile, 'r') as fp:
        infileReader = csv.reader(fp)
        for line in infileReader:
            if len(line) == 0:
                continue
            fields_type.append(line[0].strip())
            fields_id.append(line[1].strip())
            fields_des.append(line[2].strip())

    outfile1 = open(outfile1, 'w')
    outfile2 = open(outfile2, 'w', newline='')
    outfile2Writer = csv.writer(outfile2)
    for i, field_id in enumerate(fields_id):
        try:
            if np.mod(i, 100) == 0:
                print('Has cleaned field number:', i)
            filed_path = '../data/field_extraction/fields/field_' + field_id + '.npy'
            data = np.load(filed_path)

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
                    outfile2Writer.writerow([fields_type[i], fields_id[i], fields_des[i]])
            elif 'Categorical' == fields_type[i]:
                missing = 0
                newdata = []
                cat_num = {}
                for d in data:
                    if d == 'NAN':
                        missing += 1
                        newdata.append(111111)
                    elif is_number(d) and float(d) < 0:
                        missing += 1
                        newdata.append(111111)
                    else:
                        newdata.append(str(d))
                        if str(d) in cat_num.keys():
                            cat_num[str(d)] += 1
                        else:
                            cat_num[str(d)] = 1
                if missing > 50000:
                    continue
                else:
                    max_num = 0
                    max_num_cat = None
                    for key in cat_num.keys():
                        if cat_num[key] > max_num:
                            max_num_cat = key
                            max_num = cat_num[key]
                    newdata = np.array(newdata)
                    newdata[newdata == '111111'] = max_num_cat

                    X = newdata.reshape((-1, 1))
                    for x in X:
                        outfile1.write(str(x[0]) + ' ')
                    outfile1.write('\n')
                    outfile2Writer.writerow([fields_type[i], fields_id[i], fields_des[i]])
        except Exception as e:
            print(e)
            print('field ', field_id, ' appear error when cleaning!')
    outfile1.close()
    outfile2.close()


# 通过判断p_value<0.05来筛选特征
def features_selection():
    out_file1 = '../data/features_selection/features_selection_info.csv'
    out_file2 = '../data/features_selection/features_selection_data.txt'
    in_file1 = '../data/clean_data/cleaned_lifeStyle_and_physical_measures_type.csv'
    in_file2 = '../data/clean_data/cleaned_lifeStyle_and_physical_measures_data.txt'
    eid_filter_file = '../../data/eid_filter/eid_filter.csv'

    out_file1 = open(out_file1, 'w', newline='')
    out_file2 = open(out_file2, 'w')
    out_file1_writer = csv.writer(out_file1)

    n_participants = 502505
    field_id_ukb_idx = np.genfromtxt(eid_filter_file, delimiter=',', dtype=np.int32)[1:, 0]
    y = np.zeros(n_participants)
    y[field_id_ukb_idx] = 1
    fields_info = []
    with open(in_file1, 'r') as fp:
        reader = csv.reader(fp)
        for line in reader:
            if len(line) == 0:
                continue
            else:
                fields_info.append(line)

    x = []
    with open(in_file2, 'r') as fp:
        for line in fp:
            if line == '\n':
                continue
            d = line.strip().split()
            x.append(d)
    x = np.asarray(x, dtype=np.float64)

    for i in range(x.shape[0]):
        if np.mod(i, 50) == 0:
            print('Has tested feature number:', i)
        train_x = x[i].reshape((-1, 1))
        sm_model = sm.Logit(y, sm.add_constant(train_x)).fit(disp=0)
        p_value = sm_model.pvalues
        params = sm_model.params
        print('field_' + fields_info[i][1] + ' p-value: ' + str(p_value))
        print('field_' + fields_info[i][1] + '  params: ' + str(params))

        if p_value[1] < 0.05:
            fields_info[i].append(p_value[1])
            fields_info[i].append(np.exp(params[1]))
            out_file1_writer.writerow(fields_info[i])
            out_file2.write(' '.join(x[i].astype(str).tolist()))
            out_file2.write('\n')
    out_file1.close()
    out_file2.close()


# 将Category类型的数据转变成01类型的数据
def category_features_transform():
    infile_data = open('../../data/features_selection/features_selection_data.txt', 'r')
    infile_info = open('../../data/features_selection/features_selection_info.csv', 'r')

    field_info = []
    reader = csv.reader(infile_info)
    for line in reader:
        field_info.append([line[0], line[1], line[2]])
    infile_info.close()

    outfile_result = open('../../data/features_selection/features_selection_data_dummy_data.txt', 'w')
    outfile_label = open('../../data/features_selection/features_selection_data_dummy_info.csv', 'w', newline='')
    outfile_label_writer = csv.writer(outfile_label)
    index = 0
    for line in infile_data:
        if np.mod(index, 50) == 0:
            print('Iterated transform:', index)
        if line == '\n':
            continue
        if 'Categorical' in field_info[index][0]:
            A = line.strip().split(' ')
            mylist = list(set(A))
            if len(mylist) == 1 or len(mylist) > 30:
                index += 1
                continue
            # if len(mylist) == 2:
            #     outfile_label.write(fields_id[index] + '\n')
            #     outfile_result.write(line)
            # else:
            df = pd.DataFrame(data=np.asarray(A), columns=['data'])
            just_dummies = pd.get_dummies(df['data'])
            step_1 = pd.concat([df, just_dummies], axis=1)
            step_1.drop(['data'], inplace=True, axis=1)
            step_1 = step_1.applymap(np.int32)
            Y = step_1.columns
            X = np.asarray(step_1).transpose()
            sizelabel = X.shape[0]
            info = [field_info[index][0], field_info[index][1], field_info[index][2]]
            for i in range(sizelabel):
                info[1] = field_info[index][1] + '_' + str(int(float(Y[i])))
                outfile_label_writer.writerow(info)
                outfile_result.write(" ".join(map(str, X[i])))
                outfile_result.write('\n')
        else:
            outfile_label_writer.writerow(field_info[index])
            outfile_result.write(line)
        index += 1
    outfile_result.close()
    infile_data.close()
    outfile_label.close()


# 获取模型训练数据
def access_model_training_data():
    feature_data_path = '../../data/features_selection/features_selection_data_dummy_data.txt'
    feature_info_path = '../../data/features_selection/features_selection_data_dummy_info.csv'
    field_age_at_recruitment_path = '../../data/field_extraction/field_21022.npy'
    eids_path = '../../data/field_extraction/eids.csv'
    training_eids_path = '../../data/eid_filter/eid_training.csv'
    evaluation_eids_path = '../../data/eid_filter/eid_evaluation.csv'
    mr_data_path = '../../data/MR_analysis/ivw_ebi.csv'
    prs_data_path = '../../data/model_training_data/t2d/prs.txt'
    training_data_fam_path = '../../data/testing_data.fam'
    training_data_info_save_path = '../../data/model_training_data/training_data_info.csv'
    training_data_save_path = '../../data/model_training_data/training_data.npy'
    evaluation_data_save_path = '../../data/model_training_data/evaluation_data.npy'
    training_prs_save_path = '../../data/model_training_data/training_prs.npy'
    evaluation_prs_save_path = '../../data/model_training_data/evaluation_prs.npy'

    # features data loading
    feature_data = np.genfromtxt(feature_data_path)
    feature_info = pd.read_csv(feature_info_path, header=None,
                               names=['Type', 'id', 'Description']).tolist()
    list.insert(feature_info, 0, ['Integer', 34, 'Age at recruitment'])
    feature_info = pd.DataFrame(feature_info, columns=['Type', 'id', 'Description'])

    # eids loading
    eids = pd.read_csv(eids_path)['eid'].to_numpy()

    # recruitment age loading
    recruitment_age = np.load(field_age_at_recruitment_path)
    feature_data = feature_data.tolist()
    list.insert(feature_data, 0, recruitment_age)
    feature_data = np.array(feature_data)

    # mr data loading
    ivw_data = pd.read_csv(mr_data_path)
    sig_phenotype_list = ivw_data[ivw_data['qval'] < 0.05].reset_index()['exposure'].to_list()
    sig_phenotype = []
    for pheno in sig_phenotype_list:
        sig_phenotype.append(str.split(pheno, "||")[0].strip())

    # PRS data loading
    prs_data = np.genfromtxt(prs_data_path)

    # Training data processing
    # reading training & evaluation eid
    training_eids_df = pd.read_csv(training_eids_path)
    training_eids = training_eids_df['eid'].values
    evaluation_eids_df = pd.read_csv(evaluation_eids_path)
    evaluation_eids = evaluation_eids_df['eid'].values

    fam_train = pd.read_table(training_data_fam_path, header=None)
    all_eids = fam_train[0].to_numpy()
    eids_training_list = []
    eids_evaluation_list = []
    prs_training_list = []
    prs_evaluation_list = []
    for i, eid in enumerate(all_eids):
        if eid in training_eids:
            eids_training_list.append(training_eids_df[training_eids_df['eid'] == eid]['ukb_index'].values[0])
            prs_training_list.append(prs_data[i])
        elif eid in evaluation_eids:
            eids_evaluation_list.append(evaluation_eids_df[evaluation_eids_df['eid'] == eid]['ukb_index'].values[0])
            prs_evaluation_list.append(prs_data[i])

    training_size = len(eids_training_list) * 2
    evaluation_size = len(eids_evaluation_list) * 2
    for i, eid in enumerate(all_eids):
        if eid not in training_eids and eid not in evaluation_eids and eid in eids:
            if len(eids_training_list) < training_size:
                eids_training_list.append(np.where(eids == eid)[0][0])
                prs_training_list.append(prs_data[i])
            elif len(eids_evaluation_list) < evaluation_size:
                eids_evaluation_list.append(np.where(eids == eid)[0][0])
                prs_evaluation_list.append(prs_data[i])
            else:
                break
    feature_data_training = feature_data[:, eids_training_list]
    feature_data_evaluation = feature_data[:, eids_evaluation_list]

    data_training = []
    data_evaluation = []
    data_info = []
    for i in range(feature_info.shape[0]):
        if i == 0:
            # adding the age infomation into the model_training data
            data_training.append(feature_data_training[i])
            data_evaluation.append(feature_data_evaluation[i])
            data_info.append(feature_info.loc[i].to_numpy())

        description = feature_info.loc[i, 'Description']
        if description in sig_phenotype:
            data_training.append(feature_data_training[i])
            data_evaluation.append(feature_data_evaluation[i])
            data_info.append(feature_info.loc[i].to_numpy())
    # training data saving
    with open(training_data_info_save_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        for info in data_info:
            writer.writerow(info)
    np.save(training_data_save_path, data_training)
    np.save(evaluation_data_save_path, data_evaluation)
    np.save(training_prs_save_path, prs_training_list)
    np.save(evaluation_prs_save_path, prs_evaluation_list)


if __name__ == '__main__':
    category_features_transform()
    pass
