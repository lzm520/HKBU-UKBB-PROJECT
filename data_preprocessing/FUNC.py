import os
import sys

import numpy as np
import pandas as pd
import re
import csv
from hes_diag_filter import HES_diagnosis
from ukbb_field_extract import Field_extract_for_self_report, Field_extraction
from sklearn.impute import SimpleImputer
import statsmodels.api as sm


# 通过指定的icd9，icd10，self-report获取eid
def extract_Eid_According_to_icd9_or_icd10_or_selfReport():
    disease_name = 'Type2-Diabetes'
    global ukb_self_report_cancer, ukb_self_report_non_cancer
    field_20001_path = '/tmp/local/cszmli/data/field_extraction/field_20001.csv'
    field_20002_path = '/tmp/local/cszmli/data/field_extraction/field_20002.csv'
    field_birth_path = '/tmp/local/cszmli/data/field_extraction/fields/field_34.npy'
    field_age_at_recruitment_path = '/tmp/local/cszmli/data/field_extraction/fields/field_21022.npy'
    save_path = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_filter.csv'
    training_eid_path = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_training.csv'
    evaluation_eid_path = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_evaluation.csv'

    # CAD
    # icd10_list = [r'I21.*', r'I22.*', r'I23.*', r'I241', r'I252']
    # icd9_list = [r'410.*', r'4110.*', r'412.*', r'42979']

    # Type2 Diabetes
    icd10_list = [r'E11.*']
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
        if np.isnan(hes_diag_data.loc[i, 'disdate']) or (
                hes_diag_data.loc[i, 'disdate'] - float(birth[hes_diag_data.loc[i, 'ukb_index']])) < float(
                recruitment_age[hes_diag_data.loc[i, 'ukb_index']]):
            training_data[hes_diag_data.loc[i, 'ukb_index']] = hes_diag_data.loc[i, 'eid']
        else:
            evaluation_data[hes_diag_data.loc[i, 'ukb_index']] = hes_diag_data.loc[i, 'eid']

    self_report_cancer_list = []

    # CAD
    # self_report_non_cancer_list = [r'1092']

    # Type2 Diabetes
    self_report_non_cancer_list = [r'1248']

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


# 通过判断p_value<0.05来筛选特征
def features_selection():
    disease_name = 'Type2-Diabetes'
    out_file1 = f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_info.csv'
    out_file2 = f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_data.txt'
    in_file1 = '/tmp/local/cszmli/data/clean_data/cleaned_lifeStyle_and_physical_measures_type.csv'
    in_file2 = '/tmp/local/cszmli/data/clean_data/cleaned_lifeStyle_and_physical_measures_data.txt'
    eid_filter_file = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_filter.csv'

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
    disease_name = 'Type2-Diabetes'
    infile_data = open(f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_data.txt', 'r')
    infile_info = open(f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_info.csv', 'r')

    field_info = []
    reader = csv.reader(infile_info)
    for line in reader:
        field_info.append([line[0], line[1], line[2]])
    infile_info.close()

    outfile_result = open(
        f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_data_dummy_data.txt', 'w')
    outfile_label = open(
        f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_data_dummy_info.csv', 'w',
        newline='')
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


# 获取模型训练数据(dummy)
def access_model_training_data():
    disease_name = 'Type2-Diabetes'
    feature_data_path = f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_data_dummy_data.txt'
    feature_info_path = f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_data_dummy_info.csv'
    field_age_at_recruitment_path = '/tmp/local/cszmli/data/field_extraction/fields/field_21022.npy'
    eids_path = '/tmp/local/cszmli/data/field_extraction/eids.csv'
    training_eids_path = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_training.csv'
    evaluation_eids_path = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_evaluation.csv'
    # mr_data_path = f'/tmp/local/cszmli/data/{disease_name}/MR_analysis/ivw_ebi.csv'
    mr_data_path = f'/tmp/local/cszmli/data/{disease_name}/MR_analysis/ivw_summary_statics.csv'
    prs_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/prs.txt'
    training_data_fam_path = f'/tmp/local/cszmli/{disease_name}-chr-data/testing_data/testing_data.fam'
    training_data_info_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data_info.csv'
    training_data_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data.npy'
    training_eid_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid.npy'
    evaluation_data_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_data.npy'
    training_prs_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_prs.npy'
    evaluation_prs_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_prs.npy'
    evaluation_eid_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid.npy'

    # features data loading
    feature_data = np.genfromtxt(feature_data_path)
    feature_info = pd.read_csv(feature_info_path, header=None, names=['Type', 'id', 'Description']).to_numpy().tolist()
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

    print('Already loaded all data')
    print('Now processing data')

    # training data processing
    # reading training & evaluation eid
    training_eids_df = pd.read_csv(training_eids_path)
    training_eids = training_eids_df['eid'].values
    evaluation_eids_df = pd.read_csv(evaluation_eids_path)
    evaluation_eids = evaluation_eids_df['eid'].values

    fam_train = pd.read_table(training_data_fam_path, header=None)
    all_eids = fam_train[0].to_numpy()
    eids_training_list = []
    eids_evaluation_list = []
    training_eids_ls = []
    evaluation_eids_ls = []
    prs_training_list = []
    prs_evaluation_list = []
    for i, eid in enumerate(all_eids):
        if eid in training_eids:
            eids_training_list.append(training_eids_df[training_eids_df['eid'] == eid]['ukb_index'].values[0])
            training_eids_ls.append(eid)
            prs_training_list.append(prs_data[i])
        elif eid in evaluation_eids:
            eids_evaluation_list.append(evaluation_eids_df[evaluation_eids_df['eid'] == eid]['ukb_index'].values[0])
            evaluation_eids_ls.append(eid)
            prs_evaluation_list.append(prs_data[i])

    training_size = len(eids_training_list) * 2
    evaluation_size = len(eids_evaluation_list) * 2
    for i, eid in enumerate(all_eids):
        if not (eid in training_eids) and not (eid in evaluation_eids) and (eid in eids):
            if prs_data[i][1] == 1.:
                print(eid)
            if len(eids_training_list) < training_size:
                eids_training_list.append(np.where(eids == eid)[0][0])
                training_eids_ls.append(eid)
                prs_training_list.append(prs_data[i])
            elif len(eids_evaluation_list) < evaluation_size:
                eids_evaluation_list.append(np.where(eids == eid)[0][0])
                evaluation_eids_ls.append(eid)
                prs_evaluation_list.append(prs_data[i])
            else:
                break
        elif not (eid in eids):
            print(eid)
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
    np.save(training_eid_save_path, training_eids_ls)
    np.save(evaluation_eid_save_path, evaluation_eids_ls)


# 获取模型训练数据(not dummy)
def access_model_training_data_not_dummy():
    disease_name = 'CAD'
    feature_data_path = f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_data.txt'
    feature_info_path = f'/tmp/local/cszmli/data/{disease_name}/features_selection/features_selection_info.csv'
    field_age_at_recruitment_path = '/tmp/local/cszmli/data/field_extraction/fields/field_21022.npy'
    eids_path = '/tmp/local/cszmli/data/field_extraction/eids.csv'
    training_eids_path = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_training.csv'
    evaluation_eids_path = f'/tmp/local/cszmli/data/{disease_name}/eid_filter/eid_evaluation.csv'
    mr_data_path = f'/tmp/local/cszmli/data/{disease_name}/MR_analysis/ivw_ebi.csv'
    # mr_data_path = f'/tmp/local/cszmli/data/{disease_name}/MR_analysis/ivw_summary_statics.csv'
    prs_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/prs.txt'
    training_data_fam_path = f'/tmp/local/cszmli/{disease_name}-chr-data/testing_data/testing_data.fam'
    training_data_info_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/training_data_info.csv'
    training_data_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/training_data.npy'
    training_eid_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/training_eid.npy'
    evaluation_data_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/evaluation_data.npy'
    training_prs_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/training_prs.npy'
    evaluation_prs_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/evaluation_prs.npy'
    evaluation_eid_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/evaluation_eid.npy'

    if not os.path.isdir(f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/'):
        os.mkdir(f'/tmp/local/cszmli/data/{disease_name}/model_training_data_not_dummy/')

    # features data loading
    feature_data = np.genfromtxt(feature_data_path)
    feature_info = pd.read_csv(feature_info_path, header=None).iloc[:, :3].to_numpy().tolist()
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

    print('Already loaded all data')
    print('Now processing data')

    # training data processing
    # reading training & evaluation eid
    training_eids_df = pd.read_csv(training_eids_path)
    training_eids = training_eids_df['eid'].values
    evaluation_eids_df = pd.read_csv(evaluation_eids_path)
    evaluation_eids = evaluation_eids_df['eid'].values

    fam_train = pd.read_table(training_data_fam_path, header=None)
    all_eids = fam_train[0].to_numpy()
    eids_training_list = []
    eids_evaluation_list = []
    training_eids_ls = []
    evaluation_eids_ls = []
    prs_training_list = []
    prs_evaluation_list = []
    for i, eid in enumerate(all_eids):
        if eid in training_eids:
            eids_training_list.append(training_eids_df[training_eids_df['eid'] == eid]['ukb_index'].values[0])
            training_eids_ls.append(eid)
            prs_training_list.append(prs_data[i])
        elif eid in evaluation_eids:
            eids_evaluation_list.append(evaluation_eids_df[evaluation_eids_df['eid'] == eid]['ukb_index'].values[0])
            evaluation_eids_ls.append(eid)
            prs_evaluation_list.append(prs_data[i])

    training_size = len(eids_training_list) * 2
    evaluation_size = len(eids_evaluation_list) * 2
    for i, eid in enumerate(all_eids):
        if not (eid in training_eids) and not (eid in evaluation_eids) and (eid in eids):
            if prs_data[i][1] == 1.:
                print(eid)
            if len(eids_training_list) < training_size:
                eids_training_list.append(np.where(eids == eid)[0][0])
                training_eids_ls.append(eid)
                prs_training_list.append(prs_data[i])
            elif len(eids_evaluation_list) < evaluation_size:
                eids_evaluation_list.append(np.where(eids == eid)[0][0])
                evaluation_eids_ls.append(eid)
                prs_evaluation_list.append(prs_data[i])
            else:
                break
        elif not (eid in eids):
            print(eid)
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
    np.save(training_eid_save_path, training_eids_ls)
    np.save(evaluation_eid_save_path, evaluation_eids_ls)


# 提取历史疾病
def extract_medical_history():
    disease_name = 'Type2-Diabetes'
    hesin_info_f = '/home/comp/ericluzhang/UKBB/HES/hesin.txt'
    hesin_diag_f = '/home/comp/ericluzhang/UKBB/HES/hesin_diag.txt'
    training_eid_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid.npy'
    evaluation_eid_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid.npy'
    fam_path = f'/tmp/local/cszmli/{disease_name}-chr-data/testing_data/testing_data.fam'
    training_eid_icd9_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid_icd9.txt'
    training_eid_icd10_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid_icd10.txt'
    evaluation_eid_icd9_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid_icd9.txt'
    evaluation_eid_icd10_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid_icd10.txt'
    training_eid_icd9_diag_info_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid_icd9_diag_info.txt'
    training_eid_icd10_diag_info_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid_icd10_diag_info.txt'
    evaluation_eid_icd9_diag_info_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid_icd9_diag_info.txt'
    evaluation_eid_icd10_diag_info_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid_icd10_diag_info.txt'

    # loading hesin_info
    hes_info = pd.read_table(hesin_info_f)

    # load training_eid & evaluation_eid
    training_eids = np.load(training_eid_save_path).astype(np.str_).tolist()
    evaluation_eids = np.load(evaluation_eid_save_path).astype(np.str_).tolist()

    # CAD
    # search_icd10_list = [r'I21.*', r'I22.*', r'I23.*', r'I241', r'I252']
    # search_icd9_list = [r'410.*', r'4110.*', r'412.*', r'42979']

    # Type2 Diabetes
    search_icd10_list = [r'E11.*']
    search_icd9_list = []

    # load eid from fam
    eid_training = pd.read_table(fam_path, header=None, sep='\t')[0]
    eid_training = eid_training.values.astype(np.str_)

    # extract hesin_diag, list total icd9/10 and get the top 3 characteristic
    eid_diag = {}
    for eid in eid_training:
        if not eid_diag.get(eid):
            eid_diag[eid] = {'icd9': [], 'icd10': []}
    icd9_set = set()
    icd10_set = set()
    with open(hesin_diag_f, 'r') as fp:
        for i, line in enumerate(fp):
            A = line.strip('\n').split('\t')
            # print(A)
            if i == 0:
                continue
            eid = A[0]
            ins_index = A[1]
            level = A[3]
            ICD9 = A[4].strip()
            ICD10 = A[6].strip()
            if level == '1':
                if eid in eid_training:
                    if ICD9 != '':
                        disdate = hes_info[
                            (hes_info['eid'] == int(float(eid))) & (hes_info['ins_index'] == int(float(ins_index)))][
                            'disdate'].to_list()[0]
                        eid_diag[eid]['icd9'].append((ICD9, ins_index, disdate))
                    if ICD10 != '':
                        disdate = hes_info[
                            (hes_info['eid'] == int(float(eid))) & (hes_info['ins_index'] == int(float(ins_index)))][
                            'disdate'].to_list()[0]
                        eid_diag[eid]['icd10'].append((ICD10, ins_index, disdate))
                if ICD9 != '':
                    icd9_set.add(ICD9[:3])
                if ICD10 != '':
                    icd10_set.add(ICD10[:3])
            if np.mod(i, 10000) == 0:
                print('Iterated entries hes_diag_filter:', i)
    icd9_list = list(icd9_set)
    icd10_list = list(icd10_set)

    # create one-hot matrix
    icd10_df = pd.DataFrame(columns=['eid'] + icd10_list)
    icd10_df['eid'] = eid_training
    icd10_df = icd10_df.fillna(0)
    icd9_df = pd.DataFrame(columns=['eid'] + icd9_list)
    icd9_df['eid'] = eid_training
    icd9_df = icd9_df.fillna(0)
    # create diag info matrix
    eid_icd10_diag_info_df = pd.DataFrame(columns=['eid', 'icd10', 'disdate'])
    eid_icd9_diag_info_df = pd.DataFrame(columns=['eid', 'icd9', 'disdate'])

    print('Now processing one-hot matrix')
    for eid in eid_training:
        t = 999999999
        eid_icd9 = eid_diag[eid]['icd9']
        eid_icd10 = eid_diag[eid]['icd10']
        for (dis, idx, disdate) in eid_icd9:
            for icd9 in search_icd9_list:
                if re.match(icd9, dis):
                    if not np.isnan(disdate):
                        if disdate < t:
                            t = disdate
        for (dis, idx, disdate) in eid_icd10:
            for icd10 in search_icd10_list:
                if re.match(icd10, dis):
                    if not np.isnan(disdate):
                        if disdate < t:
                            t = disdate
        for (dis, idx, disdate) in eid_icd9:
            if disdate < t:
                icd9_df.loc[icd9_df['eid'] == eid, dis[:3]] = 1
                if dis[:3] in eid_icd9_diag_info_df[eid_icd9_diag_info_df['eid'] == eid]['icd9'].values:
                    if (eid_icd9_diag_info_df.disdate[(eid_icd9_diag_info_df['eid'] == eid) & (eid_icd9_diag_info_df['icd9'] == dis[:3])] > disdate).values[0]:
                        eid_icd9_diag_info_df.disdate[(eid_icd9_diag_info_df['eid'] == eid) & (eid_icd9_diag_info_df['icd9'] == dis[:3])] = disdate
                else:
                    eid_icd9_diag_info_df = eid_icd9_diag_info_df.append({'eid': eid, 'icd9': dis, 'disdate': disdate},
                                                                         ignore_index=True)
        for (dis, idx, disdate) in eid_icd10:
            if disdate < t:
                icd10_df.loc[icd10_df['eid'] == eid, dis[:3]] = 1
                if dis[:3] in eid_icd10_diag_info_df[eid_icd10_diag_info_df['eid'] == eid]['icd10'].values:
                    if (eid_icd10_diag_info_df.disdate[(eid_icd10_diag_info_df['eid'] == eid) & (eid_icd10_diag_info_df['icd10'] == dis[:3])] > disdate).values[0]:
                        eid_icd10_diag_info_df.disdate[(eid_icd10_diag_info_df['eid'] == eid) & (eid_icd10_diag_info_df['icd10'] == dis[:3])] = disdate
                else:
                    eid_icd10_diag_info_df = eid_icd10_diag_info_df.append({'eid': eid, 'icd10': dis, 'disdate': disdate},
                                                                       ignore_index=True)
    training_eid_icd9_df = icd9_df[icd9_df['eid'].isin(training_eids)]
    training_eid_icd9_df['eid'] = training_eid_icd9_df['eid'].astype('category')
    training_eid_icd9_df['eid'].cat.set_categories(training_eids, inplace=True)
    training_eid_icd9_df = training_eid_icd9_df.sort_values('eid', ascending=True)

    evaluation_eid_icd9_df = icd9_df[icd9_df['eid'].isin(evaluation_eids)]
    evaluation_eid_icd9_df['eid'] = evaluation_eid_icd9_df['eid'].astype('category')
    evaluation_eid_icd9_df['eid'].cat.set_categories(evaluation_eids, inplace=True)
    evaluation_eid_icd9_df = evaluation_eid_icd9_df.sort_values('eid', ascending=True)

    training_eid_icd10_df = icd10_df[icd10_df['eid'].isin(training_eids)]
    training_eid_icd10_df['eid'] = training_eid_icd10_df['eid'].astype('category')
    training_eid_icd10_df['eid'].cat.set_categories(training_eids, inplace=True)
    training_eid_icd10_df = training_eid_icd10_df.sort_values('eid', ascending=True)

    evaluation_eid_icd10_df = icd10_df[icd10_df['eid'].isin(evaluation_eids)]
    evaluation_eid_icd10_df['eid'] = evaluation_eid_icd10_df['eid'].astype('category')
    evaluation_eid_icd10_df['eid'].cat.set_categories(evaluation_eids, inplace=True)
    evaluation_eid_icd10_df = evaluation_eid_icd10_df.sort_values('eid', ascending=True)

    training_eid_icd9_diag_info_df = eid_icd9_diag_info_df[eid_icd9_diag_info_df['eid'].isin(training_eids)]
    evaluation_eid_icd9_diag_info_df = eid_icd9_diag_info_df[eid_icd9_diag_info_df['eid'].isin(evaluation_eids)]

    training_eid_icd10_diag_info_df = eid_icd10_diag_info_df[eid_icd10_diag_info_df['eid'].isin(training_eids)]
    evaluation_eid_icd10_diag_info_df = eid_icd10_diag_info_df[eid_icd10_diag_info_df['eid'].isin(evaluation_eids)]

    with open(training_eid_icd9_save_path, 'w') as fp:
        fp.write('\t'.join(icd9_list) + '\n')
        for i in range(training_eid_icd9_df.shape[0]):
            fp.write('\t'.join(training_eid_icd9_df.iloc[i, 1:].values.astype(np.str_)) + '\n')

    with open(evaluation_eid_icd9_save_path, 'w') as fp:
        fp.write('\t'.join(icd9_list) + '\n')
        for i in range(evaluation_eid_icd9_df.shape[0]):
            fp.write('\t'.join(evaluation_eid_icd9_df.iloc[i, 1:].values.astype(np.str_)) + '\n')

    with open(training_eid_icd10_save_path, 'w') as fp:
        fp.write('\t'.join(icd10_list) + '\n')
        for i in range(training_eid_icd10_df.shape[0]):
            fp.write('\t'.join(training_eid_icd10_df.iloc[i, 1:].values.astype(np.str_)) + '\n')

    with open(evaluation_eid_icd10_save_path, 'w') as fp:
        fp.write('\t'.join(icd10_list) + '\n')
        for i in range(evaluation_eid_icd10_df.shape[0]):
            fp.write('\t'.join(evaluation_eid_icd10_df.iloc[i, 1:].values.astype(np.str_)) + '\n')

    with open(training_eid_icd9_diag_info_save_path, 'w') as fp:
        fp.write('\t'.join(training_eid_icd9_diag_info_df.columns.values) + '\n')
        for i in range(training_eid_icd9_diag_info_df.shape[0]):
            fp.write('\t'.join(training_eid_icd9_diag_info_df.iloc[i, :].values.astype(np.str_)) + '\n')

    with open(evaluation_eid_icd9_diag_info_save_path, 'w') as fp:
        fp.write('\t'.join(evaluation_eid_icd9_diag_info_df.columns.values) + '\n')
        for i in range(evaluation_eid_icd9_diag_info_df.shape[0]):
            fp.write('\t'.join(evaluation_eid_icd9_diag_info_df.iloc[i, :].values.astype(np.str_)) + '\n')

    with open(training_eid_icd10_diag_info_save_path, 'w') as fp:
        fp.write('\t'.join(training_eid_icd10_diag_info_df.columns.values) + '\n')
        for i in range(training_eid_icd10_diag_info_df.shape[0]):
            fp.write('\t'.join(training_eid_icd10_diag_info_df.iloc[i, :].values.astype(np.str_)) + '\n')

    with open(evaluation_eid_icd10_diag_info_save_path, 'w') as fp:
        fp.write('\t'.join(evaluation_eid_icd10_diag_info_df.columns.values) + '\n')
        for i in range(evaluation_eid_icd10_diag_info_df.shape[0]):
            fp.write('\t'.join(evaluation_eid_icd10_diag_info_df.iloc[i, :].values.astype(np.str_)) + '\n')


if __name__ == '__main__':
    #    extract_Eid_According_to_icd9_or_icd10_or_selfReport()
    #    features_selection()
    category_features_transform()
#    access_model_training_data()
#    extract_medical_history()
