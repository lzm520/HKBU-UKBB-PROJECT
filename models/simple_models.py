import os
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle


def svm_training(training_x, training_y):
    svm_model = SVC(C=2, probability=True, random_state=9999)
    svm_model.fit(training_x, training_y)
    return svm_model


def lr_training(training_x, training_y):
    lr_model = LogisticRegression(max_iter=70000, random_state=9999)
    lr_model.fit(training_x, training_y)
    return lr_model


def rf_training(training_x, training_y):
    rf_model = RandomForestClassifier(max_depth=3, random_state=9999)
    rf_model.fit(training_x, training_y)
    return rf_model


def model_evaluation(model, evaluation_x, evaluation_y):
    prob = model.predict_proba(evaluation_x)
    auroc = sklearn.metrics.roc_auc_score(evaluation_y, prob[:, 1])
    return auroc


if __name__ == '__main__':
    disease_name = 'CAD'
    training_info_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data_info.csv'
    training_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data.npy'
    training_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_prs.npy'
    evaluation_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_data.npy'
    evaluation_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_prs.npy'
    training_eid_icd9_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid_icd9.txt'
    training_eid_icd10_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_eid_icd10.txt'
    evaluation_eid_icd9_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid_icd9.txt'
    evaluation_eid_icd10_save_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_eid_icd10.txt'
    save_path = f'./result/simpleModels/{disease_name}/'

    ensemble_all_phenotype_flag = True
    add_age_flag = True
    add_prs_flag = True
    add_medical_history_flag = True

    # creating save path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #  data info loading
    training_info = pd.read_csv(training_info_path, names=['Type', 'id', 'Description'], header=None)
    # training data loading
    training_data = np.load(training_data_path).astype(np.float32)
    # training prs & label loading
    training_prs_label = np.load(training_label_path).astype(np.float32)
    training_prs = np.reshape(training_prs_label[:, 0], newshape=[1, -1])
    training_label = training_prs_label[:, 1]
    # evaluation data loading
    evaluation_data = np.load(evaluation_data_path).astype(np.float32)
    # evaluation prs & label loading
    evaluation_prs_label = np.load(evaluation_label_path).astype(np.float32)
    evaluation_prs = np.reshape(evaluation_prs_label[:, 0], newshape=[1, -1])
    evaluation_label = evaluation_prs_label[:, 1]
    # icd data loading
    training_eid_icd9 = None
    training_eid_icd10 = None
    evaluation_eid_icd9 = None
    evaluation_eid_icd10 = None
    if add_medical_history_flag:
        training_eid_icd9 = np.genfromtxt(training_eid_icd9_save_path)[1:, :].astype(np.float32).T
        training_eid_icd10 = np.genfromtxt(training_eid_icd10_save_path)[1:, :].astype(np.float32).T
        evaluation_eid_icd9 = np.genfromtxt(evaluation_eid_icd9_save_path)[1:, :].astype(np.float32).T
        evaluation_eid_icd10 = np.genfromtxt(evaluation_eid_icd10_save_path)[1:, :].astype(np.float32).T

    # processing training X & evaluation X
    training_age_data = training_data[[0]]
    evaluation_age_data = evaluation_data[[0]]
    training_xs = []
    evaluation_xs = []
    features_info = []
    i = 1
    while i < training_info.shape[0]:
        feature_id = training_info.loc[i, 'id'].strip().split('_')[0]
        if ensemble_all_phenotype_flag:
            features_info.append('Ensemble all phenotypes')
            k = training_info.shape[0]
        else:
            features_info.append(training_info.loc[i, 'Description'].strip())
            k = i
        while k < training_info.shape[0]:
            if feature_id != training_info.loc[k, 'id'].strip().split('_')[0]:
                break
            k += 1
        # adding subset training x into the list
        training_x = training_data[i:k]
        if add_prs_flag:
            training_x = np.concatenate((training_prs, training_x), axis=0)
        if add_age_flag:
            training_x = np.concatenate((training_age_data, training_x), axis=0)
        if add_medical_history_flag:
            training_x = np.concatenate((training_x, training_eid_icd9, training_eid_icd10), axis=0)
        training_xs.append(np.transpose(training_x))
        # adding subset evaluation x into the list
        evaluation_x = evaluation_data[i:k]
        if add_prs_flag:
            evaluation_x = np.concatenate((evaluation_prs, evaluation_x), axis=0)
        if add_age_flag:
            evaluation_x = np.concatenate((evaluation_age_data, evaluation_x), axis=0)
        if add_medical_history_flag:
            evaluation_x = np.concatenate((evaluation_x, evaluation_eid_icd9, evaluation_eid_icd10), axis=0)
        evaluation_xs.append(np.transpose(evaluation_x))
        i = k

    svm_file_name = 'svm.txt'
    lr_file_name = 'lr.txt'
    rf_file_name = 'rf.txt'
    if add_age_flag:
        svm_file_name = 'icd_' + svm_file_name
        lr_file_name = 'icd_' + lr_file_name
        rf_file_name = 'icd_' + rf_file_name
    if ensemble_all_phenotype_flag:
        svm_file_name = 'all_phenotype_' + svm_file_name
        lr_file_name = 'all_phenotype_' + lr_file_name
        rf_file_name = 'all_phenotype_' + rf_file_name
    if add_prs_flag:
        svm_file_name = 'prs_' + svm_file_name
        lr_file_name = 'prs_' + lr_file_name
        rf_file_name = 'prs_' + rf_file_name
    if add_age_flag:
        svm_file_name = 'age_' + svm_file_name
        lr_file_name = 'age_' + lr_file_name
        rf_file_name = 'age_' + rf_file_name
    f1 = open(save_path + svm_file_name, 'w')
    f1.write('\t'.join(['AUROC', 'FeatureDescription']))
    f1.write('\n')
    f2 = open(save_path + lr_file_name, 'w')
    f2.write('\t'.join(['AUROC', 'FeatureDescription']))
    f2.write('\n')
    f3 = open(save_path + rf_file_name, 'w')
    f3.write('\t'.join(['AUROC', 'FeatureDescription']))
    f3.write('\n')
    for i in range(len(training_xs)):
        # shuffle training data
        x, y = sklearn.utils.shuffle(training_xs[i], training_label, random_state=9999)
        # SVM model training
        print(f'Now training SVM on {features_info[i]}')
        svm_model = svm_training(x, y)
        f = open(f'{save_path}models/{svm_file_name[:-4]}.pkl', 'wb')
        pickle.dump(svm_model, f)
        f.close()
        # SVM model evaluating
        svm_auroc = model_evaluation(svm_model, evaluation_xs[i], evaluation_label)
        f1.write('\t'.join([str(svm_auroc), features_info[i]]))
        f1.write('\n')

        # LogisticRegression model training
        print(f'Now training LogisticRegression on {features_info[i]}')
        lr_model = lr_training(x, y)
        f = open(f'{save_path}models/{lr_file_name[:-4]}.pkl', 'wb')
        pickle.dump(lr_model, f)
        f.close()
        # LogisticRegression model evaluating
        lr_auroc = model_evaluation(lr_model, evaluation_xs[i], evaluation_label)
        f2.write('\t'.join([str(lr_auroc), features_info[i]]))
        f2.write('\n')

        # RandomForest model training
        print(f'Now training RandomForest on {features_info[i]}')
        rf_model = rf_training(x, y)
        f = open(f'{save_path}models/{rf_file_name[:-4]}.pkl', 'wb')
        pickle.dump(rf_model, f)
        f.close()
        # RandomForest model evaluating
        rf_auroc = model_evaluation(rf_model, evaluation_xs[i], evaluation_label)
        f3.write('\t'.join([str(rf_auroc), features_info[i]]))
        f3.write('\n')
    f1.close()
    f2.close()
    f3.close()
