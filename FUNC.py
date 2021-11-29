""" 此pyhon文件集中了所有的功能 """
import os
import sys

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import csv
from lxml import etree
from LZM.hes_diag_filter import HES_diagnosis
from LZM.ukbb_field_extract import Field_extract_for_self_report, Field_extraction
from sklearn.impute import SimpleImputer
import statsmodels.api as sm


# 通过指定的icd9，icd10，self-report获取eid
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


# 从ukb41910文件中将字段类型为Categorical, Integer, Continuous的字段抽出来
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


# 将所有字段都分别抽出来
def Function_two():
    cols_id = []
    with open('../data/cols_type.txt', 'r') as fp:
        for line in fp:
            if line == '\n':
                continue
            row = line.strip().split('\t')
            cols_id.append(row[1])
    cols_id.remove('20001')
    cols_id.remove('20002')
    A = None
    for _, _, c in os.walk('../data/field_extraction/fields'):
        A = c
    for f in A:
        cols_id.remove(f[6: -4])
        print('remove field:', f[6: -4])
    Field_extraction(cols_id)


# 对各个字段的数据进行清洗
def Clean_field():
    outfile1 = '../data/clean_data/raw_impute_data.txt'
    outfile2 = '../data/clean_data/raw_impute_type.txt'

    fields_id = []
    fields_type = []
    with open('../data/cols_filter.txt', 'r') as fp:
        for line in fp:
            if line == '\n':
                continue
            row = line.strip().split('\t')
            fields_type.append(row[0])
            fields_id.append(row[1])

    # testing
    # fields_id = ['19', '46', '48']
    # fields_type = ['Categorical', 'Integer', 'Continuous']

    outfile1 = open(outfile1, 'w')
    outfile2 = open(outfile2, 'w')
    for i, field_id in enumerate(fields_id):
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
                outfile2.write(fields_type[i] + ' ' + fields_id[i])
                outfile2.write('\n')
        elif 'Categorical' == fields_type[i]:
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
                outfile2.write(fields_type[i] + ' ' + fields_id[i])
                outfile2.write('\n')


# 将Category类型的数据转变成01类型的数据
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


# 通过判断p_value<0.05来筛选特征
def Features_selection():
    out_file1 = open('../data/features_selection/features_selection_info.txt', 'w')
    out_file2 = open('../data/features_selection/features_selection_data.txt', 'w')
    n_participants = 502506
    field_id_ukb_idx = np.genfromtxt('../data/eid_filter/eid_filter.csv', delimiter=',', dtype=np.int32)[1:, 0]
    y = np.zeros(n_participants)
    y[field_id_ukb_idx] = 1
    fields_info = []
    with open('../data/clean_data/raw_impute_type.txt') as fp:
        for line in fp:
            if line == '\n':
                continue
            fields_info.append(line.strip().split())
    x = []
    with open('../data/clean_data/raw_impute_data.txt') as fp:
        for line in fp:
            if line == '\n':
                continue
            d = line.strip().split()
            x.append(d)
    x = np.asarray(x, dtype=np.float64)
    features_selection = []
    for i in range(x.shape[0]):
        train_x = x[i].reshape((-1, 1))
        sm_model = sm.Logit(y, sm.add_constant(train_x)).fit(disp=0)
        p_value = sm_model.pvalues
        print('field_' + fields_info[i][1] + ': ' + str(p_value))
        if p_value[1] < 0.05:
            out_file1.write(' '.join(fields_info[i]) + '\n')
            out_file2.write(' '.join(x[i].tolist()))
            out_file2.write('\n')
    out_file1.close()
    out_file2.close()


class MyModel(keras.Model):
    def __init__(self, fixed_field_index):
        super(MyModel, self).__init__()
        self.fixed_field_index = fixed_field_index
        self.fc1 = keras.layers.Dense(1)
        self.fc2 = keras.layers.Dense(1)
        pass

    def call(self, inputs, training=None):
        if len(self.fixed_field_index) > 0:
            control_input = tf.gather(inputs, self.fixed_field_index, axis=1)
        c = np.zeros(inputs.shape[1])
        c[self.fixed_field_index] = 1
        c = np.argwhere(c == 0).squeeze()
        train_input = tf.gather(inputs, c, axis=1)
        if len(self.fixed_field_index) < 1:
            out = self.fc1(train_input)
        else:
            out1 = self.fc1(train_input)
            out2 = self.fc2(control_input)
            out = out1 + out2
        return out


# 用单层的全连接层训练并将连接层中的weight提取出来进行特征权重的判断
# fixed_field_index: 需要把weight固定住的字段的坐标列表
def Function_three():
    # 此处设置控制变量的坐标号
    fixed_field_index = []
    n_participants = 502506
    epoch = 2
    lr = 0.001
    lamda = tf.constant(0.01)
    batch_sz = 512
    # x
    x = []
    with open('../data/clean_data/raw_impute_data.txt') as fp:
        for line in fp:
            if line == '\n':
                continue
            d = line.strip().split()
            x.append(d)
    x = np.asarray(x, dtype=np.float64).transpose()
    x = tf.constant(x)
    # y
    y = np.zeros(n_participants)
    eid_filter_index = np.genfromtxt('../data/eid_filter/eid_filter.csv', delimiter=',', dtype=np.int32)[1:, 0]
    y[eid_filter_index] = 1
    y = tf.constant(y)
    db_train = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(y.shape[0]).batch(batch_sz)
    # x = tf.random.normal(shape=[200, 10])
    # y = tf.random.normal(shape=[200])

    model = MyModel(fixed_field_index=fixed_field_index)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    for epo in range(epoch):
        for step, (train_x, train_y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                out = model(train_x)
                loss = tf.losses.binary_crossentropy(train_y, tf.squeeze(out), from_logits=True)
                norm = 0
                for a in range(len(model.trainable_variables)):
                    norm += tf.norm(model.trainable_variables[a], ord=1)
                loss = loss + lamda * norm
            grades = tape.gradient(loss, [model.fc1.trainable_variables[0]])
            optimizer.apply_gradients(zip(grades, [model.fc1.trainable_variables[0]]))
            if np.mod(step + 1, 10) == 0:
                print("epoch:", epo, "step:", step)
        model.save_weights('../data/model_wights/MyModel_' + str(epo) + '.ckpt')


if __name__ == '__main__':
    # Function_one()
    # Function_two()
    # Cols_filter_type()
    # Clean_field()
    # Category_features_transform()
    # Features_selection()
    Function_three()
