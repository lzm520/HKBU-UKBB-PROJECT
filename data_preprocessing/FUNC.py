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
    save_path = '../../data/eid_filter/eid_filter.csv'

    # CAD
    # icd10_list = [r'I20.*', r'I21.*', r'I22.*', r'I23.*', r'I241', r'I252']
    # icd9_list = [r'410.*', r'4110.*', r'412.*', r'42979']

    # Type2 Diabetes
    icd10_list = [r'E11*']
    icd9_list = []
    # Extract patients from clinical records
    hes_diag_data = HES_diagnosis(icd9_list, icd10_list)
    data = {x: y for x, y in zip(hes_diag_data['ukb_index'], hes_diag_data['eid'])}

    self_report_cancer_list = []
    self_report_non_cancer_list = [r'1248']

    # Extract patients from self-report records (Field 20001)
    if len(self_report_cancer_list) > 0 and self_report_cancer_list[0] != '':
        if os.path.exists(field_20001_path):
            dat = []
            with open(field_20001_path, 'r') as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    if np.mod(i, 50000) == 0:
                        print('Iterated entries 20001:', i)
                    dat.append(row)
            ukb_self_report_cancer = pd.DataFrame(dat, columns=['eid', '20001'])
        else:
            ukb_self_report_cancer = Field_extract_for_self_report('20001')
            ukb_self_report_cancer = pd.DataFrame(ukb_self_report_cancer)

        for idx in range(ukb_self_report_cancer.shape[0]):
            cancer = ukb_self_report_cancer.loc[idx, '20001']
            for cancer_id in self_report_cancer_list:
                if re.search(cancer_id, cancer):
                    data[idx] = ukb_self_report_cancer.loc[idx, 'eid']
                    break

    # Extract patients from self-report records (Field 20002)
    if len(self_report_non_cancer_list) > 0 and self_report_non_cancer_list[0] != '':
        if os.path.exists(field_20002_path):
            dat = []
            with open(field_20002_path, 'r') as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    if np.mod(i, 50000) == 0:
                        print('Iterated entries 20002:', i)
                    dat.append(row)
            ukb_self_report_non_cancer = pd.DataFrame(dat, columns=['eid', '20002'])
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
def linearRegression_training():
    # 此处设置控制变量的坐标号
    fixed_field_id = []
    fixed_field_index = []
    with open('../data/features_selection/features_selection_info.txt') as fp:
        for index, line in enumerate(fp):
            if line == '\n':
                continue
            d = line.strip().split()
            if d[1] in fixed_field_id:
                fixed_field_index.append(index)

    n_participants = 502505
    epoch = 2
    lr = 0.001
    lamda = tf.constant(0.01)
    batch_sz = 512

    # x
    x = []
    with open('../../data/features_selection/features_selection_data.txt') as fp:
        for line in fp:
            if line == '\n':
                continue
            d = line.strip().split()
            x.append(d)
    x = np.asarray(x, dtype=np.float64).transpose()
    x = tf.constant(x)
    # y
    y = np.zeros(n_participants)
    eid_filter_index = np.genfromtxt('../../data/eid_filter/eid_filter.csv', delimiter=',', dtype=np.int32)[1:, 0]
    y[eid_filter_index] = 1
    y = tf.constant(y)

    # x = tf.random.normal(shape=[200, 10])
    # y = tf.random.normal(shape=[200])
    db_train = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(y.shape[0]).batch(batch_sz)

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


# 从回归模型参数中提取出不为0的参数并进行排序
def sub_Function_three(fixed_field_index):
    out_file = '../data/features_selection/features_filter_info.npy'
    model_path = ''
    model = MyModel()

    x = []
    with open('../data/features_selection/features_selection_data.txt.txt') as fp:
        for line in fp:
            if line == '\n':
                continue
            d = line.strip().split()
            x.append(d)
    x = np.asarray(x, dtype=np.float64).transpose()

    # 模型加载参数
    model.build(input_shape = [None, x.shape[1]])
    model.load_weights(model_path)
    model_weights = np.array(model.fc1.trainable_variables[0])

    features_id = []
    with open('../data/features_selection/features_selection_info.txt') as fp:
        for line in fp:
            if line == '\n':
                continue
            d = line.split()
            features_id.append(d[1])
    features_id = np.array(features_id)

    C = np.zeros(x.shape[1])
    C[fixed_field_index] = 1
    C = np.argwhere(C == 0).squeeze()
    features_id = features_id[C]

    features_id = features_id[~(model_weights == 0)]
    model_weights = model_weights[~(model_weights == 0)]

    features_id = features_id[np.argsort(model_weights)[-1::-1]]
    features_id = features_id.reshape((-1, 1))
    np.save(out_file, features_id)


def pca(train_x, pca_hidden_layers):
    epoch = 5
    lr = 0.01
    batchsz = 512

    encoder_hidden = []
    for i, neuro_num in enumerate(pca_hidden_layers):
        if i == len(pca_hidden_layers) - 1:
            encoder_hidden.append(tf.keras.layers.Dense(neuro_num))
        else:
            encoder_hidden.append(tf.keras.layers.Dense(neuro_num, activation=tf.nn.sigmoid))
    encoder = keras.Sequential(encoder_hidden)

    decoder_hidden = []
    if len(pca_hidden_layers) > 1:
        for neuro_num in pca_hidden_layers[-2::-1]:
            decoder_hidden.append(tf.keras.layers.Dense(neuro_num, activation=tf.nn.sigmoid))
    decoder_hidden.append(tf.keras.layers.Dense(train_x.shape[1]))
    decoder = keras.Sequential(decoder_hidden)

    train_db = tf.data.Dataset.from_tensor_slices(train_x).shuffle(train_x.shape[0]).batch(batchsz)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    for epo in range(epoch):
        for step, x in enumerate(train_db):
            with tf.GradientTape() as tape:
                output = encoder(x)
                output = decoder(output)
                loss = tf.keras.losses.mean_squared_error(x, output)
            grades = tape.gradient(loss, [encoder.trainable_variables, decoder.trainable_variables])
            optimizer.apply_gradients(zip(grades[0], encoder.trainable_variables))
            optimizer.apply_gradients(zip(grades[1], decoder.trainable_variables))
    return encoder


# 1：首先将想要参考的特征拿出来
# 2：控制参数进行PCA处理
# 3：然后用降为后的数据进行聚类分成不同的人群，并训练一个线性回归模型
# 4：最后修改参考的特征的值，观察它的增幅对每一类人群的影响
def pca_and_Linear():
    pass


if __name__ == '__main__':
    category_features_transform()
    pass
