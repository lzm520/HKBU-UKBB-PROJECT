'''
功能一：从ukb41910.csv文件中提取self-report字段
功能二：合并字段形成一个新的数据集
'''
import os
import sys

import numpy as np
import pandas as pd
import json


def Field_extract_for_self_report(field_id):
    n_participants = 502506
    source_file = '../ukb41910.csv'
    df = pd.read_csv(source_file, encoding='cp936', nrows=1)
    columns = df.columns
    cols = ['eid']
    for col in columns:
        if str.startswith(col, field_id):
            cols.append(col)
    if len(cols) == 1:
        return

    n = 4000
    skip = 0
    ls = []
    while skip < n_participants:
        df = pd.read_csv(source_file, encoding='cp936', nrows=n, skiprows=range(1, skip), usecols=cols)
        print('Iterated entry self report:', skip)
        skip += n

        eids = df['eid']
        data = df.loc[:, cols[1:]]

        for i in range(data.shape[0]):
            locs = ~data.loc[i, cols[1:]].isna()
            field = '&'.join(set(df.loc[i, cols[1:]][locs].to_numpy().astype(np.str)))
            ls.append({'eid': str(eids[i]), field_id: field})
    return ls


def Field_conbine(path):
    x = os.walk(path)
    for root, dirs, files in x:
        root = root  # 当前目录路径
        files = files  # 当前路径下所有非目录子文件
        break
    df = None
    for i, file_name in enumerate(files):
        if not str.endswith(file_name, 'json'):
            continue
        if i == 1:
            df = pd.read_json(root + '/' + file_name)
        else:
            d = pd.read_json(root + '/' + file_name).iloc[:, 1:]
            df = pd.concat([df, d], axis=1)
    return root, df


if __name__ == '__main__':
    Field_extract_for_self_report('20001')


    # ''' 将不同的字段合并，首先先将想要合并的json文件放在同一个文件夹中 '''
    # folder_path = input('please input the absolute path of the folder:')
    # save_file_name = input('please input the saved file name:')
    # root, df = Field_conbine(folder_path)
    #
    # if not isinstance(df, pd.DataFrame):
    #     sys.exit(0)
    # data = json.loads(df.to_json(orient='records'))
    # with open(root + '/' + save_file_name, 'w', encoding='utf-8') as fp:
    #     json.dump(data, fp)
    # print('########## ' + save_file_name + ' is saved ##########')
