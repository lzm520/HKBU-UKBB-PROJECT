'''
功能一：从ukb41910.csv文件中提取self-report字段
'''
import os
import re
import sys

import numpy as np
import pandas as pd
import json
from collections import defaultdict
import csv


# 只针对于20001，20002字段的提取
def Field_extract_for_self_report(field_id):
    n_participants = 502505
    source_file = '../../ukb41910.csv'
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


# ukb文件字段提取
def Field_extraction(cols_id):
    source_file = '../../ukb41910.csv'
    df = pd.read_csv(source_file, encoding='cp936', nrows=1)
    columns = df.columns

    fields = []
    all_cols_idx = []
    for field_id in cols_id[:]:
        print('Processing field:', field_id)
        cols_idx = []
        for idx, col in enumerate(columns):
            if re.search('^' + field_id + '-', col):
                cols_idx.append(idx)
        if len(cols_idx) == 0:
            continue
        all_cols_idx.append(cols_idx)
        fields.append(field_id)

    field_content_dict = defaultdict(list)
    with open(source_file, 'r', encoding='cp936') as fp:
        reader = csv.reader(fp)
        for i, cols in enumerate(reader):
            if i == 0:
                continue
            if np.mod(i, 200) == 0:
                print('Has extracted people number:', i)
            cols = np.asarray(cols)
            for k, cols_idx in enumerate(all_cols_idx):
                A = cols[cols_idx]
                field_content_dict[fields[k]].append(A)

    for i, key in enumerate(field_content_dict.keys()):
        if np.mod(i+1, 200) == 0:
            print('Has save field number:', i)
        try:
            np.save('../data/field_extraction/fields/field_' + key + '.npy', field_content_dict[key])
        except Exception as e:
            print(e)


if __name__ == '__main__':
    Field_extract_for_self_report('20002')
