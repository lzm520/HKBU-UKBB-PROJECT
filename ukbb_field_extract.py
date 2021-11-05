'''
功能一：从ukb41910.csv文件中提取self-report字段
功能二：合并字段形成一个新的数据集
'''
import os
import re
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


def Field_extraction(field_id):
    n_participants = 502506
    source_file = '../ukb41910.csv'
    out_file = '../data/field_extraction/field_' + field_id + '.txt'
    out_fp = open(out_file, 'w')
    df = pd.read_csv(source_file, encoding='cp936', nrows=1)
    columns = df.columns
    cols = []
    for col in columns:
        if re.search('^' + field_id + '-', col):
            cols.append(col)
    if len(cols) == 0:
        return

    n = 4000
    skip = 0
    while skip < n_participants:
        df = pd.read_csv(source_file, encoding='cp936', nrows=n, skiprows=range(1, skip), usecols=cols)
        print('iterated extraction field_' + field_id + ':', skip)
        skip += n

        data = df.loc[:, cols[:]]

        for i in range(data.shape[0]):
            locs = ~data.loc[i, cols[:]].isna()
            field = df.loc[i, cols[:]][locs]
            if len(field) == 0:
                out_fp.write('')
            else:
                out_fp.write(str(field[-1]))
            # field = '&'.join(df.loc[i, cols[:]][locs].to_numpy().astype(np.str))
            # out_fp.write(field)
            out_fp.write('\n')
    out_fp.close()


if __name__ == '__main__':
    Field_extract_for_self_report('20002')
