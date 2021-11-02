import pandas as pd
import numpy as np


def Eid_extract():
    df = pd.read_csv('../ukb41910.csv', encoding='cp936')
    eids = df['eid'].to_numpy()
    np.save('../data/field_extraction/eids.npy', eids)


if __name__ == '__main__':
    Eid_extract()