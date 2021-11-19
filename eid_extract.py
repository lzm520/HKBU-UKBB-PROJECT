import pandas as pd
import numpy as np
import csv


def Eid_extract():
    outfile = open('../data/field_extraction/eids.csv', 'w+')
    with open('../ukb41910.csv', encoding='cp936') as fp:
        render = csv.reader(fp, delimiter=',')
        for i, row in enumerate(render):
            outfile.write(row[0])
            outfile.write('\n')
            if np.mod(i, 1000) == 0:
                print(i)
    outfile.close()


if __name__ == '__main__':
    Eid_extract()
