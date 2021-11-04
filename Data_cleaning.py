import re

import numpy as np
import pandas as pd
from lxml import etree


def Cols_type_extraction():
    fp = open('../ukb41910.html', 'r')
    outfile = open('../data/cols_type.txt', 'w')
    f = fp.read()
    fp.close()

    html = etree.HTML(f)
    contents_rows = html.xpath('/html/body/table[2]/tr')[2:]
    uids = []
    for i, row in enumerate(contents_rows):
        if i == 5000:
            break
        if np.mod(i, 5000) == 0:
            print('iterated entries:', i)

        row_content = etree.tostring(row, encoding='utf-8').decode('utf-8')
        uid = re.search(r'<a.*?>(.*)</a>', row_content).group(1).split('-')[0]
        if uid in uids:
            continue
        uids.append(uid)
        type = re.search(r'<span.*?>(\w*).*?</span>', row_content).group(1)
        outfile.write(type + '\t' + uid)
        outfile.write('\n')
    outfile.close()


if __name__ == '__main__':
    Cols_type_extraction()