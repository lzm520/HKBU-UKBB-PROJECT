import csv
import numpy as np
import re

snps_dict = {}
snps_allele_file = f'/tmp/local/cszmli/plink/snps_allele.csv'
with open(snps_allele_file, 'r') as fp:
    reader = csv.reader(fp)
    for line in reader:
        snps_dict[line[1]] = dict()
        snps_dict[line[1]][line[4]] = line[5]
        snps_dict[line[1]][line[5]] = line[4]
print(snps_allele_file + ' 读取完成')

outfile = '/tmp/local/cszmli/plink/snps_outcome.csv'
flag = True
with open(outfile, 'w') as outfile:
    writer = csv.writer(outfile)
    for i in range(1, 23):
        infile = f'/tmp/local/cszmli/plink/UKBB_case_assoc/UKBB_chr{i}_case.assoc.logistic'
        with open(infile, 'r') as fp:
            headers = next(fp)
            headers = re.split(r'\s+', headers.strip())
            if flag:
                headers.append('A2')
                writer.writerow(headers)
                flag = False
            headers = np.array(headers)
            SNP_pos = np.argwhere(headers == 'SNP')[0][0]
            A1_pos = np.argwhere(headers == 'A1')[0][0]
            for idx, line in enumerate(fp):
                line = re.split(r'\s+', line.strip())
                if snps_dict.get(line[SNP_pos]):
                    line.append(snps_dict[line[SNP_pos]][line[A1_pos]])
                writer.writerow(line)
        print(infile + ' finish')
        break