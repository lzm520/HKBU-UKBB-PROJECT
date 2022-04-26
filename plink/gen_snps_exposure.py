import csv
import re
import numpy as np

snps_dict = {}
snps_allele_file = f'/tmp/local/cszmli/plink/snps_allele.csv'
with open(snps_allele_file, 'r') as fp:
    reader = csv.reader(fp)
    for line in reader:
        if not snps_dict.get(line[1]):
            snps_dict[line[1]] = dict()
        snps_dict[line[1]][line[4]] = line[5]
        snps_dict[line[1]][line[5]] = line[4]
print(snps_allele_file + ' 读取完成')

features_info_infile = '/tmp/local/cszmli/data/features_selection/features_selection_data_dummy_info.csv'
features_info = []
with open(features_info_infile, 'r') as fp:
    reader = csv.reader(fp)
    for line in reader:
        features_info.append(line)

for line in features_info:
    cat = line[0]
    exp = line[1]
    folder = f'/tmp/local/cszmli/plink/UKBB_col{exp.split("_")[0]}_assoc/'
    outfile = f'/tmp/local/cszmli/plink/snps_exps/UKBB_col{exp}_assoc.csv'
    with open(outfile, 'w', newline='') as writefile:
        flag = True
        writer = csv.writer(writefile)
        for chr in range(1, 23):
            if cat == 'Categorical':
                file = f'UKBB_chr{chr}_col{exp}.assoc.logistic'
            else:
                file = f'UKBB_chr{chr}_col{exp}.assoc.linear'
            with open(folder + file, 'r') as readfile:
                headers = next(readfile)
                headers = re.split(r'\s+', headers.strip())
                if flag:
                    headers.append('A2')
                    writer.writerow(headers)
                    flag = False
                headers = np.array(headers)
                SNP_pos = np.argwhere(headers == 'SNP')[0][0]
                A1_pos = np.argwhere(headers == 'A1')[0][0]
                for A in readfile:
                    if A == '\n':
                        continue
                    A = re.split(r'\s+', A.strip())
                    if snps_dict.get(A[SNP_pos]):
                        A.append(snps_dict[A[SNP_pos]][A[A1_pos]])
                    writer.writerow(A)