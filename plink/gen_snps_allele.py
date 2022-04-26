import csv
import re
import numpy as np

snps_file = 'snps.txt'
snps = np.genfromtxt(snps_file, dtype=str)
snps = snps.flatten()

outfile = '/tmp/local/cszmli/plink/snps_allele.csv'
with open(outfile, 'w') as outfp:
    writer = csv.writer(outfp)
    for i in range(1, 23):
        bim_file = f'/tmp/local/cszmli/plink/UKBB_chr{i}.bim'
        with open(bim_file, 'r') as fp:
            for idx, line in enumerate(fp):
                if np.mod(idx + 1, 800000) == 0:
                    print(f'已查看{idx+1}行')
                line = re.split(r'\s+', line.strip())
                if line[1] in snps:
                    writer.writerow([line[1], line[4], line[5]])
        print(bim_file + ' 读取完成')