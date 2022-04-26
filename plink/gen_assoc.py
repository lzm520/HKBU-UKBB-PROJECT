import os
import csv

features_selection_info_infile = '/tmp/local/cszmli/data/features_selection/features_selection_data_dummy_info.csv'
features_info = []
with open(features_selection_info_infile, 'r', encoding='utf-8') as fp:
    reader = csv.reader(fp)
    for line in reader:
        features_info.append([line[0], line[1]])
print('特征数据读取完成')

for info in features_info:
    path = '/tmp/local/cszmli/plink/'
    start_name = f'UKBB_col_id{info[1]}.fam'
    print('Now processing file: ', start_name)
    k = start_name
    if not os.path.exists(path + f'UKBB_col{info[1].split("_")[0]}_assoc/'):
        os.mkdir(path + f'UKBB_col{info[1].split("_")[0]}_assoc/')
    creat_cmd = f'cp {path}fam_folder/{start_name} {path + start_name}'
    os.system(creat_cmd)
    for i in range(22, 0, -1):
        fam_name = f'UKBB_chr{i}.fam'
        if info[0] == 'Integer' or info[0] == 'Continuous':
            cmd = f'/home/comp/cszmli/LZM/plink --bfile /tmp/local/cszmli/plink/UKBB_chr{i} --geno 0.1 --hwe 0.000001 --maf ' \
                  '0.01 --mind 0.1 --linear --ci 0.95 --extract snps.txt --out ' \
                  f'/tmp/local/cszmli/plink/UKBB_col{info[1].split("_")[0]}_assoc/UKBB_chr{i}_col{info[1]} --allow-no-covars '
        else:
            cmd = f'/home/comp/cszmli/LZM/plink --bfile /tmp/local/cszmli/plink/UKBB_chr{i} --geno 0.1 --hwe 0.000001 --maf ' \
                  '0.01 --mind 0.1 --logistic --ci 0.95 --extract snps.txt --out ' \
                  f'/tmp/local/cszmli/plink/UKBB_col{info[1].split("_")[0]}_assoc/UKBB_chr{i}_col{info[1]} --allow-no-covars '
        os.system(f'mv {path + k} {path + fam_name}')
        os.system(cmd)
        k = fam_name
    os.system(f'mv {path + k} {path + start_name}')
    delete_cmd = f'rm  {path + start_name}'
    os.system(delete_cmd)
