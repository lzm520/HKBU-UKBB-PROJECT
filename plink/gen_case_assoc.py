import os

path = '/tmp/local/cszmli/plink/'
start_name = 'UKBB_case.fam'
print('Now processing file: ', start_name)
k = start_name
if not os.path.exists(path + 'UKBB_case_assoc/'):
    os.mkdir(path + 'UKBB_case_assoc/')
for i in range(22, 0, -1):
    fam_name = f'UKBB_chr{i}.fam'
    cmd = f'/home/comp/cszmli/LZM/plink --bfile /tmp/local/cszmli/plink/UKBB_chr{i} --geno 0.1 --hwe 0.000001 --maf ' \
          '0.01 --mind 0.1 --pfilter 0.000001 --logistic --ci 0.95 --out ' \
          f'/tmp/local/cszmli/plink/UKBB_case_assoc/UKBB_chr{i}_case --allow-no-covars '
    os.system(f'mv {path + k} {path + fam_name}')
    os.system(cmd)
    k = fam_name

os.system(f'mv {path + k} {path + start_name}')
