import os

for chro in range(1, 23):
    bgen = f'/home/comp/ericluzhang/UKBB/ukb_imp_chr{chro}_v3.bgen'
    sample = f'/home/comp/ericluzhang/UKBB/ukb60434_imp_chr{chro}_v3_s487280.sample'
    out = f'/tmp/local/cszmli/plink/UKBB_chr{chro}'
    cmd = f'plink2 --bgen {bgen} ref-first --sample {sample} --make-bed --out {out} '
    os.system(cmd)

A1 = '/home/comp/ericluzhang/UKBB/ukb_imp_chr22_v3.bgen'
A2 = '/home/comp/ericluzhang/UKBB/ukb60434_imp_chr22_v3_s487280.sample'
A3 = '/tmp/local/cszmli/plink/UKBB_chr22'
os.system(f'/home/comp/cszmli/LZM/plink2 --bgen {A1} ref-first --sample {A2} --make-bed --out {A3} ')

B1 = '/home/comp/ericluzhang/UKBB/ukb_imp_chrX_v3.bgen'
B2 = '/home/comp/ericluzhang/UKBB/ukb60434_imp_chrX_v3_s486629.sample'
B3 = '/tmp/local/cszmli/plink/UKBB_chrX'
os.system(f'/home/comp/cszmli/LZM/plink2 --bgen {B1} ref-first --sample {B2} --make-bed --out {B3} ')

C1 = '/home/comp/ericluzhang/UKBB/ukb_imp_chrXY_v3.bgen'
C2 = '/home/comp/ericluzhang/UKBB/ukb60434_imp_chrXY_v3_s486315.sample'
C3 = '/tmp/local/cszmli/plink/UKBB_chrXY'
os.system(f'/home/comp/cszmli/LZM/plink2 --bgen {C1} ref-first --sample {C2} --make-bed --out {C3} ')