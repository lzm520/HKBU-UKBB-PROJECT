import csv

eid_filter_file = '../../data/eid_filter/eid_filter.csv'
case = dict()
with open(eid_filter_file, 'r', encoding='utf-8') as fp:
    reader = csv.reader(fp)
    header = next(reader)
    for line in reader:
        case[line[1]] = line[0]
print('病例人员文件读取完成')

fam_infile = '/tmp/local/cszmli/plink/standBy.fam'
fam_value = []
with open(fam_infile, 'r', encoding='utf-8') as fp:
    for line in fp:
        values = line.strip().split('\t')
        fam_value.append(values)
print('fam文件读取完成')

out_file = '/tmp/local/cszmli/plink/UKBB_case.fam'
with open(out_file, 'w') as fp:
    for value in fam_value:
        if case.get(value[0]) is not None:
            value[-1] = '2'
        else:
            value[-1] = '1'
        fp.write('\t'.join(value))
        fp.write('\n')
print('UKBB_case.fam写入完成')