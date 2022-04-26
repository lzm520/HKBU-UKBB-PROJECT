import csv

eid_infile = '/tmp/local/cszmli/data/field_extraction/eids.csv'
eid_to_idx = {}
with open(eid_infile, 'r', encoding='utf-8',) as fp:
    reader = csv.reader(fp)
    header = next(reader)
    for idx, line in enumerate(reader):
        eid_to_idx[line[0]] = idx
print('eid_to_idx读取完成')

features_selection_data_infile = '/tmp/local/cszmli/data/features_selection/features_selection_data_dummy_data.txt'
features_selection_info_infile = '/tmp/local/cszmli/data/features_selection/features_selection_data_dummy_info.csv'
features_info = []
with open(features_selection_info_infile, 'r', encoding='utf-8') as fp:
    reader = csv.reader(fp)
    for line in reader:
        features_info.append([line[0], line[1]])
features_data = []
with open(features_selection_data_infile, 'r') as fp:
    for line in fp:
        data = line.strip().split()
        features_data.append(data)
print('特征数据读取完成')

fam_infile = '/tmp/local/cszmli/plink/standBy.fam'
fam_value = []
with open(fam_infile, 'r', encoding='utf-8') as fp:
    for line in fp:
        values = line.strip().split('\t')
        fam_value.append(values)
print('fam文件读取完成')

for i, info in enumerate(features_info):
    col_ID = info[1]
    out_file = f'/tmp/local/cszmli/plink/fam_folder/UKBB_col_id{col_ID}.fam'
    with open(out_file, 'w') as fp:
        for value in fam_value:
            if eid_to_idx.get(value[0]) is not None:
                idx = eid_to_idx[value[0]]
                if info[0] == 'Categorical':
                    val = str(int(float(features_data[i][idx])))
                    if val == '1':
                        value[-1] = '2'
                    else:
                        value[-1] = '1'
                else:
                    value[-1] = features_data[i][idx]
            fp.write('\t'.join(value))
            fp.write('\n')
    print(f'UKBB_col_id{col_ID}.fam写入完成')
