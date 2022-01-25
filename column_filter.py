import csv


# 从ukb41910文件中将字段类型为Categorical, Integer, Continuous的字段抽出来
def Cols_filter_type():
    outfp = open('../data/cols_filter.txt', 'w')
    with open('../data/cols_type.txt', 'r') as fp:
        for line in fp:
            if line == '\n':
                continue
            row = line.strip().split('\t')
            if row[0] in ['Categorical', 'Integer', 'Continuous']:
                outfp.write(line)
    outfp.close()


def lifeStyle_and_physical_measures_cols_filter():
    infile1 = '../data/cols_type.txt'
    infile2 = '../data/data_category/Lifestyle/Lifestyle.csv'
    infile3 = '../data/data_category/Physical measures/Physical measures.csv'
    outfile = '../data/cols_filter/lifeStyle_and_physical_measures_cols_filter.csv'
    outfp = open(outfile, 'w', newline='')
    outWriter = csv.writer(outfp)

    cols_type = {}
    with open(infile1, 'r') as fp:
        for line in fp:
            if line == '\n':
                continue
            row = line.strip().split('\t')
            cols_type[row[1]] = row[0]

    with open(infile2, 'r') as fp:
        reader = csv.reader(fp)
        for line in reader:
            if cols_type.get(line[0]):
                cat = cols_type[line[0]]
                line.insert(0, cat)
                outWriter.writerow(line)

    with open(infile3, 'r') as fp:
        reader = csv.reader(fp)
        for line in reader:
            if cols_type.get(line[0]):
                cat = cols_type[line[0]]
                line.insert(0, cat)
                outWriter.writerow(line)

    outfp.close()


if __name__ == '__main__':
    lifeStyle_and_physical_measures_cols_filter()
