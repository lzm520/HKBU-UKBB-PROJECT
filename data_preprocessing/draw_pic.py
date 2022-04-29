import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import csv


# logistRegression_pic
def logit_pic(X, y, title):
    logist = linear_model.LogisticRegression()
    logist.fit(X, y)
    x_pred = np.linspace(np.min(X), np.max(X), 100).reshape((-1, 1))
    y_prob = logist.predict_proba(x_pred)[:, 1]
    plt.plot(x_pred, y_prob, 'b.-')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    data_fiile = '../data/features_selection/features_selection_data.txt'
    read_file = '../../data/features_selection/features_selection_info.csv'
    eid_filter_file = '../../data/eid_filter/eid_filter.csv'
    find_filed_id = '48'

    n_participants = 502505
    field_id_ukb_idx = np.genfromtxt(eid_filter_file, delimiter=',', dtype=np.int32)[1:, 0]
    y = np.zeros(n_participants)
    y[field_id_ukb_idx] = 1

    find_idx = 0
    with open(read_file, 'r') as fp:
        reader = csv.reader(fp)
        for idx, line in enumerate(reader):
            if len(line) == 0:
                continue
            if line[1] == find_filed_id:
                find_idx = idx
                break

    X = None
    with open(data_fiile, 'r') as fp:
        for idx, line in enumerate(fp):
            if line == '\n':
                continue
            if idx == find_idx:
                X = line.strip().split()
                break
    X = np.array(X, dtype=float)
    X = X.reshape((-1, 1))
    print(X.shape)
    print(y.shape)
    logit_pic(X, y, 'field_'+find_filed_id+' logistic regression curve')
