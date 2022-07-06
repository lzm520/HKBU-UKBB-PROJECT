import os
import random
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.backends import cudnn
import time
from sklearn.metrics import roc_auc_score


class myRnn(nn.Module):
    def __init__(self, icd_size, icd_embed_size, hidden_size, num_layers, dense_input_size):
        super(myRnn, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(icd_size, icd_embed_size)
        self.rnn = nn.GRU(input_size=icd_embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.dense = nn.Linear(dense_input_size, 1)

    def forward(self, icd_x, prs_phenotype_x):
        X = self.embedding(icd_x)
        output, h_n = self.rnn(X)
        output = output[[-1], :]
        dense_input = torch.cat((prs_phenotype_x, output), dim=1)
        res = torch.sigmoid(self.dense(dense_input))
        return res


def model_training(save_path, model, training_icd_x, training_prs_phenotype_x, training_y,
                   evaluation_icd_x, evaluation_prs_phenotype_x, evaluation_y, epochs, batch_sz=256, device='cuda'):
    model = model.to(device)

    param_save = save_path + '/param'
    if param_save is not None:
        if not os.path.isdir(param_save):
            os.mkdir(param_save)
    acc_auc_fp = open(save_path + '/acc_auroc.txt', 'w')
    acc_auc_fp.write('\t'.join(['epoch', 'loss', 'accuracy', 'auroc']) + '\n')

    training_index = [i for i in range(len(training_icd_x))]
    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=0.001)
    loss_fc = nn.BCELoss()
    val_acc_best = 0.
    val_auc_best = 0.

    start_time = time.time()
    for epoch in range(epochs):
        random.shuffle(training_index)
        model.train()
        y_had = torch.tensor([])
        y_had = y_had.to(device)
        idx_ls = []
        for i, idx in enumerate(training_index):
            # data
            icd_x = torch.tensor(training_icd_x[idx]).to(torch.int32)
            if len(icd_x) == 0:
                continue
            icd_x = icd_x.to(device)
            prs_phenotype_x = training_prs_phenotype_x[[idx], :]
            prs_phenotype_x = prs_phenotype_x.to(device)
            idx_ls.append(idx)
            # prediction
            res = model(icd_x, prs_phenotype_x).squeeze(dim=0)
            y_had = torch.cat((y_had, res), dim=0)

            if np.mod(idx, batch_sz) == 0 or i + 1 == len(training_index):
                y = training_y[idx_ls]
                y = y.to(device)
                loss = loss_fc(y_had, y)
                loss.backward()
                # param update
                optimizer.step()
                optimizer.zero_grad()
                idx_ls = []
                y_had = torch.tensor([])
                y_had = y_had.to(device)

        val_loss, val_acc, val_auc = validation_loss_acc_auc(model, loss_fc, evaluation_icd_x,
                                                             evaluation_prs_phenotype_x, evaluation_y, device)
        acc_auc_fp.write('\t'.join([str(epoch), str(val_loss), str(val_acc), str(val_auc)]) + '\n')
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            val_auc_best = val_auc
            state = {'myRNN': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'batch_size': batch_sz}
            torch.save(obj=state, f=param_save + '/myRNN_model_param.pth')

        if np.mod(epoch, 20) == 0:
            count_time = time.time() - start_time
            print(
                f'run epoch: {epoch} , time consumed: {int(count_time)}s, acc_best: {val_acc_best}, auc_best: {val_auc_best}')

    acc_auc_fp.close()


def validation_loss_acc_auc(model, loss_fc, evaluation_icd_x, evaluation_prs_phenotype_x, evaluation_y, device):
    model.eval()
    evaluation_index = [i for i in range(len(evaluation_icd_x))]

    y_had = torch.tensor([])
    y_had = y_had.to(device)
    idx_ls = []
    for idx in evaluation_index:
        icd_x = torch.tensor(evaluation_icd_x[idx]).to(torch.int32)
        if len(icd_x) == 0:
            continue
        icd_x = icd_x.to(device)
        prs_phenotype_x = evaluation_prs_phenotype_x[[idx], :]
        prs_phenotype_x = prs_phenotype_x.to(device)
        idx_ls.append(idx)

        with torch.no_grad():
            res = model(icd_x, prs_phenotype_x).squeeze(dim=0)
            y_had = torch.cat((y_had, res), dim=0)

    # calculate loss
    y = evaluation_y[idx_ls]
    y = y.to(device)
    loss = loss_fc(y_had, y)

    pred_result = y_had.cpu().numpy()
    label_list = y.cpu().numpy()
    threshold = np.mean(pred_result)
    classification = []
    for i in range(len(label_list)):
        if pred_result[i] >= threshold:
            classification.append(1)
        else:
            classification.append(0)
    val_acc = np.sum(np.array(classification) == label_list)
    final_val_acc = val_acc / len(label_list)
    auc = roc_auc_score(label_list, pred_result)
    return loss, final_val_acc, auc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True


def prs_label_data_loading(file_path):
    prs_data = np.load(file_path).astype(np.float32)
    label_data = prs_data[:, 1]
    prs_data = prs_data[:, [0]]
    return prs_data, label_data


if __name__ == '__main__':
    work_path = os.getcwd()
    seed = 9999
    setup_seed(seed)

    # data loading
    disease_name = 'Type2-Diabetes'
    training_info_path = f'./data/{disease_name}/model_training_data/training_data_info.csv'
    training_phenotype_data_path = f'./data/{disease_name}/model_training_data/training_data.npy'
    training_label_path = f'./data/{disease_name}/model_training_data/training_prs.npy'
    evaluation_phenotype_data_path = f'./data/{disease_name}/model_training_data/evaluation_data.npy'
    evaluation_label_path = f'./data/{disease_name}/model_training_data/evaluation_prs.npy'
    training_eid_path = f'./data/{disease_name}/model_training_data/training_eid.npy'
    evaluation_eid_path = f'./data/{disease_name}/model_training_data/evaluation_eid.npy'
    training_eid_icd9_save_path = f'./data/{disease_name}/model_training_data/training_eid_icd9.txt'
    training_eid_icd10_save_path = f'./data/{disease_name}/model_training_data/training_eid_icd10.txt'
    training_eid_icd9_diag_info_save_path = f'./data/{disease_name}/model_training_data/training_eid_icd9_diag_info.txt'
    training_eid_icd10_diag_info_save_path = f'./data/{disease_name}/model_training_data/training_eid_icd10_diag_info.txt'
    evaluation_eid_icd9_diag_info_save_path = f'./data/{disease_name}/model_training_data/evaluation_eid_icd9_diag_info.txt'
    evaluation_eid_icd10_diag_info_save_path = f'./data/{disease_name}/model_training_data/evaluation_eid_icd10_diag_info.txt'
    save_path = f'{work_path}/result/myRNN/{disease_name}'

    hidden_size = 512
    icd_embeded_szie = 256
    batch_size = 256
    n_epoch = 1000
    device = 'cuda:0'

    # training & evaluation PRS score & Label loading
    training_prs, training_label = prs_label_data_loading(training_label_path)
    evaluation_prs, evaluation_label = prs_label_data_loading(evaluation_label_path)
    training_prs, training_label, evaluation_prs, evaluation_label = \
        torch.tensor(training_prs), torch.tensor(training_label), torch.tensor(evaluation_prs), torch.tensor(
            evaluation_label)

    # training & evaluation phenotype data loading
    training_data = torch.FloatTensor(np.load(training_phenotype_data_path).astype(np.float32).T)
    evaluation_data = torch.FloatTensor(np.load(evaluation_phenotype_data_path).astype(np.float32).T)

    # concat prs and phenotype
    training_prs_phenotype_x = torch.cat((training_prs, training_data), dim=1)
    evaluation_prs_phenotype_x = torch.cat((evaluation_prs, evaluation_data), dim=1)

    # training & evaluation eid loading
    training_eid = np.load(training_eid_path).astype(str)
    evaluation_eid = np.load(evaluation_eid_path).astype(str)

    # loading icd9 & icd10 names
    icd9_cols = pd.read_table(training_eid_icd9_save_path).columns.values
    icd10_cols = pd.read_table(training_eid_icd10_save_path).columns.values
    icd_cols = np.concatenate((icd9_cols, icd10_cols))

    # loading training & evaluation eid_icd data
    training_eid_icd9_diag = pd.read_table(training_eid_icd9_diag_info_save_path).rename(columns={'icd9': 'icd'})
    training_eid_icd10_diag = pd.read_table(training_eid_icd10_diag_info_save_path).rename(columns={'icd10': 'icd'})
    evaluation_eid_icd9_diag = pd.read_table(evaluation_eid_icd9_diag_info_save_path).rename(columns={'icd9': 'icd'})
    evaluation_eid_icd10_diag = pd.read_table(evaluation_eid_icd10_diag_info_save_path).rename(columns={'icd10': 'icd'})

    # data preprocessing
    training_icd_x, evaluation_icd_x = [], []
    for eid in training_eid:
        eid_icd9_diag = training_eid_icd9_diag[training_eid_icd9_diag['eid'] == int(float(eid))]
        eid_icd10_diag = training_eid_icd10_diag[training_eid_icd10_diag['eid'] == int(float(eid))]
        eid_icd_diag = pd.concat((eid_icd9_diag, eid_icd10_diag))
        eid_icd_diag = eid_icd_diag.sort_values(by=['disdate'])
        icd = eid_icd_diag['icd'].values
        x = [np.where(icd_cols == icd)[0][0] for icd in icd]
        training_icd_x.append(x)
    for eid in evaluation_eid:
        eid_icd9_diag = evaluation_eid_icd9_diag[evaluation_eid_icd9_diag['eid'] == int(float(eid))]
        eid_icd10_diag = evaluation_eid_icd10_diag[evaluation_eid_icd10_diag['eid'] == int(float(eid))]
        eid_icd_diag = pd.concat((eid_icd9_diag, eid_icd10_diag))
        eid_icd_diag = eid_icd_diag.sort_values(by=['disdate'])
        icd = eid_icd_diag['icd'].values
        x = [np.where(icd_cols == icd)[0][0] for icd in icd]
        evaluation_icd_x.append(x)

    # result folder
    result = save_path
    if result is not None:
        if not os.path.isdir(result):
            os.mkdir(result)

    print(len(training_icd_x))
    print(training_prs_phenotype_x.shape)
    print(len(evaluation_icd_x))
    print(evaluation_prs_phenotype_x.shape)

    model = myRnn(len(icd_cols), icd_embeded_szie, hidden_size, 2, hidden_size + training_prs_phenotype_x.shape[1])
    model_training(save_path, model, training_icd_x, training_prs_phenotype_x, training_label,
                   evaluation_icd_x, evaluation_prs_phenotype_x, evaluation_label,
                   epochs=n_epoch, batch_sz=batch_size, device=device)
