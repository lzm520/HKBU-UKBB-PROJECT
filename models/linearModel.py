import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from torch.backends import cudnn
import time
from sklearn.metrics import roc_auc_score


class myModel(nn.Module):
    def __init__(self, n_phenotype):
        super().__init__()
        # weight for categorical phenotype
        self.W = nn.Parameter(torch.FloatTensor(torch.randn(2, n_phenotype)))

    def forward(self, inputs):
        outputs = []
        # processing data
        k = 0
        for i in range(inputs.shape[1]):
            x = inputs[:, [i]]
            out = self.W[:, [i]] @ x.mT
            outputs.append(out)
            k += 1

        f = torch.zeros_like(outputs[0])
        for out in outputs:
            f = f + out
        f = torch.exp(f)
        f_sum = torch.sum(f, 0)
        prob = f / f_sum
        return prob


def model_train(data, label, val_data, val_label, num_phenotype,
                param_save, train_loss_save, val_loss_save, batch_size=100, n_epoch=1000):
    # model instance
    pred_model = myModel(num_phenotype)
    optimizer_pred = torch.optim.Adadelta(pred_model.parameters(), rho=0.95, weight_decay=0.001)

    bce_loss = nn.BCELoss()

    dataset = TensorDataset(data, label)
    db = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_acc_best = 0.0
    train_loss_epoch = []

    start_time = time.time()

    # training k times epoch
    for epoch in range(n_epoch):
        pred_model.train()
        loss_batch = []
        for step, (x, y) in enumerate(db):
            optimizer_pred.zero_grad()
            # prediction
            result_pred = pred_model(x).T
            # loss calculation
            loss = bce_loss(result_pred[:, 1], y)
            loss.backward()

            # parameters update
            optimizer_pred.step()

            # loss information every batch
            loss_batch.append(loss.detach().numpy())

        # training loss information every epoch
        train_loss_epoch.append(np.mean(loss_batch))

        if np.mod(epoch, 50) == 0:
            count_time = time.time() - start_time
            print(f'run epoch: {epoch} , time consumed: {int(count_time)}s')

        # validation loss information every epoch
        val_acc = validation_acc(pred_model, val_data, val_label)

        if val_acc > val_acc_best:
            val_acc_best = val_acc
            state = {'pred': pred_model.state_dict(), 'optimizer': optimizer_pred.state_dict()}
            torch.save(obj=state, f=param_save + '/model_param.pth')

    # for epoch error plot
    np.save(train_loss_save + '/train_loss', train_loss_epoch)
    np.save(val_loss_save + '/val_loss', train_loss_epoch)


def validation_acc(pred_model, val_data, val_label):
    pred_model.eval()
    with torch.no_grad():
        val_result_pre = pred_model(val_data).T
        pred_result = val_result_pre[:, 1].numpy()
        threshold = np.mean(pred_result)
    classification = []
    for i in range(len(val_label)):
        if pred_result[i] >= threshold:
            classification.append(1)
        else:
            classification.append(0)
    val_acc = np.sum(np.array(classification) == val_label.numpy())
    final_val_acc = val_acc / len(val_label)

    return final_val_acc


def evaluation_auc(pred_model, data, label, param_path, save_result=False, result_folder=None):

    trained_param = param_path + '/model_param.pth'
    reload_states = torch.load(trained_param)

    pred_model.load_state_dict(reload_states['pred'])

    pred_model.eval()
    with torch.no_grad():
        result_pre = pred_model(data).T
        PRSPR_Pred = result_pre[:, 1]
    result_list = PRSPR_Pred.numpy()
    label_list = label

    if save_result:
        np.save(result_folder + '/y_test', label)
        np.save(result_folder + '/y_pred', result_list)
        f1 = open(result_folder + '/y_test.txt', 'w')
        f2 = open(result_folder + '/y_pred', 'w')
        for i in range(len(label_list)):
            f1.write(str(label_list[i]) + '\n')
            f2.write(str(result_list[i]) + '\n')
        f1.close()
        f2.close()
    auc = roc_auc_score(label, result_list)

    return auc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True


def prs_label_data_loading(file_path):
    prs_data = np.load(file_path).astype(np.float32)
    label_data = prs_data[:, 1]
    prs_data = prs_data[:, [0]].T
    return prs_data, label_data


if __name__ == '__main__':
    work_path = os.getcwd()
    seed = 9999
    setup_seed(seed)
    # data preprocessing
    disease_name = 'Type2-Diabetes'
    training_info_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data_info.csv'
    training_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data.npy'
    training_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_prs.npy'
    evaluation_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_data.npy'
    evaluation_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_prs.npy'
    save_path = f'{work_path}/result/linearModel/{disease_name}'

    ensemble_all_phenotype_flag = False
    add_age_flag = True
    add_prs_flag = True

    # creating save path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # training and evaluation PRS score & Label loading
    training_prs, training_label = prs_label_data_loading(training_label_path)
    evaluation_prs, evaluation_label = prs_label_data_loading(evaluation_label_path)
    training_label, evaluation_label = torch.tensor(training_label), torch.tensor(evaluation_label)

    # training and testing data loading
    training_data = np.load(training_data_path).astype(np.float32)
    evaluation_data = np.load(evaluation_data_path).astype(np.float32)
    training_data = np.concatenate((training_prs, training_data), axis=0)
    evaluation_data = np.concatenate((evaluation_prs, evaluation_data), axis=0)

    # feature info loading
    data_info = pd.read_csv(training_info_path, header=None, names=['Type', 'id', 'Description'])

    # standardise training and evaluation data
    # save means and standard deviation
    t_ms = []
    t_stds = []
    for i in range(data_info.shape[0]):
        type = data_info.loc[i, 'Type'].strip()
        if type in ['Integer', 'Continuous']:
            # normalize training data
            t_m = np.mean(training_data[i])
            t_std = np.std(training_data[i])
            training_data[i] = (training_data[i] - t_m) / t_std
            # normalize evaluation data
            evaluation_data[i] = (evaluation_data[i] - t_m) / t_m
        else:
            t_m = 0
            t_std = 1
        t_ms.append(t_m)
        t_stds.append(t_std)
    training_data, evaluation_data = torch.tensor(training_data.T), torch.tensor(evaluation_data.T)
    num_phenotype = training_data.shape[1]

    # result folder
    result_folder = save_path
    if result_folder is not None:
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
    param_path = result_folder + '/param'
    if param_path is not None:
        if not os.path.isdir(param_path):
            os.mkdir(param_path)
    train_loss_path = result_folder + '/loss'
    val_loss_path = result_folder + '/loss'
    if train_loss_path is not None:
        if not os.path.isdir(train_loss_path):
            os.mkdir(train_loss_path)

    auc_txt = open(result_folder + '/auc_collected.txt', 'w')
    auc_txt.write('AUROC\n')
    # train model
    model_train(data=training_data, label=training_label,  val_data=evaluation_data, val_label=evaluation_label,
                num_phenotype=num_phenotype, param_save=param_path,
                train_loss_save=train_loss_path, val_loss_save=val_loss_path, n_epoch=200)
    # test after training
    eval_pred_model = myModel(num_phenotype)
    auc_test = evaluation_auc(pred_model=eval_pred_model,
                              data=evaluation_data, label=evaluation_label,
                              param_path=param_path, save_result=False, result_folder=result_folder)
    print('test auc:', auc_test)
    auc_txt.write(f'{str(auc_test)}\n')
    auc_txt.close()
