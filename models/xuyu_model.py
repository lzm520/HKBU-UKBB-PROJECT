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
    def __init__(self, n_phenotype, n_feature):
        super().__init__()
        # weight for feature
        self.W = nn.Parameter(torch.FloatTensor(torch.randn(2, n_feature)))

        # confidence matrix for every features
        self.M = nn.Parameter(torch.FloatTensor(torch.randn(2, n_phenotype)))

    def forward(self, inputs):
        numerical_x, categorical_x = inputs
        outputs = []
        # processing numerical data
        k = 0
        j = 0
        for x in numerical_x:
            out = self.W[:, [k]] @ x.mT
            out = torch.tanh(out)
            out = torch.reshape(self.M[:, j], shape=[2, 1]) * out
            outputs.append(out)
            k += 1
            j += 1
        # processing categorical data
        for x in categorical_x:
            n = x.shape[1]
            out = self.W[:, k:k+n] @ x.mT
            out = torch.tanh(out)
            out = torch.reshape(self.M[:, j], shape=[2, 1]) * out
            outputs.append(out)
            k += n
            j += 1

        f = torch.zeros_like(outputs[0])
        for out in outputs:
            f = f + out
        f = torch.exp(f)
        f_sum = torch.sum(f, 0)
        prob = f / f_sum
        return prob


# Logistic model definition
class PRS_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.FloatTensor(torch.randn(1)))
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(1)))

    def forward(self, score):
        logit = 1 / (1 + torch.exp(-(self.w * score + self.b)))
        return logit


def model_train(data, label, PRS, numerical_phenotype_idx, categorical_phenotype_idx, val_data, val_label, val_PRS,
                num_phenotype, num_feature, param_save, train_loss_save, val_loss_save, alpha=1., beta=1., batch_size=100,
                n_epoch=1000):
    # model instance
    pred_model = myModel(num_phenotype, num_feature)
    optimizer_pred = torch.optim.Adadelta(pred_model.parameters(), rho=0.95, weight_decay=0.001)

    prs_model = PRS_model()
    optimizer_prs = torch.optim.Adadelta(prs_model.parameters(), rho=0.95, weight_decay=0.001)

    bce_loss = nn.BCELoss()
    kl_loss = nn.KLDivLoss()

    dataset = TensorDataset(data, label, PRS)
    db = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_acc_best = 0.0
    train_loss_epoch = []

    start_time = time.time()

    # training k times epoch
    for epoch in range(n_epoch):
        pred_model.train()
        prs_model.train()
        loss_batch = []
        for step, (x, y, prs) in enumerate(db):
            numerical_x = [x[:, [idx]] for idx in numerical_phenotype_idx]
            categorical_x = [x[:, idx_s:idx_e] for (idx_s, idx_e) in categorical_phenotype_idx]

            optimizer_pred.zero_grad()
            optimizer_prs.zero_grad()
            # prediction
            result_pred = pred_model((numerical_x, categorical_x)).T
            result_prs = prs_model(prs)
            # loss calculation
            kl = kl_loss(result_pred[:, 1], result_prs)
            loss_f = bce_loss(result_pred[:, 1], y)
            loss_prs = -torch.mean(y * torch.log(result_prs) + (1 - y) * torch.log(1 - result_prs))
            loss = loss_prs + alpha * kl + beta * loss_f
            loss.backward()

            # parameters update
            optimizer_pred.step()
            optimizer_prs.step()

            # loss information every batch
            loss_batch.append(loss.detach().numpy())

        # training loss information every epoch
        train_loss_epoch.append(np.mean(loss_batch))

        if np.mod(epoch, 50) == 0:
            count_time = time.time() - start_time
            print(f'run epoch: {epoch} , time consumed: {int(count_time)}s')

        # validation loss information every epoch
        val_acc = validation_acc(pred_model, prs_model, val_data, val_label, val_PRS, numerical_phenotype_idx, categorical_phenotype_idx)

        if val_acc > val_acc_best:
            val_acc_best = val_acc
            state = {'pred': pred_model.state_dict(), 'optimizer': optimizer_pred.state_dict(),
                     'prs': prs_model.state_dict(), 'optimizer_prs': optimizer_prs.state_dict(), 'epoch': epoch,
                     'alpha': alpha, 'beta': beta}
            torch.save(obj=state, f=param_save + '/model_param_g_t2d_' + str(alpha) + '_' + str(beta) + '.pth')

    # for epoch error plot
    np.save(train_loss_save + '/train_loss_t2d_' + str(alpha) + '_' + str(beta), train_loss_epoch)
    np.save(val_loss_save + '/val_loss_t2d_' + str(alpha) + '_' + str(beta), train_loss_epoch)


def validation_acc(pred_model, prs_model, val_data, val_label, val_PRS, numerical_phenotype_idx,
                   categorical_phenotype_idx):
    numerical_x = [val_data[:, [idx]] for idx in numerical_phenotype_idx]
    categorical_x = [val_data[:, idx_s:idx_e] for (idx_s, idx_e) in categorical_phenotype_idx]
    pred_model.eval()
    prs_model.eval()
    with torch.no_grad():
        val_result_pre = pred_model((numerical_x, categorical_x)).T
        val_result_prs = prs_model(val_PRS)
        PRSPR_Pred = val_result_pre[:, 1] + val_result_prs
        pred_result = PRSPR_Pred.numpy()
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


def evaluation_auc(pred_model, prs_model, data, label, PRS, param_path, numerical_phenotype_idx,
                   categorical_phenotype_idx, alpha_test, beta_test, save_result=False, result_folder=None):
    numerical_x = [data[:, [idx]] for idx in numerical_phenotype_idx]
    categorical_x = [data[:, idx_s:idx_e] for (idx_s, idx_e) in categorical_phenotype_idx]

    trained_param = param_path + '/model_param_g_t2d_' + str(alpha_test) + '_' + str(beta_test) + '.pth'
    reload_states = torch.load(trained_param)

    pred_model.load_state_dict(reload_states['pred'])
    prs_model.load_state_dict(reload_states['prs'])

    pred_model.eval()
    prs_model.eval()
    with torch.no_grad():
        result_pre = pred_model((numerical_x, categorical_x)).T
        result_prs = prs_model(PRS)
        PRSPR_Pred = result_pre[:, 1] + result_prs
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
    prs_data = prs_data[:, 0]
    return prs_data, label_data


if __name__ == '__main__':
    work_path = os.getcwd()
    seed = 9999
    setup_seed(seed)

    # data preprocessing
    disease_name = 'CAD'
    training_info_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data_info.csv'
    training_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data.npy'
    training_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_prs.npy'
    evaluation_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_data.npy'
    evaluation_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_prs.npy'
    save_path = f'{work_path}/result/xuyuModel/{disease_name}'
    is_age = True

    # training and evaluation PRS score & Label loading
    training_prs, training_label = prs_label_data_loading(training_label_path)
    evaluation_prs, evaluation_label = prs_label_data_loading(evaluation_label_path)
    training_prs, training_label, evaluation_prs, evaluation_label = \
        torch.tensor(training_prs), torch.tensor(training_label), torch.tensor(evaluation_prs), torch.tensor(evaluation_label)

    # training and testing data loading
    training_data = np.load(training_data_path).astype(np.float32)
    evaluation_data = np.load(evaluation_data_path).astype(np.float32)
    if is_age:
        training_data = training_data.T
        evaluation_data = evaluation_data.T
    else:
        training_data = training_data[1:, :].T
        evaluation_data = evaluation_data[1:, :].T
    training_data, evaluation_data = torch.tensor(training_data), torch.tensor(evaluation_data)

    # feature info loading
    data_info = pd.read_csv(training_info_path, header=None, names=['Type', 'id', 'Description'])
    if not is_age:
        data_info = data_info[1:]
    num_feature = data_info.shape[0]
    num_phenotype = len(set([i[0] for i in np.char.split(data_info['id'].to_numpy(dtype=str), '_')]))
    #  categorical phenotype data preprocessing
    numerical_phenotype_idx = []
    categorical_phenotype_idx = []
    idx = 0
    while idx < data_info.shape[0]:
        k = idx
        id = str.split(data_info.iloc[idx]['id'], sep='_')[0]
        if not data_info.iloc[idx]['Type'] == 'Categorical':
            numerical_phenotype_idx.append(idx)
            idx += 1
        else:
            while k < data_info.shape[0]:
                if str.startswith(data_info.iloc[k]['id'], id):
                    k += 1
                else:
                    break
            categorical_phenotype_idx.append((idx, k))
            idx = k

    mode = input('train or evaluation mode?\t')

    # result folder
    result = save_path
    if result is not None:
        if not os.path.isdir(result):
            os.mkdir(result)
    if is_age:
        result_folder = result + f'f/{disease_name}_200epoch_monitor_acc-age'
    else:
        result_folder = result + f'f/{disease_name}_200epoch_monitor_acc-no_age'
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

    alpha_candi = [1.0, 0.1, 0.05, 0.01, 0.005]
    beta_candi = [1.0, 0.1, 0.05, 0.01, 0.005]

    error_input = False

    if mode == 'train':
        auc_txt = open(result_folder + '/auc_collected.txt', 'w')
        auc_txt.write('alpha\tbeta\tAUROC\n')
        for i in range(len(alpha_candi)):
            for j in range(len(beta_candi)):
                # train model
                model_train(data=training_data, label=training_label, PRS=training_prs,
                            numerical_phenotype_idx=numerical_phenotype_idx,
                            categorical_phenotype_idx=categorical_phenotype_idx,
                            val_data=evaluation_data, val_label=evaluation_label, val_PRS=evaluation_prs,
                            num_phenotype=num_phenotype, num_feature=num_feature,
                            param_save=param_path, train_loss_save=train_loss_path, val_loss_save=val_loss_path,
                            alpha=alpha_candi[i], beta=beta_candi[j], n_epoch=200)
                # test after training
                eval_pred_model = myModel(num_phenotype, num_feature)
                eval_prs_model = PRS_model()
                auc_test = evaluation_auc(pred_model=eval_pred_model, prs_model=eval_prs_model,
                                          data=evaluation_data, label=evaluation_label, PRS=evaluation_prs,
                                          param_path=param_path,
                                          numerical_phenotype_idx=numerical_phenotype_idx,
                                          categorical_phenotype_idx=categorical_phenotype_idx, alpha_test=alpha_candi[i],
                                          beta_test=beta_candi[j], save_result=False, result_folder=result_folder)
                print('test auc:', auc_test)
                auc_txt.write(f'{str(alpha_candi[i])}\t{str(beta_candi[j])}\t{str(auc_test)}\n')
        auc_txt.close()

    elif mode == 'evaluation':
        alpha_input = input('alphe select(1.0, 0.1, 0.05, 0.01, 0.005)\t')
        if float(alpha_input) not in alpha_candi:
            print('no this alpha value')
            error_input = True

        beta_input = input('beta select(1.0, 0.1, 0.05, 0.01, 0.005)\t')
        if float(beta_input) not in beta_candi:
            print('no this beta value')
            error_input = True

        sava_evaluation = input('save result?(yes/no)\t')
        if sava_evaluation == 'yes':
            sava_evaluation = True
        elif sava_evaluation == 'no':
            sava_evaluation = False
        else:
            print('no this choice')
            error_input = True

        if not error_input:
            eval_pred_model = myModel(num_phenotype, num_feature)
            eval_prs_model = PRS_model()
            auc_test = evaluation_auc(pred_model=eval_pred_model, prs_model=eval_prs_model,
                                      data=evaluation_data, label=evaluation_label, PRS=evaluation_prs,
                                      param_path=param_path,
                                      numerical_phenotype_idx=numerical_phenotype_idx,
                                      categorical_phenotype_idx=categorical_phenotype_idx, alpha_test=alpha_input,
                                      beta_test=beta_input, save_result=sava_evaluation, result_folder=result_folder)
            print('test auc:', auc_test)
        else:
            print('alpha/beta input error')
    else:
        print('Input error. Please restart')