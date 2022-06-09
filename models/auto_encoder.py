import time

import torch
from torch import nn
from torch.utils import data
import pandas as pd
import numpy as np
from d2l import torch as d2l
from sklearn.metrics import roc_auc_score


class encoder_decoder(nn.Module):
    def __init__(self, input_dim, middle_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 60), nn.ReLU(), nn.Dropout(),
                                     nn.Linear(60, 45), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(45, middle_dim))
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(middle_dim, 60),
                                     nn.ReLU(), nn.Linear(60, input_dim))

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out


class prediction_model(nn.Module):
    def __init__(self, input_dim, middle_dim=35):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 60), nn.ReLU(), nn.Dropout(),
                                     nn.Linear(60, 45), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(45, middle_dim))
        self.fullyConnectedLayer = nn.Sequential(nn.ReLU(), nn.Linear(middle_dim + 1, 20), nn.ReLU(),
                                                 nn.Linear(20, 1), nn.Sigmoid())

    def forward(self, x_prs):
        out = self.encoder(x_prs[0])
        out = torch.cat((out, x_prs[1]), dim=1)
        out = self.fullyConnectedLayer(out)
        return out


def auto_encoder_training(save_encoder_path, training_x, evaluation_x, input_dim, middle_dim=35, learning_rate=0.001,
                          epochs=25, batch_sz=256, device='cpu'):
    training_x = torch.tensor(training_x).T
    evaluation_x = torch.tensor(evaluation_x).T
    train_iter = data.DataLoader(training_x, batch_size=batch_sz, shuffle=True)
    en_de_model = encoder_decoder(input_dim=input_dim, middle_dim=middle_dim)
    loss_fc = nn.MSELoss()
    optimizer_fc = torch.optim.Adam(en_de_model.parameters(), lr=learning_rate)
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs], legend=['train loss', 'test loss'])
    # num_batches = len(train_iter)
    max_loss = 999999
    for epoch in range(epochs):
        start_time = time.time()
        en_de_model.train()
        metric = d2l.Accumulator(2)
        for i, x in enumerate(train_iter):
            optimizer_fc.zero_grad()
            x = x.to(device)
            out = en_de_model(x)
            # L1 regularization
            regularization_loss = 0
            for param in en_de_model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            train_loss = loss_fc(out, x) + regularization_loss
            train_loss.backward()
            optimizer_fc.step()
            with torch.no_grad():
                metric.add(train_loss * x.shape[0], x.shape[0])
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches, (train_loss, None))
        en_de_model.eval()
        with torch.no_grad():
            out = en_de_model(evaluation_x)
            test_loss = loss_fc(out, evaluation_x)
            # animator.add(epoch + 1, (None, test_loss))
            if test_loss < max_loss:
                torch.save(en_de_model.encoder.state_dict(), save_encoder_path)
                max_loss = test_loss
        print(f'epoch{epoch + 1}, train loss:{metric[0] / metric[1]}, test loss:{test_loss}, time:{time.time() - start_time}s')


def prediction_model_training(save_prediction_model_path, pre_model, training_x, training_y, training_prs, evaluation_x,
                              evaluation_y, evaluation_prs, learning_rate=0.001, epochs=25, batch_sz=256, device='cpu'):
    training_x = torch.tensor(training_x).T
    training_y = torch.tensor(training_y)
    training_prs = torch.tensor(training_prs)
    evaluation_x = torch.tensor(evaluation_x).T.to(device)
    evaluation_y = torch.tensor(evaluation_y).to(device)
    evaluation_prs = torch.tensor(evaluation_prs).to(device)

    train_dateset = data.TensorDataset(training_x, training_prs, training_y)
    train_iter = data.DataLoader(train_dateset, batch_size=batch_sz, shuffle=True)
    loss_fc = nn.BCELoss()
    optimizer_fc = torch.optim.Adam(pre_model.parameters(), lr=learning_rate)
    max_loss = 999999
    for epoch in range(epochs):
        start_time = time.time()
        pre_model.train()
        metric = d2l.Accumulator(3)
        for i, (x, prs, y) in enumerate(train_iter):
            optimizer_fc.zero_grad()
            x = x.to(device)
            prs = prs.to(device)
            y = y.to(device)
            out = pre_model((x, prs))
            # L2 regularization
            regularization_loss = 0
            for param in pre_model.parameters():
                regularization_loss += torch.sqrt(torch.sum(torch.pow(param, 2)))
            train_loss = loss_fc(out, y) + regularization_loss
            train_loss.backward()
            optimizer_fc.step()
            with torch.no_grad():
                metric.add(train_loss * x.shape[0], x.shape[0])
        pre_model.eval()
        with torch.no_grad():
            out = pre_model((evaluation_x, evaluation_prs))
            test_loss = loss_fc(out, evaluation_y)
            if test_loss < max_loss:
                torch.save(pre_model.state_dict(), save_prediction_model_path)
                max_loss = test_loss
            print(f'epoch{epoch + 1}, train loss:{metric[0] / metric[1]}, test loss:{test_loss}, time:{time.time() - start_time}s')


def evaluation_prediction_model(pre_model, evaluation_x, evaluation_y, evaluation_prs, device='cpu'):
    pre_model.eval()
    pre_model.to(device)
    evaluation_x = torch.tensor(evaluation_x).T.to(device)
    evaluation_prs = torch.tensor(evaluation_prs).to(device)
    with torch.no_grad():
        prob = pre_model((evaluation_x, evaluation_prs))
    prob = torch.squeeze(prob).numpy()
    auroc = roc_auc_score(evaluation_y, prob)
    return auroc


if __name__ == '__main__':
    # data preprocessing
    disease_name = 'Type2-Diabetes'
    training_info_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data_info.csv'
    training_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_data.npy'
    training_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/training_prs.npy'
    evaluation_data_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_data.npy'
    evaluation_label_path = f'/tmp/local/cszmli/data/{disease_name}/model_training_data/evaluation_prs.npy'
    save_encoder_path = f'./result/autoEncoder/{disease_name}/encoder.params'
    save_prediction_model_path = f'./result/autoEncoder/{disease_name}/prediction_model.params'
    save_auroc_file_path = f'./result/autoEncoder/{disease_name}/auto_encoder_result.txt'

    #  data info loading
    training_info = pd.read_csv(training_info_path, names=['Type', 'id', 'Description'], header=None)
    # training data loading
    training_data = np.load(training_data_path).astype(np.float32)
    # training prs & label loading
    training_prs_label = np.load(training_label_path).astype(np.float32)
    training_prs = np.reshape(training_prs_label[:, 0], newshape=[-1, 1])
    training_label = np.reshape(training_prs_label[:, 1], newshape=[-1, 1])
    # evaluation data loading
    evaluation_data = np.load(evaluation_data_path).astype(np.float32)
    # evaluation prs & label loading
    evaluation_prs_label = np.load(evaluation_label_path).astype(np.float32)
    evaluation_prs = np.reshape(evaluation_prs_label[:, 0], newshape=[-1, 1])
    evaluation_label = np.reshape(evaluation_prs_label[:, 1], newshape=[-1, 1])

    # save means and standard deviation
    t_ms = []
    t_stds = []
    for i in range(training_info.shape[0]):
        type = training_info.loc[i, 'Type'].strip()
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

    # auto encoder training
    print('Now training encoder Model')
    auto_encoder_training(save_encoder_path=save_encoder_path, training_x=training_data, evaluation_x=evaluation_data,
                          input_dim=70, middle_dim=35)

    # prediction model training
    print('\nNow training prediction Model')
    pre_model = prediction_model(input_dim=70, middle_dim=35)
    pre_model.encoder.load_state_dict(torch.load(save_encoder_path))
    prediction_model_training(save_prediction_model_path=save_prediction_model_path,
                              pre_model=pre_model, training_x=training_data, training_y=training_label,
                              training_prs=training_prs, evaluation_x=evaluation_data, evaluation_y=evaluation_label,
                              evaluation_prs=evaluation_prs)

    # evaluate prediction model
    pre_model = prediction_model(input_dim=70, middle_dim=35)
    pre_model.load_state_dict(torch.load(save_prediction_model_path))
    auroc = evaluation_prediction_model(pre_model=pre_model, evaluation_x=evaluation_data,
                                        evaluation_y=evaluation_label, evaluation_prs=evaluation_prs)
    f1 = open(save_auroc_file_path, 'w')
    f1.write('AUROC')
    f1.write('\n')
    f1.write(str(auroc))
    f1.close()