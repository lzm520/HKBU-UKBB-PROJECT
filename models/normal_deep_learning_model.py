import time

import torch
from torch import nn
from torch.utils import data
import pandas as pd
import numpy as np
from d2l import torch as d2l
from sklearn.metrics import roc_auc_score


class prediction_model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fullyConnectedLayer = nn.Sequential(nn.Linear(input_dim + 1, 50), nn.Sigmoid(),
                                                 nn.Linear(50, 30), nn.Sigmoid(), nn.Linear(30, 1), nn.Sigmoid())

    def forward(self, x_prs):
        x = torch.cat((x_prs[0], x_prs[1]), dim=1)
        out = self.fullyConnectedLayer(x)
        return out


def prediction_model_training(save_prediction_model_path, pre_model, training_x, training_y, training_prs, evaluation_x,
                              evaluation_y, evaluation_prs, learning_rate=0.001, epochs=40, batch_sz=256, device='cpu'):
    training_x = torch.tensor(training_x).T
    training_y = torch.tensor(training_y)
    training_prs = torch.tensor(training_prs)
    evaluation_x = torch.tensor(evaluation_x).T.to(device)
    evaluation_y = torch.tensor(evaluation_y).to(device)
    evaluation_prs = torch.tensor(evaluation_prs).to(device)

    train_dateset = data.TensorDataset(training_x, training_prs, training_y)
    train_iter = data.DataLoader(train_dateset, batch_size=batch_sz, shuffle=True)
    loss_fc = nn.BCELoss()
    optimizer_fc = torch.optim.Adam(pre_model.parameters(), lr=learning_rate, weight_decay=0.01)
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
            train_loss = loss_fc(out, y)
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
    save_prediction_model_path = f'./result/normalDeepModel/{disease_name}/prediction_model.params'
    save_auroc_file_path = f'./result/normalDeepModel/{disease_name}/auto_encoder_result.txt'

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

    # prediction model training
    print('\nNow training prediction Model')
    pre_model = prediction_model(input_dim=70)
    prediction_model_training(save_prediction_model_path=save_prediction_model_path,
                              pre_model=pre_model, training_x=training_data, training_y=training_label,
                              training_prs=training_prs, evaluation_x=evaluation_data, evaluation_y=evaluation_label,
                              evaluation_prs=evaluation_prs)

    # evaluate prediction model
    pre_model = prediction_model(input_dim=70)
    pre_model.load_state_dict(torch.load(save_prediction_model_path))
    auroc = evaluation_prediction_model(pre_model=pre_model, evaluation_x=evaluation_data,
                                        evaluation_y=evaluation_label, evaluation_prs=evaluation_prs)
    f1 = open(save_auroc_file_path, 'w')
    f1.write('AUROC')
    f1.write('\n')
    f1.write(str(auroc))
    f1.write('\n')
    f1.close()