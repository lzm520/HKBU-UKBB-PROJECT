import time

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PRS_models import *
from sklearn.model_selection import train_test_split


def model_train(data, label, PRS, numerical_phenotype_idx, categorical_phenotype_idx, val_data, val_label, val_PRS,
                num_phenotype, param_save, train_loss_save, val_loss_save, alpha=1, beta=1, batch_size=100,
                n_epoch=1000):
    # model instance
    n_categorical_phenotype_feature = [idx_e - idx_s for (idx_s, idx_e) in categorical_phenotype_idx]
    pred = myModel(num_phenotype, n_categorical_phenotype_feature)
    optimizer_pred = tf.optimizers.Adadelta(rho=0.95, clipvalue=0.001)

    prs = PRS_model()
    optimizer_prs = tf.optimizers.Adadelta(rho=0.95, clipvalue=0.001)

    loss_fc = tf.losses.KLDivergence()

    db = tf.data.Dataset.from_tensor_slices((data, label, PRS)).shuffle(data.shape[0]).batch(batch_size)
    val_acc_best = 0.0
    val_loss_epoch = []
    train_loss_epoch = []

    start_time = time.time()

    # training k times epoch
    for epoch in range(n_epoch):
        loss_batch = []
        for step, (x, y, prs) in enumerate(db):
            numerical_x = [x[:, idx] for idx in numerical_phenotype_idx]
            categorical_x = [x[:, idx_s:idx_e] for (idx_s, idx_e) in categorical_phenotype_idx]
            y_true = tf.one_hot(indices=y, depth=2, dtype=tf.float64)
            with tf.GradientTape(persistent=True) as tape:
                # result
                result_pred = tf.transpose(pred(numerical_x, categorical_x))
                result_prs = prs(prs)

                # loss calculation
                kl = loss_fc(result_pred[: 1], result_prs)
                loss_f = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, result_pred))
                loss_prs = -tf.reduce_mean(y * tf.math.log(result_prs) + (1 - y) * tf.math.log(1 - result_prs))
                loss = loss_prs + alpha * kl + beta * loss_f

            # parameters update
            grads_pred = tape.gradient(loss, pred.trainable_variables)
            grads_prs = tape.gradient(loss, prs.trainable_variables)
            del tape
            optimizer_pred.apply_gradients(zip(grads_pred, pred.trainable_variables))
            optimizer_prs.apply_gradients((zip(grads_prs, prs.trainable_variables)))
            # loss information every batch
            loss_batch.append(loss.numpy())

    # training loss information every epoch
    train_loss_epoch.append(np.mean(loss_batch))

    if np.mod(epoch, 50) == 0:
        count_time = time.time() - start_time
        print(f'run epoch: {epoch} , time consumed: {int(count_time)} s')


def validation_acc(pred_model, prs_model, val_data, val_label, val_PRS, numerical_phenotype_idx,
                   categorical_phenotype_idx):
    numerical_x = [val_data[:, idx] for idx in numerical_phenotype_idx]
    categorical_x = [val_data[:, idx_s:idx_e] for (idx_s, idx_e) in categorical_phenotype_idx]
    y_true = tf.one_hot(indices=val_label, depth=2, dtype=tf.float64)
    val_result_pre = tf.transpose(pred_model(numerical_x, categorical_x))
    val_result_prs = prs_model(val_PRS)
    PRSPR_Pred = val_result_pre[: 1] + val_result_prs
    pred_result = PRSPR_Pred.numpy()
    threshold = np.mean(pred_result)
    classification = []
    for i in range(len(val_label)):
        if pred_result[i] >= threshold:
            classification.append(1)
        else:
            classification.append(0)
    val_acc = np.sum(np.array(classification) == val_label)
    final_val_acc = val_acc / len(val_label)

    return final_val_acc


if __name__ == '__main__':
    seed = 9999
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # PRS score loading

    # phenotype data loading
    d_file = '../../data/features_selection/features_selection_data_dummy_data.txt'
    d_info_file = '../../data/features_selection/features_selection_data_dummy_data.txt'
    data = np.genfromtxt(d_file)
    data = data.T
    data_info = pd.read_csv(d_info_file, header=None, names=['Type', 'id', 'Description'])
    # categorical phenotype data preprocessing
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
    # Label loading

    # Train & Test data splitting
    x_train, x_test = train_test_split(data, test_size=0.25, random_state=seed, shuffle=True)
