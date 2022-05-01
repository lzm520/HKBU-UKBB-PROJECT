import os
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PRS_models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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

        # validation loss information every epoch
        val_acc = validation_acc(pred, prs, val_data, val_label, val_PRS, numerical_phenotype_idx, categorical_phenotype_idx)

        if val_acc > val_acc_best:
            val_acc_best = val_acc
            pred.save_weights(param_save + '/pred_model_param_g_t2d_' + str(alpha) + '_' + str(beta) +'ckpt')
            prs.save_weights(param_save + '/prs_model_param_g_t2d_' + str(alpha) + '_' + str(beta) +'ckpt')

    # for epoch error plot
    np.save(train_loss_save + '/train_loss_t2d_' + str(alpha) + '_' + str(beta), train_loss_epoch)
    np.save(val_loss_save + '/val_loss_t2d_' + str(alpha) + '_' + str(beta), train_loss_epoch)


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


def evaluation_auc(pred_model, prs_model, data, label, PRS, param_path, numerical_phenotype_idx,
                   categorical_phenotype_idx, alpha_test, beta_test, save_result=False, result_folder=None):
    numerical_x = [data[:, idx] for idx in numerical_phenotype_idx]
    categorical_x = [label[:, idx_s:idx_e] for (idx_s, idx_e) in categorical_phenotype_idx]
    y_true = tf.one_hot(indices=label, depth=2, dtype=tf.float64)

    pred_model_trained_param = param_path + '/pred_model_param_g_t2d_' + str(alpha_test) + '_' + str(beta_test) +'ckpt'
    prs_model_trained_param = param_path + '/prs_model_param_g_t2d_' + str(alpha_test) + '_' + str(beta_test) + 'ckpt'

    pred_model.load_weights(pred_model_trained_param)
    prs_model.load_weights(prs_model_trained_param)

    result_pre = tf.transpose(pred_model(numerical_x, categorical_x))
    result_prs = prs_model(PRS)
    PRSPR_Pred = result_pre[: 1] + result_prs
    result_list = PRSPR_Pred.numpy()
    label_list = label.numpy()

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


if __name__ == '__main__':
    work_path = os.getcwd()
    seed = 9999
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # PRS score loading

    # phenotype data loading
    d_file = '../../data/features_selection/features_selection_data_dummy_data.txt'
    d_info_file = '../../data/features_selection/features_selection_data_dummy_info.csv'
    data = np.genfromtxt(d_file)
    data = data.T
    data_info = pd.read_csv(d_info_file, header=None, names=['Type', 'id', 'Description'])
    num_phenotype = len(set([i[0] for i in np.char.split(data_info['id'].to_numpy(dtype=str), '_')]))
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
    n_categorical_phenotype_feature = [idx_e - idx_s for (idx_s, idx_e) in categorical_phenotype_idx]
    # Label loading

    # Train & Test data splitting
    x_train, x_test = train_test_split(data, test_size=0.25, random_state=seed, shuffle=True)
    print('Data loading succeed')

    mode = input('train or evaluation mode?\t')

    # result folder
    result = work_path +'/result'
    if result is not None:
        if not os.path.isdir(result):
            os.mkdir(result)
    result_folder = work_path + '/result/t2d_200epoch_monitor_acc'
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

    if mode == ' train':
        auc_txt = open(result_folder + '/auc_collected.txt', 'w')
        auc_txt.write('alpha\tbeta\tAUROC\n')
        for i in range(len(alpha_candi)):
            for j in range(len(beta_candi)):
                # train model
                model_train(numerical_phenotype_idx=numerical_phenotype_idx,
                            categorical_phenotype_idx=categorical_phenotype_idx, num_phenotype=num_phenotype,
                            param_save=param_path, train_loss_save=train_loss_path, val_loss_save=val_loss_path,
                            alpha=alpha_candi[i], beta=beta_candi[j], n_epoch=200)
                # test after training
                eval_pred_model = myModel(num_phenotype, n_categorical_phenotype_feature)
                eval_prs_model = PRS_model()
                auc_test = evaluation_auc(pred_model=eval_pred_model, prs_model=eval_prs_model,
                                          param_path=param_path, numerical_phenotype_idx=numerical_phenotype_idx,
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
            eval_pred_model = myModel(num_phenotype, n_categorical_phenotype_feature)
            eval_prs_model = PRS_model()
            auc_test = evaluation_auc(pred_model=eval_pred_model, prs_model=eval_prs_model,
                                      param_path=param_path, numerical_phenotype_idx=numerical_phenotype_idx,
                                      categorical_phenotype_idx=categorical_phenotype_idx, alpha_test=alpha_input,
                                      beta_test=beta_input, save_result=sava_evaluation, result_folder=result_folder)
            print('test auc:', auc_test)
        else:
            print('alpha/beta input error')
    else:
        print('Input error. Please restart')