# -*- coding: utf-8 -*-
""" main.py """

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.set_seed import set_seed
from utils.set_device import set_gpu
from dataset.preprocessing import preprocess
from dataset.sin_data import generate_data
from model.fin import build_fin
from model.model import build
from utils.callbacks import callback
from loss.mean_squared_error import mse_loss
from loss.categorical_crossentropy import cc_loss
from optimizer.adamax import adamax_opt
from optimizer.adam import adam_opt
from train.train_fin import TrainFIN
from train.train import TrainModel
from test.test_fin import test_fin
from test.test_model import test_model
from dataloader.dataloader import DataGenerator
from utils.kurtosis import get_kurtosis, get_all_kurtosis
from configs.config import CFG_preprocessing, CFG_FIN, CFG_stages, CFG_gen


def run_preprocessing(configs):
    preprocess(configs['init_path'], configs['final_path'], length=7500)


def run_FIN(configs):
    dataset = generate_data(configs["data_num"], configs["data_length"])
    kurtosis = get_all_kurtosis(dataset)

    plt.figure(figsize=(12, 4))
    plt.hist(np.array(kurtosis))
    plt.grid()
    plt.show()

    X_train = dataset.sample(frac=0.85, random_state=1)
    X_test = dataset.drop(X_train.index)
    y_train = np.array(kurtosis)[X_train.index]
    y_test = np.array(kurtosis)[X_test.index]

    model = build_fin(configs["input_shape"], configs["units"], configs["drp"])

    steps_per_epoch = len(y_train) // configs['batch_size']
    steps_per_test = len(y_test) // configs['batch_size']

    callbacks, model_name = callback('fin', configs["patience"], configs['monitor'], configs['mode'])

    if configs['loss'] == 'mse':
        my_loss = mse_loss()
    else:
        raise ValueError("The loss is invalid")

    if configs['optimizer'] == 'adamax':
        my_opt = adamax_opt(configs['learning_rate'], clipvalue=None)
    elif configs['optimizer'] == 'adam':
        my_opt = adam_opt(configs['learning_rate'])
    else:
        raise ValueError("The optimizer is invalid")

    train_class = TrainFIN(model, callbacks, my_loss, my_opt, configs['metrics'], configs['num_epochs'],
                           configs['batch_size'], X_train, y_train, X_test, y_test, steps_per_epoch, steps_per_test)

    train_class.train()
    test_fin(model_name, X_test, y_test)


def run_gen(configs):
    data = pd.read_csv(configs['data_path'])
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['Unnamed: 0'], axis=1)

    ln = len(data)
    train = data.iloc[:int(ln * 0.8), :]
    test = data.iloc[int(ln * 0.8):int(ln * 0.9), :]
    val = data.iloc[int(ln * 0.9):, :]

    steps_per_epoch = len(train) // configs['batch_size']
    steps_per_test = len(test) // configs['batch_size']
    steps_per_val = len(val) // configs['batch_size']


def run_stages(configs):
    if configs['use_gen_data']:
        data = pd.read_csv(configs['gen_data_path'])
    else:
        data = pd.read_csv(configs['data_path'])

    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['Unnamed: 0'], axis=1)

    ln = len(data)
    train = data.iloc[:int(ln * 0.8), :]
    test = data.iloc[int(ln * 0.8):int(ln * 0.9), :]
    val = data.iloc[int(ln * 0.9):, :]

    steps_per_epoch = len(train) // configs['batch_size']
    steps_per_test = len(test) // configs['batch_size']
    steps_per_val = len(val) // configs['batch_size']

    train_gen = DataGenerator(train, configs['image_shape'], configs['signal_shape'], configs['batch_size'],
                              configs['cls_num'], configs['width'], configs['overlap'])
    test_gen = DataGenerator(test, configs['image_shape'], configs['signal_shape'], configs['batch_size'],
                             configs['cls_num'], configs['width'], configs['overlap'])
    val_gen = DataGenerator(val, configs['image_shape'], configs['signal_shape'], configs['batch_size'],
                            configs['cls_num'], configs['width'], configs['overlap'])

    model = build(configs['image_shape'], configs['signal_shape'], configs['cls_num'], configs['lnn_units'],
                  configs['FIN_model'], configs['kan_units'])
    callbacks, model_name = callback('stages', configs["patience"], configs['monitor'], configs['mode'])

    if configs['loss'] == 'categorical_crossentropy':
        my_loss = cc_loss(from_logits=False, label_smoothing=0.0)
    else:
        raise ValueError("The loss is invalid")

    if configs['optimizer'] == 'adamax':
        my_opt = adamax_opt(configs['learning_rate'], clipvalue=None)
    else:
        raise ValueError("The optimizer is invalid")

    train_class = TrainModel(model, callbacks, my_loss, my_opt, configs['metrics'], configs['num_epochs'],
                             configs['batch_size'], train_gen, val_gen, steps_per_epoch, steps_per_val)

    train_class.train()
    test_model(model_name, test_gen, steps_per_test)


if __name__ == '__main__':
    set_seed()
    set_gpu()

    print('===================================== PREPROCESSING ========================================')
    head_tail = os.path.split(CFG_preprocessing['final_path'])
    if head_tail[1] not in os.listdir(head_tail[0]):
        run_preprocessing(CFG_preprocessing)
    else:
        pass

    print("PREPROCESSING: Done")

    print('======================================= TRAIN FIN ==========================================')
    if not CFG_FIN["trained"]:
        run_FIN(CFG_FIN)
    else:
        pass

    print("TRAIN FIN: Done")

    print('==================================== DATA GENERATION =======================================')
    if CFG_gen["data_gen"]:
        run_gen(CFG_gen)
    else:
        pass

    print("DATA GENERATION: Done")

    print('===================================== TRAIN STAGES =========================================')
    if CFG_stages["FIN_model"] is not None:
        run_stages(CFG_stages)
    else:
        pass

    print("TRAIN STAGES: Done")
