# -*- coding: utf-8 -*-
""" main.py """

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import pandas as pd
import matplotlib.pyplot as plt
from dataset.oversampling import smote_oversampling, adasyn_oversampling
from utils.set_seed import set_seed
from utils.set_device import set_gpu
from dataset.preprocessing import preprocess
from model.fin import build_fin
from dataset.sin_data import generate_data
from model.model import build
from utils.callbacks import callback
from loss.huber import huber_loss
from loss.mean_squared_error import mse_loss
from loss.categorical_crossentropy import cc_loss
from optimizer.adamax import adamax_opt
from optimizer.adam import adam_opt
from optimizer.sgd import sgd_opt
from train.train_fin import TrainFIN
from train.train import TrainModel
from test.test_fin import test_fin
from test.test_model import test_model
from dataloader.dataloader import DataGenerator
from dataloader.findataloader import FINDataGenerator
from configs.config import CFG_preprocessing, CFG_FIN, CFG_gen


def run_preprocessing(configs):
    preprocess(configs['init_path'], configs['final_path'], length=7500)


def run_FIN(configs):
    data_train, skew_train, kurt_train = generate_data(configs['data_num'], configs['input_shape'][0])
    data_test, skew_test, kurt_test = generate_data(configs['data_num'] // 10, configs['input_shape'][0])
    data_val, skew_val, kurt_val = generate_data(configs['data_num'] // 10, configs['input_shape'][0])

    plt.hist(skew_train)
    plt.show()

    plt.hist(kurt_train)
    plt.show()

    steps_per_epoch = len(data_train) // configs['batch_size']
    steps_per_val = len(data_val) // configs['batch_size']

    model = build_fin(configs["input_shape"], configs["conv_units"], configs["lnn_units"], configs["fc_units"])

    callbacks, model_name = callback('fin', configs["patience"], configs['monitor'], configs['mode'])

    if configs['loss'] == 'mse':
        my_loss = mse_loss()
    elif configs['loss'] == 'mae':
        my_loss = mse_loss()
    elif configs['loss'] == 'huber':
        my_loss = huber_loss()
    else:
        raise ValueError("The loss is invalid")

    if configs['optimizer'] == 'adamax':
        my_opt = adamax_opt(configs['learning_rate'], clipvalue=0.5)
    elif configs['optimizer'] == 'adam':
        my_opt = adam_opt(configs['learning_rate'])
    elif configs['optimizer'] == 'sgd':
        my_opt = sgd_opt(configs['learning_rate'])
    else:
        raise ValueError("The optimizer is invalid")

    train_class = TrainFIN(model, callbacks, my_loss, my_opt, configs['metrics'], configs['num_epochs'],
                           configs['batch_size'], data_train, skew_train, kurt_train, data_val, skew_val, kurt_val,
                           steps_per_epoch, steps_per_val)

    train_class.train()
    test_fin(model_name, data_test, skew_test, kurt_test)


def run_gen(configs):
    if not configs['SMOTE']:
        smote_oversampling(configs['data_path'], configs['SMOTE_saved'])
    if not configs['ADASYN']:
        adasyn_oversampling(configs['data_path'], configs['ADASYN_saved'])
    else:
        pass


def run_stages(configs):
    gc.collect()

    if configs['data_type'] == 'normal':
        data = pd.read_csv(configs['data_path'])
        data = data.drop(['Unnamed: 0'], axis=1)
    elif configs['data_type'] == 'smote':
        data = pd.read_csv(configs['smote_data_path'])
    elif configs['data_type'] == 'adasyn':
        data = pd.read_csv(configs['adasyn_data_path'])
    else:
        raise ValueError("Invalid data type!")

    data = data.sample(frac=1).reset_index(drop=True)

    ln = len(data)
    train = data.iloc[:int(ln * 0.8), :]
    test = data.iloc[int(ln * 0.8):int(ln * 0.9), :]
    val = data.iloc[int(ln * 0.9):, :]

    del data

    steps_per_epoch = len(train) // configs['batch_size']
    steps_per_test = len(test) // configs['batch_size']
    steps_per_val = len(val) // configs['batch_size']

    train_gen = DataGenerator(train, configs['image_shape'], configs['signal_shape'], configs['batch_size'],
                              configs['cls_num'], configs['overlap'])
    test_gen = DataGenerator(test, configs['image_shape'], configs['signal_shape'], configs['batch_size'],
                             configs['cls_num'], configs['overlap'])
    val_gen = DataGenerator(val, configs['image_shape'], configs['signal_shape'], configs['batch_size'],
                            configs['cls_num'], configs['overlap'])

    model = build(configs['image_shape'], configs['signal_shape'], configs['cls_num'], configs['lnn_units'],
                  configs['FIN_model'], configs['kan_units'])
    callbacks, model_name = callback('stages', configs["patience"], configs['monitor'], configs['mode'])

    if configs['loss'] == 'categorical_crossentropy':
        my_loss = cc_loss(from_logits=False, label_smoothing=0.0)
    else:
        raise ValueError("The loss is invalid")

    if configs['optimizer'] == 'adamax':
        my_opt = adamax_opt(configs['learning_rate'], clipvalue=None)
    elif configs['optimizer'] == 'adam':
        my_opt = adam_opt(configs['learning_rate'])
    elif configs['optimizer'] == 'sgd':
        my_opt = sgd_opt(configs['learning_rate'])
    else:
        raise ValueError("The optimizer is invalid")

    train_class = TrainModel(model, callbacks, my_loss, my_opt, configs['metrics'], configs['num_epochs'],
                             configs['batch_size'], train_gen, val_gen, steps_per_epoch, steps_per_val)

    train_class.train()
    test_model(model_name, test_gen, steps_per_test)


if __name__ == '__main__':
    print('\n==================================== CONFIGURATIONS ========================================\n')
    set_seed()
    set_gpu()

    print('\n===================================== PREPROCESSING ========================================\n')
    head_tail = os.path.split(CFG_preprocessing['final_path'])
    if head_tail[1] not in os.listdir(head_tail[0]):
        run_preprocessing(CFG_preprocessing)
    else:
        pass

    print("PREPROCESSING: Done")

    print('\n======================================= TRAIN FIN ==========================================\n')
    if not CFG_FIN["trained"]:
        run_FIN(CFG_FIN)
    else:
        pass

    print("TRAIN FIN: Done")

    print('\n==================================== GENERATE DATA =========================================\n')
    if not CFG_gen["all_generated"]:
        run_gen(CFG_gen)
    else:
        pass

    print("GENERATE DATA: Done")

    # print('\n===================================== TRAIN STAGES =========================================\n')
    # if CFG_stages["FIN_model"] is not None:
    #     run_stages(CFG_stages)
    # else:
    #     pass
    #
    # print("TRAIN STAGES: Done")
