# -*- coding: utf-8 -*-
""" main.py """

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from utils.torch_callback import t_callback
from train.train_cvae import TrainCVAE
from model.cvae import CVAE
from optimizer.torch_opt import torch_adam
from loss.cvae_loss import CVAELoss
from utils.set_torch import set_torch_device, set_torch_seed
from dataloader.cvae_dataloader import CVAEDataGenerator
from utils.set_seed import set_seed
from utils.set_device import set_gpu
from dataset.preprocessing import preprocess
from model.fin import build_fin
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
from configs.config import CFG_preprocessing, CFG_FIN, CFG_stages, CFG_gen


def run_preprocessing(configs):
    preprocess(configs['init_path'], configs['final_path'], length=7500)


def run_FIN(configs):
    dataset = pd.read_csv(configs['data_path'])
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = dataset.drop(['Unnamed: 0'], axis=1)

    ln = len(dataset)
    train_data = dataset.iloc[:int(ln * 0.8), :]
    test_data = dataset.iloc[int(ln * 0.8):int(ln * 0.9), :]
    val_data = dataset.iloc[int(ln * 0.9):, :]

    steps_per_epoch = len(train_data) // configs['batch_size']
    steps_per_val = len(val_data) // configs['batch_size']

    train_gen = FINDataGenerator(train_data, configs['input_shape'], configs['batch_size'])
    val_gen = FINDataGenerator(val_data, configs['input_shape'], configs['batch_size'])

    model = build_fin(configs["input_shape"], configs["units"])

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
                           configs['batch_size'], train_gen, val_gen, steps_per_epoch, steps_per_val)

    train_class.train()
    test_fin(model_name, test_data)


def run_train_gen(configs):
    device = set_torch_device()
    generator = set_torch_seed(device=device)

    dataset = pd.read_csv(configs['data_path'])
    dataset = dataset.drop(['Unnamed: 0'], axis=1)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    ln = len(dataset)
    train_data = dataset.iloc[:int(ln * 0.8), :]
    test_data = dataset.iloc[int(ln * 0.8):int(ln * 0.9), :]
    val_data = dataset.iloc[int(ln * 0.9):, :]

    steps_per_epoch = len(train_data) // configs['batch_size']
    steps_per_test = len(test_data) // configs['batch_size']
    steps_per_val = len(val_data) // configs['batch_size']

    train_gen = CVAEDataGenerator(train_data, configs['signal_length'], configs['label_length'], configs['batch_size'],
                                  device)
    test_gen = CVAEDataGenerator(test_data, configs['signal_length'], configs['label_length'], configs['batch_size'],
                                 device)
    val_gen = CVAEDataGenerator(val_data, configs['signal_length'], configs['label_length'], configs['batch_size'],
                                device)

    model = CVAE(configs['signal_length'], configs['latent_space'], configs['label_length']).to(device)

    my_loss = CVAELoss()
    my_opt = torch_adam(model, configs['learning_rate'])

    sum_writer, model_name = t_callback()
    train_class = TrainCVAE(model, train_gen, val_gen, my_opt, my_loss, configs['num_epochs'], configs['batch_size'],
                            steps_per_epoch, steps_per_val, sum_writer, model_name,
                            steps_per_epoch // configs['info_interval'])
    train_class.training()


def run_gen(configs):
    print('In progress ...')


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

    print('\n==================================== DATA GENERATION =======================================\n')
    if not CFG_gen["generate"]["trained"]:
        run_train_gen(CFG_gen["train"])
    elif CFG_gen["generate"]["trained"] and not CFG_gen["generate"]["generated"]:
        run_gen((CFG_gen["generate"]))
    else:
        pass

    print("DATA GENERATION: Done")

    print('\n===================================== TRAIN STAGES =========================================\n')
    if CFG_stages["FIN_model"] is not None:
        run_stages(CFG_stages)
    else:
        pass

    print("TRAIN STAGES: Done")
