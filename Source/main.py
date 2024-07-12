# -*- coding: utf-8 -*-
""" main.py """

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from configs.config import CFG
from utils.set_seed import set_seed
from utils.set_device import set_gpu
from model.model import build
from utils.callbacks import callback
from train.train import TrainModel
from dataset.preprocessing import preprocess
from dataloader.dataloader import DataGenerator

mode = 'train'


def run(configs, mode):
    set_seed()
    set_gpu()

    head_tail = os.path.split(configs['final_path'])
    if head_tail[1] not in os.listdir(head_tail[0]):
        preprocess(configs['init_path'], configs['final_path'])
    print("The dataset has been processed!")

    data = np.load(configs['final_path'])
    ln = len(data)
    train = data[:int(ln * 0.8), :]
    test = data[int(ln * 0.8):int(ln * 0.9), :]
    val = data[int(ln * 0.9):, :]

    steps_per_epoch = len(train) // configs['batch_size']
    steps_per_test = len(test) // configs['batch_size']
    steps_per_val = len(val) // configs['batch_size']

    train_gen = DataGenerator(train, configs['shape'], configs['batch_size'], configs['cls_num'])
    test_gen = DataGenerator(train, configs['shape'], configs['batch_size'], configs['cls_num'])
    val_gen = DataGenerator(train, configs['shape'], configs['batch_size'], configs['cls_num'])

    model = build(configs['shape'], configs['cls_num'])
    callbacks, model_name = callback(configs['monitor'], configs['mode'])

    if configs['loss'] == 'categorical_crossentropy':
        from loss.categorical_crossentropy import cc_loss
        my_loss = cc_loss(from_logits=False, label_smoothing=configs['label_smoothing'])
    else:
        raise ValueError("The loss is invalid")

    if configs['optimizer'] == 'adam':
        from optimizer.adam import adam_opt
        my_opt = adam_opt(configs['learning_rate'], epsilon=1e-7, clipvalue=None)
    else:
        raise ValueError("The optimizer is invalid")

    train_class = TrainModel(model, callbacks, my_loss, my_opt, configs['metrics'], configs['num_epochs'],
                             configs['batch_size'], train_gen, val_gen, steps_per_epoch, steps_per_val)

    train_class.train()


if __name__ == '__main__':
    run(CFG, mode)
