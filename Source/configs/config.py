# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "init_path": "../Dataset/mitbih_train.csv",
    "final_path": "dataset/Dataset.npy",
    "shape": (128, 128, 2),
    "batch_size": 32,
    "cls_num": 5,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "loss": "categorical_crossentropy",
    "optimizer": 'adam',
    "monitor": "val_loss",
    "mode": "min",
    "label_smoothing": 0.0,
    "metrics": ['accuracy']
}
