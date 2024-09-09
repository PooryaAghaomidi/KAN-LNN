# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG_preprocessing = {
    "init_path": "../Dataset/Sleep/",
    "final_path": "dataset/dataset.csv",
}

CFG_FIN = {
    "trained": True,
    "data_num": 20000,
    "input_shape": (3000, 1),
    "conv_units": [32, 64, 128],
    "fc_units": [64, 32],
    "lnn_units": [128, 64],
    "loss": "mse",
    "batch_size": 8,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "patience": 30,
    "monitor": "val_loss",
    "mode": "min",
    "metrics": [],
    "num_epochs": 100,
}

CFG_gen = {
    "all_generated": True,
    "data_path": "dataset/dataset.csv",
    "SMOTE": True,
    "SMOTE_saved": "dataset/smote_dataset.csv",
    "ADASYN": True,
    "ADASYN_saved": "dataset/adasyn_dataset.csv",
    "N1": False,
    "N1_saved": "dataset/n1_dataset.csv",
}

CFG_N1 = {
    "trained": False,
    "data_path": "dataset/n1_dataset.csv"
}

CFG_stages = {
    "FIN_model": "checkpoints/fin_20240811_044828",
    "data_path": "dataset/dataset.csv",
    "smote_data_path": "dataset/smote_dataset.csv",
    "adasyn_data_path": "dataset/adasyn_dataset.csv",
    "data_type": "smote",
    "batch_size": 4,
    "signal_shape": (3000, 1),
    "image_shape": (6, 256, 512, 1),
    "cls_num": 5,
    "overlap": 17,
    "lnn_units": [256, 128],
    "kan_units": [32],
    "patience": 5,
    "monitor": "val_loss",
    "mode": "min",
    "loss": "categorical_crossentropy",
    "optimizer": "sgd",
    "learning_rate": 0.01,
    "metrics": ["accuracy"],
    "num_epochs": 100,
}
