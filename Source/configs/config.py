# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG_preprocessing = {
    "init_path": "../Dataset/Sleep/",
    "final_path": "dataset/dataset.csv"
}

CFG_FIN = {
    "trained": False,
    "data_num": 50000,
    "data_length": 3000,
    "units": [512, 256, 128, 64],
    "input_shape": (3000, 1),
    "drp": 0.0,
    "loss": "mse",
    "batch_size": 8,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "patience": 8,
    "monitor": "val_loss",
    "mode": "min",
    "metrics": ["mean_absolute_percentage_error"],
    "num_epochs": 100,
}

CFG_gen = {
    "data_gen": False,
    "data_path": "dataset/dataset.csv",
}

CFG_stages = {
    "FIN_model": None,
    "data_path": "dataset/dataset.csv",
    "gen_data_path": "dataset/gen_dataset.csv",
    "use_gen_data": True,
    "batch_size": 2,
    "signal_shape": (3000, 1),
    "image_shape": (6, 256, 512, 1),
    "cls_num": 5,
    "width": 512,
    "overlap": 17,
    "lnn_units": [512, 128],
    "kan_units": [32],
    "patience": 5,
    "monitor": "val_loss",
    "mode": "min",
    "loss": "categorical_crossentropy",
    "optimizer": "adamax",
    "learning_rate": 0.001,
    "metrics": ["accuracy"],
    "num_epochs": 100,
}
