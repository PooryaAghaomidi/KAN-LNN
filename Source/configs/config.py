# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG_preprocessing = {
    "init_path": "../Dataset/Sleep/",
    "final_path": "dataset/dataset.csv",
}

CFG_FIN = {
    "trained": True,
    "data_path": "dataset/dataset.csv",
    "units": [512, 256, 128],
    "input_shape": (3000, 1),
    "loss": "mse",
    "batch_size": 128,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "patience": 20,
    "monitor": "val_loss",
    "mode": "min",
    "metrics": ["mean_absolute_percentage_error"],
    "num_epochs": 100,
}

CFG_gen = {
    "train": {
        "data_path": "dataset/dataset.csv",
        "signal_length": 3000,
        "label_length": 5,
        "latent_space": 32,
        "batch_size": 32,
        "num_epochs": 200,
        "learning_rate": 0.001,
        "info_interval": 1,
    },

    "generate": {
        "trained": False,
        "generated": False,
    }
}

CFG_stages = {
    "FIN_model": None, #"checkpoints/fin_20240811_044828",
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
