# -*- coding: utf-8 -*-
"""Callbacks"""

from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from livelossplot.inputs.tf_keras import PlotLossesCallback


def callback(train_type, patience=5, mymonitor='val_loss', mymode='min', min_delta=0.5):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    main_chk = ModelCheckpoint(filepath=f'checkpoints/{train_type}_{timestamp}', monitor=mymonitor, mode=mymode,
                               verbose=1, save_best_only=True)
    early_st = EarlyStopping(monitor=mymonitor, mode=mymode, min_delta=min_delta, patience=patience, verbose=1)
    rduce_lr = ReduceLROnPlateau(monitor=mymonitor, mode=mymode, factor=0.5, patience=patience-15, verbose=1,
                                 min_lr=0.0001)
    tsboard = TensorBoard(log_dir=f'checkpoints/{train_type}_{timestamp}')

    tr_plot = PlotLossesCallback()

    return [main_chk, early_st, rduce_lr, tr_plot, tsboard], f'checkpoints/{train_type}_{timestamp}'
