# -*- coding: utf-8 -*-
"""Device function with MirroredStrategy"""
import os
import warnings
import tensorflow as tf


def set_gpu():
    gpu_list = tf.config.list_physical_devices('GPU')
    print('The version of tensorflow is: \n', tf.__version__)
    print('List of GPU devices: \n', gpu_list)

    if gpu_list:
        # Enable memory growth for each GPU
        try:
            for gpu in gpu_list:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Create a MirroredStrategy for multi-GPU training
            strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            print(f"Number of devices: {strategy.num_replicas_in_sync}")

        except RuntimeError as e:
            print(e)
    else:
        warnings.warn("GPUs have not been recognized!")

    return strategy
