# -*- coding: utf-8 -*-
"""Train model"""


class TrainFIN:
    def __init__(self,
                 model,
                 callbacks,
                 loss,
                 optimizer,
                 metrics,
                 epoches,
                 batch_size,
                 train_data,
                 val_data,
                 steps_per_epoch,
                 validation_steps):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.epoches = epoches
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.train_data = train_data
        self.val_data = val_data
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def train(self):
        print('\nStart training ...\n')
        history = self.model.fit(self.train_data,
                                 validation_data=self.val_data,
                                 batch_size=self.batch_size,
                                 epochs=self.epoches,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_steps=self.validation_steps)

        return history.history['loss'], history.history['val_loss']
