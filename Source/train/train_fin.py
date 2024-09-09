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
                 data_train,
                 skew_train,
                 kurt_train,
                 data_val,
                 skew_val,
                 kurt_val,
                 steps_per_epoch,
                 validation_steps):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.epoches = epoches
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.data_train = data_train
        self.skew_train = skew_train
        self.kurt_train = kurt_train
        self.data_val = data_val
        self.skew_val = skew_val
        self.kurt_val = kurt_val
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def train(self):
        print('\nStart training ...\n')
        history = self.model.fit(self.data_train,
                                 [self.skew_train, self.kurt_train],
                                 validation_data=(self.data_val, [self.skew_val, self.kurt_val],),
                                 batch_size=self.batch_size,
                                 epochs=self.epoches,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_steps=self.validation_steps)

        return history.history['loss'], history.history['val_loss']
