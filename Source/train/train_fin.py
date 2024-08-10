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
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 steps_per_epoch,
                 validation_steps):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.epoches = epoches
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def train(self):
        print('\nStart training ...\n')
        history = self.model.fit(self.X_train,
                                 self.y_train,
                                 validation_data=(self.X_test, self.y_test),
                                 batch_size=self.batch_size,
                                 epochs=self.epoches,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_steps=self.validation_steps)

        return history.history['loss'], history.history['val_loss']
