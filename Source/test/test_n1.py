# -*- coding: utf-8 -*-
""" Test the model """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model

weight_zero = 2.0
weight_one = 1.0


def loss(y_true, y_pred):
    # Compute the standard binary cross-entropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Apply the class weights
    weights = tf.where(tf.equal(y_true, 1), weight_one, weight_zero)
    weighted_bce = bce * weights

    # Return the mean weighted loss
    return tf.reduce_mean(weighted_bce)


def test_n1(model_path, test_gen, steps_per_test):
    print('\nGetting test results ...\n')

    # Load the model with the custom loss
    testmodel = load_model(model_path, custom_objects={'loss': loss})

    # Evaluate the model on the test data
    tst_loss, tst_acc = testmodel.evaluate(test_gen, steps=steps_per_test, verbose=1)
    print(f"\nTest Loss: {tst_loss}, Test Accuracy: {tst_acc}")

    # Initialize arrays for true and predicted labels
    y_true = []
    y_pred = []

    # Loop through the test generator to get predictions and true labels
    for i in range(steps_per_test):
        x_batch, y_batch = test_gen[i]
        preds = testmodel.predict(x_batch)

        # preds = np.round(preds).flatten()
        preds = np.argmax(preds, axis=1)
        y_batch = np.argmax(y_batch, axis=1)

        y_pred.extend(preds)
        y_true.extend(y_batch)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Evaluate the test model for loss and accuracy
    tst_loss, tst_acc = testmodel.evaluate(test_gen, steps=steps_per_test, verbose=1)

    print(f"\nTest Loss: {tst_loss}")
    print(f"Test Accuracy: {tst_acc}")
