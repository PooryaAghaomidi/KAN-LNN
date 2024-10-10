# -*- coding: utf-8 -*-
""" Test the model """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


def test_model(model_path, test_gen, steps_per_test):
    print('\nGetting test results ...\n')

    testmodel = load_model(model_path)
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
    print(classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['W', 'N1', 'N2', 'N3', 'REM'],
                yticklabels=['W', 'N1', 'N2', 'N3', 'REM'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Class-specific accuracy
    print("\nClass-Specific Accuracy:")
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for idx, accuracy in enumerate(class_accuracy):
        print(f"Accuracy for class {['W', 'N1', 'N2', 'N3', 'REM'][idx]}: {accuracy:.4f}")
