import tensorflow as tf


def custom_weighted_binary_crossentropy(weight_zero=2.0, weight_one=1.0):
    def loss(y_true, y_pred):
        # Compute the standard binary cross-entropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Apply the class weights
        weights = tf.where(tf.equal(y_true, 1), weight_one, weight_zero)
        weighted_bce = bce * weights

        # Return the mean weighted loss
        return tf.reduce_mean(weighted_bce)

    return loss
