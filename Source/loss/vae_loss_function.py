from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class VAELossLayer(Layer):
    def __init__(self, signal_length, **kwargs):
        self.signal_length = signal_length
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_true, y_pred, mu, l_sigma = inputs
        # Reconstruction loss
        xent_loss = K.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        xent_loss = K.sum(xent_loss, axis=-1) * self.signal_length
        # KL divergence loss
        kl_loss = -0.5 * K.sum(1 + l_sigma - K.square(mu) - K.exp(l_sigma), axis=-1)
        # Total loss
        return K.mean(xent_loss + kl_loss)
