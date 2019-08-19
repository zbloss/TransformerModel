import numpy as np
import tensorflow as tf


class PositionalEncoder(object):

    def __init__(self, d_model=512):
        """
        This allows to grade positional importance for each word relative to each sequence it belongs to relative to
        the entire corpus.
        :param d_model: Dimensionality of the encoding.
        """
        self.d_model = d_model

    def get_angles(self, position, i, d_model=None):
        if d_model is None:
            d_model = self.d_model
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates

    def positional_encoding(self, position, d_model=None):
        if d_model is None:
            d_model = self.d_model
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
