import tensorflow as tf


class Masker(object):

    def __init__(self):
        """
        This class holds a collection of masking functions that are used across the entire package.
        """

    @staticmethod
    def create_padding_mask(seq):
        """
        :param seq: the sequence to mask
        :return: the padding mask in the form of (batch_size, 1, 1, seq_len)
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]

    @staticmethod
    def create_look_ahead_mask(size):
        """
        :param size:
        :return: the mask for hiding unseen values for training purposes in the form of (seq_len, seq_len)
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def create_masks(self, inp, tar):
        """
        :param self:
        :param inp: The feature tensor to mask.
        :param tar: The target tensor to mask
        :return: the Encoder, Combined, and Decoder masks
        """
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask
