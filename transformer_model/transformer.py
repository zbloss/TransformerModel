import tensorflow as tf
from transformer_model.encoder import Encoder
from transformer_model.decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
        """
        :param num_layers: Number of EncoderLayers and DecoderLayers to use
        :param d_model: Dimensionality of the EncoderLayer and DecoderLayer
        :param num_heads: Number of heads to use in the Multiheaded Attention
        :param dff: Input nodes in the Point-wise Feed Forward NN
        :param input_vocab_size: Total number of unique words in the feature set.
        :param target_vocab_size: Total number of unique words in the target set.
        :param rate: Dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    