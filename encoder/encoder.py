import tensorflow as tf
from encoder.encoder_layer import EncoderLayer
from positional_encoder.positional_encoder import PositionalEncoder


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        """
        :param num_layers: Number of EncoderLayers to use
        :param d_model: Dimensionality of the Encoder
        :param num_heads: Number of heads to use in the Multiheaded Attention
        :param dff: input nodes of the Point-wise Feed Forward NN
        :param input_vocab_size: Total number of unique words in the training set
        :param rate: Dropout rate
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoder(d_model=d_model)
        self.pos_encoding = self.pos_encoder.positional_encoding(input_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

