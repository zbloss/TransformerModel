import tensorflow as tf
import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, tokenizer_feat, tokenizer_tar, max_length, mskr, transformer):
        self.tokenizer_feat = tokenizer_feat
        self.tokenizer_tar = tokenizer_tar
        self.max_length = max_length
        self.mskr = mskr
        self.transformer = transformer

    def evaluate(self, inp_sentence):
        """
        This allows the ability to score the models effectiveness at predicting each output sequence
        :return: The model output as well as the attention weights from the calculation
        """

        start_token = [self.tokenizer_feat.vocab_size]
        end_token = [self.tokenizer_feat.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer_feat.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_tar.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.mskr.create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.tokenizer_tar.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # concatenate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(self, attention, sentence, result, layer):
        """
        A handy function to plot the attention weights.
        """

        fig = plt.figure(figsize=(16, 8))

        sentence = self.tokenizer_feat.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ['<start>'] + [self.tokenizer_feat.decode([i]) for i in sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer_tar.decode([i]) for i in result
                                if i < self.tokenizer_tar.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()

    def translate(self, sentence, plot=''):
        """
        :param sentence: input sentence you wish to receive a response to
        :param plot: Whether you want to plot the attention weighting
        :return: The response given the input text.
        """

        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.tokenizer_tar.decode([i for i in result
                                                        if i < self.tokenizer_tar.vocab_size])

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)
