import time
from masker.masker import Masker
import tensorflow as tf


class Trainer(object):
    def __init__(self, transformer, learning_rate, optimizer, epochs, train_dataset, test_dataset,
                 load_ckpt=True,
                 loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none'),
                 train_loss=tf.keras.metrics.Mean(name='train_loss'),
                 train_accuracy=tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy'),
                 checkpoint_path='./models/checkpoints/',
                 max_to_keep=5
                 ):
        """
        :param transformer: Instance of Transformer Class
        :param learning_rate: Learning Rate schedule
        :param optimizer: TF optimizer object
        :param epochs: Number of Epochs to train on
        :param train_dataset: training dataset
        :param test_dataset: testing dataset
        :param load_ckpt: bool to load recent checkpoint if True
        :param loss_object: loss calculation object
        :param train_loss: training loss object
        :param train_accuracy: training accuracy object
        :param checkpoint_path: the location of where the checkpoints are stored
        :param max_to_keep: maximum number of checkpoints to keep at a time
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.loss_object = loss_object
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.checkpoint_path = checkpoint_path
        self.mskr = Masker()

        self.transformer = transformer
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.epochs = epochs
        self.ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        self.max_to_keep = max_to_keep
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, self.max_to_keep)

        if load_ckpt:
            print('Attempting to load latest checkpoint...')
            if self.ckpt_manager.latest_checkpoint:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('Latest Checkpoint Restored!!!')
            else:
                print(f'No checkpoint found...')
                print('Training from scratch')
                pass
        else:
            pass

    def loss_function(self, real, pred):
        """
        :param real: the actual output
        :param pred: the model output
        :return: the training loss
        """

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.mskr.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.mskr.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.mskr.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.mskr.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ])
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.mskr.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp,
                                              True,
                                              enc_padding_mask,
                                              combined_mask,
                                              dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def train(self):
        loss_hist = {}
        acc_hist = {}
        for epoch in range(self.epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # inp -> client, tar -> qltm
            for (batch, (inp, tar)) in enumerate(self.train_dataset):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                self.train_loss.result(),
                                                                self.train_accuracy.result()))
            loss_hist[epoch] = self.train_loss.result()
            acc_hist[epoch] = self.train_accuracy.result()

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        return loss_hist, acc_hist
