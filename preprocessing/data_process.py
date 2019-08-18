import tensorflow_datasets as tfds
import tensorflow as tf
import pyodbc
import pandas as pd


class DataProcessor(object):

    def __init__(self, test_size=0.1, csv_path=None, sql_source='',
                 max_length=40, feature_col='', target_col='', buffer_size=20000, batch_size=64,
                 server=''):
        """
        :param test_size: The % of data to withhold for the test set
        :param csv_path: The path of the DataFrame to use. If None, queries the sql_source
        :param sql_source: The table to pull the DataFrame from
        :param feature_col: The column name of your feature input. (Defaults to the first column)
        :param target_col:  The column name of your target input. (Defaults ot the second column)
        :param max_length: Maximum number of words per input you want to use for the training set.
        """
        self.test_size = test_size
        self.csv_path = csv_path
        self.sql_source = sql_source
        self.feature_col = feature_col
        self.target_col = target_col
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.server = server
        self.SQL_SERVER = 'SQL Server'
        self.cnxn = pyodbc.connect(f'DRIVER={self.SQL_SERVER}; SERVER={self.server}; Trusted_Connection=yes')

    def load_data(self):
        """
        :return: Loads and returns the data in a DataFrame.
        """
        if self.csv_path is None:
            df = pd.read_sql(f'SELECT * FROM {self.sql_source}', con=self.cnxn)
            # inst = s.sqlConnect()
            # df = inst.script_to_df(f'SELECT * FROM {self.sql_source}')
            df.dropna(inplace=True)
        else:
            df = pd.read_csv(self.csv_path)

        return df

    def train_test_split(self, df):
        """
        :param df: The DataFrame that you will split into train and test sets.
        :return: Train and Tests DataFrames split on the test_size percentage.
        """
        test_rows = int(len(df) * self.test_size)
        test = df.sample(test_rows)
        train = df[~df.isin(test)]

        train.dropna(inplace=True)
        test.dropna(inplace=True)

        return train, test

    def tokenizer(self, train, vocab_size=2 ** 13):
        """
        :param train: Dataframe containing the training data set.
        :param vocab_size: Max number of words to tokenize
        :return: the Feature tokenizer and Target tokenizer
        """
        if self.feature_col is None:
            feature_col = train.columns[0]
        else:
            feature_col = self.feature_col

        if self.target_col is None:
            target_col = train.columns[1]
        else:
            target_col = self.target_col

        global tokenizer_feat
        global tokenizer_tar

        tokenizer_tar = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (line for line in train[target_col].values), target_vocab_size=vocab_size
        )

        tokenizer_feat = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (line for line in train[feature_col].values), target_vocab_size=vocab_size
        )

        return tokenizer_feat, tokenizer_tar

    @staticmethod
    def encode(lang1, lang2):
        """
        :param lang1: A Tensor containing the tokenized client text sentence
        :param lang2: A Tensor containing the tokenized QL TM text sentence
        :return: The originally passed tensors with a Start-of-Sequence (SOS) and
                 a End-of-Sequence (EOS) added.
        """

        lang1 = [tokenizer_feat.vocab_size] + tokenizer_feat.encode(
            lang1.numpy()) + [tokenizer_feat.vocab_size + 1]

        lang2 = [tokenizer_tar.vocab_size] + tokenizer_tar.encode(
            lang2.numpy()) + [tokenizer_tar.vocab_size + 1]

        return lang1, lang2

    def filter_max_length(self, x, y):
        """
        :param x: The Feature tensor
        :param y: The Target tensor
        :return: The pair of Tensors whose lengths are less than or equal to self.max_length
        """

        return tf.logical_and(tf.size(x) <= self.max_length,
                              tf.size(y) <= self.max_length)

    def tf_encode(self, feature, target):
        """
        :param feature: The input feature text
        :param target:  The output target text
        :return: The encoded vector representations of the feature and target text.
        """

        return tf.py_function(self.encode, [feature, target], [tf.int64, tf.int64])

    def to_tensor_dataset(self, data):
        """
        :param data: a DataFrame column that contains the feature_col and target_col
        :return: A TensorDataset made from the data DataFrame.
        """

        return tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(data[self.feature_col].values, tf.string),
                tf.cast(data[self.target_col].values, tf.string)
            )
        )

    def preprocess(self, train: pd.DataFrame, test: pd.DataFrame):
        """

        :param train: Your training set.
        :param test: Your testing set.
        :return: filtered and batched TensorDatasets for train and test.
        """

        train_data = self.to_tensor_dataset(train)
        test_data = self.to_tensor_dataset(test)

        train_dataset = train_data.map(self.tf_encode)
        train_dataset = train_dataset.filter(self.filter_max_length)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(self.buffer_size).padded_batch(self.batch_size,
                                                                             padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = test_data.map(self.tf_encode)
        test_dataset = test_dataset.filter(self.filter_max_length).padded_batch(self.batch_size,
                                                                                padded_shapes=([-1], [-1]))

        return train_dataset, test_dataset
