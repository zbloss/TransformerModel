import tensorflow as tf
from datetime import datetime
from training.training import Trainer
from transformer.transformer import Transformer
from preprocessing.data_process import DataProcessor
from lr_scheduler.custom_scheduler import CustomSchedule


# Data Processing
print('\n\nLoading data...')
data_processor = DataProcessor()
df = data_processor.load_data()
train, test = data_processor.train_test_split(df)
print('Generating Tokenizers...')
tokenizer_feat, tokenizer_tar = data_processor.tokenizer(train)
train_dataset, test_dataset = data_processor.preprocess(train, test)

# HPARAMS
EPOCHS = 100
num_layers = 6
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = tokenizer_feat.vocab_size + 2
target_vocab_size = tokenizer_tar.vocab_size + 2
dropout_rate = 0.1

# Custom Scheduler
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Transformer
transformer = Transformer(d_model=d_model, num_heads=num_heads, num_layers=num_layers,
                          target_vocab_size=target_vocab_size, input_vocab_size=input_vocab_size,
                          dff=dff, rate=dropout_rate)

# Trainer
print(f'\n\nBeginning training for {EPOCHS} epochs @ {datetime.now()}...\n')
trainer = Trainer(train_dataset=train_dataset,
                  test_dataset=test_dataset,
                  learning_rate=learning_rate,
                  optimizer=optimizer,
                  transformer=transformer,
                  epochs=EPOCHS)

loss_hist, acc_hist = trainer.train()
