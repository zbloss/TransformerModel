# TransformerModel
The Transformer model described in the ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf) paper
written in Tensorflow 2.0.

This package has been written very abstractly so that you can feel free to use any of the classes involved. For example,
you can hack the Encoder and Decoder classes to create other models such as BERT, Transformer-XL, GPT, GPT-2, 
and even XLNet if you're brave enough. All that you would need to adjust is the masking class.

<hr >

## Requirements
* `matplotlib`
* `numpy`
* `pandas`
* `tensorflow-datasets>=1.1.0`
* `tensorflow>=2.0.0b1`

<hr>

## How to install
`pip install transformer-model`

I highly recommend creating a new virtual environment for you project as this package does require
Tensorflow 2.0 which as of now is still in beta.

#### GPU
You can utilize your GPU by installing the gpu version of Tensorflow 2.0
    `pip install tensorflow-gpu==2.0.0b1`
<hr>

## How to use
I have left an example file (`example.py`) file for you to get a feel of how this model works.
The model, in it's most basic form, takes a numpy array as the input and returns a numpy array as the output.

The common use case with this model is for language translation. In that case, you would train this model with the feature
column being the original language and the target column being the language you want to translate to.

<b>1. Generate your input/output.</b>   
*   I have provided a few helper functions for this, but you essentially need to generate two Tensorflow tokenizers
as well as a pandas DataFrame with feature and target columns. 
*   You can utilize the helper functions in the DataProcess class to generate TensorDatasets from your DataFrame,
    as well as perform a train_test_split.
    
<b>2. Learning Rate / Optimizer</b>
*   The Transformer model excels when set on a custom learning rate with a sharp incline and then 
    exponential decay.
*   This work has been implemented in the CustomSchedule() class. Feel free to play around with the 
    number of warm up steps to complete the schedule!
    
<b>3. Define the Transformer Model</b>
*   Create the HPARAMS that will define how large the model should be.
    
<b>4. Train</b>
*   Once you define the Trainer class, you only need to call the `train()` method on your trainer object.
* This will return the training accuracy and loss


[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=7TZ7CL23G7BCQ&currency_code=USD&source=url)
