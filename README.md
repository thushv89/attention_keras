# Keras Attention Layer

## Version (s)

- TensorFlow: 1.12.0 (Tested)
- TensorFlow: 2.0 (Should be easily portable as all the backend functions are availalbe in TF 2.0. However has not been tested yet.)

## Introduction

This is an implementation of Attention (only supports [Bahdanau Attention](https://arxiv.org/pdf/1409.0473.pdf) right now)

## Project structure

```
data (Download data and place it here)
 |--- small_vocab_en.txt
 |--- small_vocab_fr.txt
layers
 |--- attention.py (Attention implementation)
examples
 |--- nmt
   |--- model.py (NMT model defined with Attention)
   |--- train.py ( Code for training/inferring/plotting attention with NMT model)
 |--- nmt_bidirectional
   |--- model.py (NMT birectional model defined with Attention)
   |--- train.py ( Code for training/inferring/plotting attention with NMT model)
h5.models (created by train_nmt.py to store model)
results (created by train_nmt.py to store model)

```
## How to use

Just like you would use any other `tensoflow.python.keras.layers` object.

```python
from attention_keras.layers.attention import AttentionLayer

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

```

Here,

- `encoder_outputs` - Sequence of encoder ouptputs returned by the RNN/LSTM/GRU (i.e. with `return_sequences=True`)
- `decoder_outputs` - The above for the decoder
- `attn_out` - Output context vector sequence for the decoder. This is to be concat with the output of decoder (refer `model/nmt.py` for more details)
- `attn_states` - Energy values if you like to generate the heat map of attention (refer `model.train_nmt.py` for usage)

## Visualizing Attention weights

An example of attention weights can be seen in `model.train_nmt.py`

After the model trained attention result should look like below.

![Attention heatmap](https://github.com/thushv89/attention_keras/blob/master/results/attention.png)

## Running the NMT example

In order to run the example you need to download `small_vocab_en.txt` and `small_vocab_fr.txt` from [Udacity deep learning repository](https://github.com/udacity/deep-learning/tree/master/language-translation/data) and place them in the `data` folder.

___

If you have improvements (e.g. other attention mechanisms), contributions are welcome!