# Keras Attention Layer

## Version (s)

- TensorFlow: 1.12.0

## Introduction

This is an implementation of Attention (only supports Bahdanau Attention right now)

## Project structure

```
data (Download data and place it here)
 |--- small_vocab_en.txt
 |--- small_vocab_fr.txt
layers
 |--- attention.py (Attention implementation)
model
 |--- nmt.py (NMT model defined with Attention)
 |--- train_nmt.py ( Code for training/inferring/plotting attention with NMT model)
h5.models (created to store model)

```
## How to use

Just like you would use any other `tensoflow.python.keras.layers` object.

```python
from attention_keras.layers import AttentionLayer

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

 