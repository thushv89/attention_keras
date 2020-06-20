import pytest
from layers.attention import AttentionLayer
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow as tf


def test_attention_layer_standalone_fixed_b_fixed_t():
    """
    Tests fixed batch size and time steps
    Encoder and decoder has variable seq length and latent dim
    """
    inp1 = Input(batch_shape=(5,10,15))
    inp2 = Input(batch_shape=(5,15,25))
    out, e_out = AttentionLayer()([inp1, inp2])
    assert out.shape == tf.TensorShape([inp2.shape[0], inp2.shape[1], inp1.shape[2]])
    assert e_out.shape == tf.TensorShape([inp1.shape[0], inp2.shape[1], inp1.shape[1]])

def check_tensorshape_equal(shape1, shape2):

    print(shape1, shape2)
    equal = []
    for s1, s2 in zip(shape1, shape2):
        if (s1 == s2) == None:
            equal.append(True)
        else:
            equal.append(s1==s2)
    return all(equal)

def test_attention_layer_standalone_none_b_fixed_t():
    inp1 = Input(shape=(10, 15))
    inp2 = Input(shape=(15, 25))
    out, e_out = AttentionLayer()([inp1, inp2])

    assert check_tensorshape_equal(out.shape, tf.TensorShape([None, inp2.shape[1], inp1.shape[2]]))
    assert check_tensorshape_equal(e_out.shape, tf.TensorShape([None, inp2.shape[1], inp1.shape[1]]))


def test_attention_layer_standalone_none_b_none_t():
    inp1 = Input(shape=(None, 15))
    inp2 = Input(shape=(None, 25))
    out, e_out = AttentionLayer()([inp1, inp2])

    assert check_tensorshape_equal(out.shape, tf.TensorShape([None, None, inp1.shape[2]]))
    assert check_tensorshape_equal(e_out.shape, tf.TensorShape([None, None, None]))


'''def test_attention_layer_nmt_none_b_fixed_t():


    encoder_inputs = Input(shape=(12, 75), name='encoder_inputs')
    decoder_inputs = Input(shape=(16 - 1, 80), name='decoder_inputs')

    # Encoder GRU
    encoder_gru = GRU(32, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(encoder_inputs)

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(32, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(80, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    assert decoder_pred.shape == tf.TensorShape([])

def test_attention_layer_nmt_none_b_none_t():

    pass'''