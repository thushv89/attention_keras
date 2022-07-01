import pytest
import numpy as np
import tensorflow as tf
import src.layers.attention as attention


@pytest.fixture(scope="function")
def processed_data():

    return tf.keras.utils.to_categorical(np.array(
        [[1, 2, 3, 4], [3, 4, 5, 0], [5, 6, 0, 0], [2, 4, 0, 0]]
    ), num_classes=7), tf.keras.utils.to_categorical(np.array(
        [[4, 3, 2, 1, 0], [5, 4, 3, 2, 0], [6, 5, 4, 3, 7], [3, 0, 0, 0, 0]]
    ), num_classes=8)


@pytest.fixture(scope="function")
def mock_model():

    encoder_inputs = tf.keras.layers.Input(shape=(4, 7), name='encoder_inputs')
    decoder_inputs = tf.keras.layers.Input(shape=(4, 8), name='decoder_inputs')

    encoder_gru = tf.keras.layers.GRU(16, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(encoder_inputs)

    decoder_gru = tf.keras.layers.GRU(16, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)

    attn_layer = attention.AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    dense = tf.keras.layers.Dense(8, activation='softmax', name='softmax_layer')
    dense_time = tf.keras.layers.TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    full_model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    return full_model