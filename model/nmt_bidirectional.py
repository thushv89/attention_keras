from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional, Embedding
from tensorflow.python.keras.models import Model
from layers.attention import AttentionLayer


def define_nmt(hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
    """ Defining a NMT model """

    # Define an input sequence and process it.
    encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
    decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')

    # Encoder GRU
    encoder_gru = Bidirectional(GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru'), name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_gru(encoder_inputs)

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = Bidirectional(GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru'), name='bidirectional_decoder')
    decoder_out, decoder_fwd_state, decoder_back_state = decoder_gru(decoder_inputs, initial_state=[encoder_fwd_state, encoder_back_state])

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary()

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
    encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, 2*hidden_size), name='encoder_inf_states')
    decoder_init_fwd_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_fwd_init')
    decoder_init_back_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_back_init')

    decoder_inf_out, decoder_inf_fwd_state, decoder_inf_back_state = decoder_gru(decoder_inf_inputs, initial_state=[decoder_init_fwd_state, decoder_init_back_state])
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_fwd_state, decoder_init_back_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_fwd_state, decoder_inf_back_state])

    return full_model, encoder_model, decoder_model


def define_nmt2(maxlen):

    input_1 = Input(shape=(maxlen,), name="input1")
    x = Embedding(25000, 128, input_length=maxlen, name='embedding_1', trainable=False)(input_1)
    encoder_out, forward_h, backward_h = Bidirectional(GRU(32, return_sequences=True, return_state=True))(x)
    decoder_out, forward_h, backward_h = Bidirectional(GRU(32, return_sequences=True, return_state=True))(x,
                                                                                                          initial_state=[
                                                                                                              forward_h,
                                                                                                              backward_h])
    print('encoder_out > ', encoder_out.shape)
    print('decoder_out > ', decoder_out.shape)

    attn_out, attn_states = AttentionLayer()([encoder_out, decoder_out])
    a = Concatenate([decoder_out, attn_out], axis=1)

    dense = Dense(25000, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(a)

    # Full model
    full_model = Model(inputs=x, outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary()

if __name__ == '__main__':

    """ Checking nmt model for toy example """
    #define_nmt(64, 32, 20, 30, 20, 20)
    define_nmt2(100)