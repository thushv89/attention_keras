import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import src.examples.utils.data_helper as data_helper

logger = logging.getLogger(__name__)


def train(full_model, en_seq, fr_seq, en_tokenizer, fr_tokenizer, batch_size, n_epochs=10):
    """ Training the model """

    en_vsize = data_helper.compute_vocabulary_size(en_tokenizer)
    fr_vsize = data_helper.compute_vocabulary_size(fr_tokenizer)

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, en_seq.shape[0] - batch_size, batch_size):

            en_onehot_seq = tf.keras.utils.to_categorical(en_seq[bi:bi + batch_size, :], num_classes=en_vsize)
            fr_onehot_seq = tf.keras.utils.to_categorical(fr_seq[bi:bi + batch_size, :], num_classes=fr_vsize)

            full_model.train_on_batch([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :])

            l = full_model.evaluate([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :],
                                    batch_size=batch_size, verbose=0)

            losses.append(l)
        if (ep + 1) % 1 == 0:
            logger.info("Loss in epoch {}: {}".format(ep + 1, np.mean(losses)))


def infer_nmt(encoder_model, decoder_model, test_en_seq, en_tokenizer, fr_tokenizer):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_tokenizer: English tokenizer
    :param fr_tokenizer: French tokenizer
    :return:
    """

    en_vsize = data_helper.compute_vocabulary_size(en_tokenizer)
    fr_vsize = data_helper.compute_vocabulary_size(fr_tokenizer)
    fr_index2word = data_helper.generate_index2word(fr_tokenizer)

    test_fr_seq = data_helper.sents2sequences(fr_tokenizer, ['sos'], fr_vsize)
    test_en_onehot_seq = tf.keras.utils.to_categorical(test_en_seq, num_classes=en_vsize)
    test_fr_onehot_seq = np.expand_dims(tf.keras.utils.to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

    enc_outs, enc_last_state = encoder_model.predict(test_en_onehot_seq)
    dec_state = enc_last_state
    attention_weights = []
    fr_text = ''
    for i in range(20):

        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, test_fr_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

        if dec_ind == 0:
            break
        test_fr_seq = data_helper.sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], fr_vsize)
        test_fr_onehot_seq = np.expand_dims(tf.keras.utils.to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

        attention_weights.append((dec_ind, attention))
        fr_text += fr_index2word[dec_ind] + ' '

    return fr_text, attention_weights


def plot_attention_weights(encoder_inputs, attention_weights, en_id2word, fr_id2word):
    """
    Plots attention weights
    :param encoder_inputs: Sequence of word ids (list/numpy.ndarray)
    :param attention_weights: Sequence of (<word_id_at_decode_step_t>:<attention_weights_at_decode_step_t>)
    :param en_id2word: dict
    :param fr_id2word: dict
    :return:
    """

    if len(attention_weights) == 0:
        raise ValueError(
            """Your attention weights was empty. No attention map saved to the disk.
            \nPlease check if the decoder produced  a proper translation"""
        )

    mats = []
    dec_inputs = []
    for dec_ind, attn in attention_weights:
        mats.append(attn.reshape(-1))
        dec_inputs.append(dec_ind)
    attention_mat = np.transpose(np.array(mats))

    fig, ax = plt.subplots(figsize=(32, 32))
    ax.imshow(attention_mat)

    ax.set_xticks(np.arange(attention_mat.shape[1]))
    ax.set_yticks(np.arange(attention_mat.shape[0]))

    ax.set_xticklabels([fr_id2word[inp] if inp != 0 else "<Res>" for inp in dec_inputs])
    ax.set_yticklabels([en_id2word[inp] if inp != 0 else "<Res>" for inp in encoder_inputs.ravel()])

    ax.tick_params(labelsize=32)
    ax.tick_params(axis='x', labelrotation=90)
