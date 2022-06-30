import logging

import tensorflow as tf

import numpy as np
import os

import src.examples.utils.data_helper as data_helper
import src.examples.nmt.model as model
import src.examples.utils.model_helper as model_helper

logger = logging.getLogger(__name__)

batch_size = 64
hidden_size = 96
en_timesteps, fr_timesteps = 20, 20


def run_training(input_sentences, target_sentences, model_save_dir, debug=False):

    """ Defining tokenizers """
    en_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='UNK')
    en_tokenizer.fit_on_texts(input_sentences)

    fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='UNK')
    fr_tokenizer.fit_on_texts(target_sentences)

    """ Getting preprocessed data """
    en_seq, fr_seq = data_helper.preprocess_data(en_tokenizer, fr_tokenizer, input_sentences, target_sentences, en_timesteps, fr_timesteps)

    logger.info('Vocabulary size (English): {}'.format(np.max(en_seq) + 1))
    logger.info('Vocabulary size (French): {}'.format(np.max(fr_seq) + 1))
    logger.debug('En text shape: {}'.format(en_seq.shape))
    logger.debug('Fr text shape: {}'.format(fr_seq.shape))

    en_vsize = data_helper.compute_vocabulary_size(en_tokenizer)
    fr_vsize = data_helper.compute_vocabulary_size(fr_tokenizer)

    """ Defining the full model """
    full_model, infer_enc_model, infer_dec_model = model.define_nmt(
        hidden_size=hidden_size, batch_size=batch_size,
        en_timesteps=en_timesteps, fr_timesteps=fr_timesteps,
        en_vsize=en_vsize, fr_vsize=fr_vsize)

    n_epochs = 10 if not debug else 3
    model_helper.train(full_model, en_seq, fr_seq, en_tokenizer, fr_tokenizer, batch_size, n_epochs)

    """ Save model """
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    full_model.save(model_save_dir)

    return full_model, infer_enc_model, infer_dec_model, en_tokenizer, fr_tokenizer


def translate_single(input_sentence, infer_encoder_model, infer_decoder_model, en_tokenizer, fr_tokenizer, en_pad_length, return_attention=False):

    """ Inferring with trained model """
    logger.debug('English: {}'.format(input_sentence))

    test_en_seq = data_helper.sents2sequences(en_tokenizer, [input_sentence], pad_length=en_pad_length)

    translated_sentence, attn_weights = model_helper.infer_nmt(
        encoder_model=infer_encoder_model, decoder_model=infer_decoder_model,
        test_en_seq=test_en_seq, en_tokenizer=en_tokenizer, fr_tokenizer=fr_tokenizer)

    logger.debug('\tFrench: {}'.format(translated_sentence))

    if not return_attention:
        return translated_sentence
    else:
        return translated_sentence, attn_weights

