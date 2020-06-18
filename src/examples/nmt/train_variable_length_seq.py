import tensorflow.keras as keras

from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os, sys

project_path = os.environ.get("PWD")
if project_path not in sys.path:
    sys.path.append(project_path)

from examples.utils.data_helper import read_data, sents2sequences, get_data
from examples.nmt.model import define_nmt
from examples.utils.model_helper import plot_attention_weights
from examples.utils.logger import get_logger
from examples.utils.config import Config

config = Config()

logger = get_logger("examples.nmt.train_with_none", config.LOGS_DIR)

batch_size = 64
hidden_size = 96


def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
    """ Preprocessing data and getting a sequence of word indices """

    en_seq = sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
    fr_seq = sents2sequences(fr_tokenizer, fr_text, pad_length=fr_timesteps)
    logger.info('Vocabulary size (English): {}'.format(np.max(en_seq)+1))
    logger.info('Vocabulary size (French): {}'.format(np.max(fr_seq)+1))
    logger.debug('En text shape: {}'.format(en_seq.shape))
    logger.debug('Fr text shape: {}'.format(fr_seq.shape))

    return en_seq, fr_seq


def train(full_model, en_seq, fr_seq, batch_size, n_epochs=10):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, en_seq.shape[0] - batch_size, batch_size):

            en_onehot_seq = to_categorical(en_seq[bi:bi + batch_size, :], num_classes=en_vsize)
            fr_onehot_seq = to_categorical(fr_seq[bi:bi + batch_size, :], num_classes=fr_vsize)

            full_model.train_on_batch([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :])

            l = full_model.evaluate([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :],
                                    batch_size=batch_size, verbose=0)

            losses.append(l)
        if (ep + 1) % 1 == 0:
            logger.info("Loss in epoch {}: {}".format(ep + 1, np.mean(losses)))


def infer_nmt(encoder_model, decoder_model, test_en_seq, en_vsize, fr_vsize):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_vsize: int
    :param fr_vsize: int
    :return:
    """

    test_fr_seq = sents2sequences(fr_tokenizer, ['sos'], fr_vsize)
    test_en_onehot_seq = to_categorical(test_en_seq, num_classes=en_vsize)
    test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

    enc_outs, enc_last_state = encoder_model.predict(test_en_onehot_seq)
    dec_state = enc_last_state
    attention_weights = []
    fr_text = ''
    for i in range(20):

        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, test_fr_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

        if dec_ind == 0:
            break
        test_fr_seq = sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], fr_vsize)
        test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

        attention_weights.append((dec_ind, attention))
        fr_text += fr_index2word[dec_ind] + ' '

    return fr_text, attention_weights


if __name__ == '__main__':

    debug = False
    """ Hyperparameters """

    train_size = 100000 if not debug else 10000
    filename = ''

    tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data(train_size=train_size)


    """ Defining tokenizers """
    en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    en_tokenizer.fit_on_texts(tr_en_text)

    fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    fr_tokenizer.fit_on_texts(tr_fr_text)

    en_encoded_text = en_tokenizer.texts_to_sequences(tr_en_text)
    fr_encoded_text = fr_tokenizer.texts_to_sequences(tr_fr_text)

    en_encoded_text, fr_encoded_text = zip(*sorted(zip(en_encoded_text, fr_encoded_text), key=lambda x: len(x[0])))

    " Get best pad lengths "
    n_train = len(tr_en_text)
    n_one_third = int(n_train/3)
    n_two_third = int(2*n_train / 3)

    tr_en_one_third_lengths = [len(x) for x in en_encoded_text[:n_one_third]]
    tr_fr_one_third_lengths = [len(x) for x in fr_encoded_text[:n_one_third]]
    tr_en_one_third_mean, tr_en_one_third_std = np.mean(tr_en_one_third_lengths), np.std(tr_en_one_third_lengths)
    tr_fr_one_third_mean, tr_fr_one_third_std = np.mean(tr_fr_one_third_lengths), np.std(tr_fr_one_third_lengths)
    logger.info("1/3 en training data sequence length: mean {}, std {}".format(tr_en_one_third_mean, tr_en_one_third_std))
    logger.info("1/3 fr training data sequence length: mean {}, std {}".format(tr_fr_one_third_mean, tr_fr_one_third_std))

    tr_en_two_third_lengths = [len(x) for x in en_encoded_text[n_one_third:n_two_third]]
    tr_fr_two_third_lengths = [len(x) for x in fr_encoded_text[n_one_third:n_two_third]]
    tr_en_two_third_mean, tr_en_two_third_std = np.mean(tr_en_two_third_lengths), np.std(tr_en_two_third_lengths)
    tr_fr_two_third_mean, tr_fr_two_third_std = np.mean(tr_fr_two_third_lengths), np.std(tr_fr_two_third_lengths)
    logger.info(
        "1/3-2/3 en training data sequence length: mean {}, std {}".format(tr_en_two_third_mean, tr_en_two_third_std))
    logger.info(
        "1/3-2/3 fr training data sequence length: mean {}, std {}".format(tr_fr_two_third_mean, tr_fr_two_third_std))

    tr_en_three_third_lengths = [len(x) for x in en_encoded_text[n_two_third:]]
    tr_fr_three_third_lengths = [len(x) for x in fr_encoded_text[n_two_third:]]
    tr_en_three_third_mean, tr_en_three_third_std = np.mean(tr_en_three_third_lengths), np.std(tr_en_three_third_lengths)
    tr_fr_three_third_mean, tr_fr_three_third_std = np.mean(tr_fr_three_third_lengths), np.std(tr_fr_three_third_lengths)
    logger.info(
        "2/3-3/3 en training data sequence length: mean {}, std {}".format(tr_en_three_third_mean, tr_en_three_third_std))
    logger.info(
        "2/3-3/3 fr training data sequence length: mean {}, std {}".format(tr_fr_three_third_mean, tr_fr_three_third_std))


    """ Getting preprocessed data """

    en_1_timesteps = int(tr_en_one_third_mean + 2*tr_en_one_third_std)
    fr_1_timesteps = int(tr_fr_one_third_mean + 2*tr_en_one_third_std)
    en_seq_1, fr_seq_1 = preprocess_data(
        en_tokenizer, fr_tokenizer, tr_en_text[:n_one_third], tr_fr_text[:n_one_third], en_1_timesteps, fr_1_timesteps)
    en_2_timesteps = int(tr_en_two_third_mean + 2 * tr_en_two_third_std)
    fr_2_timesteps = int(tr_fr_two_third_mean + 2 * tr_en_two_third_std)
    en_seq_2, fr_seq_2 = preprocess_data(
        en_tokenizer, fr_tokenizer, tr_en_text[n_one_third:n_two_third], tr_fr_text[n_one_third:n_two_third],
        en_2_timesteps, fr_2_timesteps)
    en_3_timesteps = int(tr_en_three_third_mean + 2 * tr_en_three_third_std)
    fr_3_timesteps = int(tr_fr_three_third_mean + 2 * tr_en_three_third_std)
    en_seq_3, fr_seq_3 = preprocess_data(
        en_tokenizer, fr_tokenizer, tr_en_text[int(2*n_train/3):], tr_fr_text[int(2*n_train/3):], en_3_timesteps, fr_3_timesteps)

    en_vsize = max(en_tokenizer.index_word.keys()) + 1
    fr_vsize = max(fr_tokenizer.index_word.keys()) + 1

    """ Defining the full model """
    full_model, infer_enc_model, infer_dec_model = define_nmt(
        hidden_size=hidden_size, batch_size=None,
        en_timesteps=None, fr_timesteps=None,
        en_vsize=en_vsize, fr_vsize=fr_vsize)

    n_epochs = 10 if not debug else 3
    for ep in range(n_epochs):
        logger.info("Running main epoch {}".format(ep))
        if ep == 0:
            logger.info("Training with {},{}".format(en_1_timesteps, fr_1_timesteps))
            logger.info("Training with {},{}".format(en_2_timesteps, fr_2_timesteps))
            logger.info("Training with {},{}".format(en_3_timesteps, fr_3_timesteps))
        train(full_model, en_seq_1, fr_seq_1, batch_size, 1)
        train(full_model, en_seq_2, fr_seq_2, batch_size, 1)
        train(full_model, en_seq_3, fr_seq_3, batch_size, 1)

    """ Save model """
    if not os.path.exists(config.MODELS_DIR):
        os.mkdir(config.MODELS_DIR)
    full_model.save(os.path.join(config.MODELS_DIR, 'nmt.h5'))

    """ Index2word """
    en_index2word = dict(zip(en_tokenizer.word_index.values(), en_tokenizer.word_index.keys()))
    fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))

    """ Inferring with trained model """
    test_en = ts_en_text[0]
    logger.info('Translating: {}'.format(test_en))

    test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=20)
    test_fr, attn_weights = infer_nmt(
        encoder_model=infer_enc_model, decoder_model=infer_dec_model,
        test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize)
    logger.info('\tFrench: {}'.format(test_fr))

    """ Attention plotting """
    plot_attention_weights(test_en_seq, attn_weights, en_index2word, fr_index2word)