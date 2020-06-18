import os
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
from examples.utils.config import Config

config = Config()

def read_data(filename):
    """ Reading the zip file to extract text """
    text = []
    with open(filename, 'r', encoding='utf-8') as f:
        i = 0
        for row in f:
            text.append(row)
            i += 1
    return text


def sents2sequences(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):
    encoded_text = tokenizer.texts_to_sequences(sentences)
    preproc_text = pad_sequences(encoded_text, padding=padding_type, maxlen=pad_length)
    if reverse:
        preproc_text = np.flip(preproc_text, axis=1)

    return preproc_text


def get_data(train_size, random_seed=100):

    """ Getting randomly shuffled training / testing data """
    en_text = read_data(os.path.join(config.DATA_DIR, 'small_vocab_en.txt'))
    fr_text = read_data(os.path.join(config.DATA_DIR, 'small_vocab_fr.txt'))

    fr_text = ['sos ' + sent[:-1] + 'eos .'  if sent.endswith('.') else 'sos ' + sent + ' eos .' for sent in fr_text]

    np.random.seed(random_seed)
    inds = np.arange(len(en_text))
    np.random.shuffle(inds)

    train_inds = inds[:train_size]
    test_inds = inds[train_size:]
    tr_en_text = [en_text[ti] for ti in train_inds]
    tr_fr_text = [fr_text[ti] for ti in train_inds]

    ts_en_text = [en_text[ti] for ti in test_inds]
    ts_fr_text = [fr_text[ti] for ti in test_inds]

    return tr_en_text, tr_fr_text, ts_en_text, ts_fr_text