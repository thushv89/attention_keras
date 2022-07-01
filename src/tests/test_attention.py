from src.layers.attention import AttentionLayer
import tensorflow as tf


def test_attention_layer_standalone_batch_fixed_time_fixed():
    """
    Tests fixed batch size and time steps
    Encoder and decoder has variable seq length and latent dim
    """
    inp1 = tf.keras.layers.Input(batch_shape=(5,10,15))
    inp2 = tf.keras.layers.Input(batch_shape=(5,15,25))
    out, e_out = AttentionLayer()([inp1, inp2])
    assert out.shape == tf.TensorShape([inp2.shape[0], inp2.shape[1], inp1.shape[2]])
    assert e_out.shape == tf.TensorShape([inp1.shape[0], inp2.shape[1], inp1.shape[1]])


def check_tensorshape_equal(shape1, shape2):

    equal = []
    for s1, s2 in zip(shape1, shape2):
        if s1 is not None and s2 is not None:
            equal.append(s1==s2)
    return all(equal)


def test_attention_layer_standalone_batch_none_time_fixed():
    inp1 = tf.keras.layers.Input(shape=(10, 15))
    inp2 = tf.keras.layers.Input(shape=(15, 25))
    out, e_out = AttentionLayer()([inp1, inp2])

    assert check_tensorshape_equal(out.shape, tf.TensorShape([None, inp2.shape[1], inp1.shape[2]]))
    assert check_tensorshape_equal(e_out.shape, tf.TensorShape([None, inp2.shape[1], inp1.shape[1]]))


def test_attention_layer_standalone_batch_none_time_none():
    inp1 = tf.keras.layers.Input(shape=(None, 15))
    inp2 = tf.keras.layers.Input(shape=(None, 25))
    out, e_out = AttentionLayer()([inp1, inp2])

    assert check_tensorshape_equal(out.shape, tf.TensorShape([None, None, inp1.shape[2]]))
    assert check_tensorshape_equal(e_out.shape, tf.TensorShape([None, None, None]))

