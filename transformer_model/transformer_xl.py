import tensorflow as tf


def positional_embedding(position_sequence, inverse_frequency, batch_size=None):
    """
    Calculates the fixed sinusoidal embedding
    :param position_sequence: the array containing the position sequence
    :param inverse_frequency: the inverse of the position sequence
    :param batch_size: batch_size of the model
    :return: position embedding vector
    """
    sinusoid_input = tf.einsum('i,j-ij', position_sequence, inverse_frequency)
    pos_emb = tf.concat(
        [tf.sin(sinusoid_input), tf.cos(sinusoid_input)], axis=-1
    )

    if batch_size is not None:
        return tf.tile(pos_emb[:, None, :],
                       [1, batch_size, 1])
    else:
        return pos_emb[:, None, :]


def positionwise_ff(input, d_model, d_inner, dropout,
                    kernel_initializer, is_training=True):
    """
    Feed forward network to process the output of each attention module into the
    needed dimension size for the next layer.
    :param input: attention layer output
    :param d_model: dimension of the next attention layer
    :param d_inner: hidden layer size
    :param dropout: amount of dropout to use
    :param kernel_initializer: which kernel_initializer to use
    :param is_training: bool for whether the layer is trainable or not
    :return: processed tensor ready for next attention layer
    """
    output = input

    output = tf.keras.layers.Dense(input, d_inner, activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   name='dense_layer_1')
    output = tf.keras.layers.Dropout(output, dropout, trainable=is_training,
                                     name='dropout_1')
    output = tf.keras.layers.Dense(output, d_model,
                                   kernel_initializer=kernel_initializer,
                                   name='dense_layer_2')
    output = tf.keras.layers.Dropout(output, dropout, trainable=is_training,
                                     name='dropout_2')
    return output


def rel_shift(x):
    """
    Helper function to reshape and shift data.
    :param x: data to be shifted
    :return: shifted data
    """
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


def relative_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                            n_head, d_head, dropout, dropout_attn, is_training,
                            kernel_initializer):
    """
    Computes the relative MultiHeaded-Attention portion of the model
    :param w: previous sequence weights
    :param r: current sequence weights
    :param r_w_bias: current weights -> previous weights bias
    :param r_r_bias: current weights -> current weights bias
    :param attn_mask: relative attention masking to use
    :param mems:
    :param d_model:
    :param n_head:
    :param d_head:
    :param dropout:
    :param dropout_attn:
    :param is_training:
    :param kernel_initializer:
    :return:
    """

    scale = 1 / (d_head ** 0.5)
    qlen = tf.shape(w)[0]
    rlen = tf.shape(w)[0]
    batch_size = tf.shape(w)[1]

    cat = tf.concat([mems, w], 0) if mems is not None and mems.shape.ndims > 1 else w

    # computing query, key, value vectors (qkv)
    w_heads = tf.keras.layers.Dense(cat, 3 * n_head * d_head, use_bias=False,
                                    kernel_initializer=kernel_initializer, name='qkv')
    # computing recurrence weight vector (r)
    r_head_k = tf.keras.layers.Dense(r, n_head * d_head, use_bias=False,
                                     kernel_initializer=kernel_initializer, name='r')

    # splitting into individual weight matrices
    w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
    w_head_q, = w_head_q[-qlen:]

    r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

    rw_head_q = w_head_q + r_w_bias
    rr_head_q = w_head_q + r_r_bias

    ac = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
    bd = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
    bd = rel_shift(bd)

    attn_score = (ac + bd) * scale
    attn_mask_t = attn_mask[:, :, None, None]
    attn_score = attn_score * (1 - attn_mask_t) - (1e30 * attn_mask_t)

    attn_prob = tf.nn.softmax(attn_score, axis=1)
    attn_prob = tf.keras.layers.Dropout(attn_prob, dropout_attn, trainable=is_training)

    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
    size_t = tf.shape(attn_vec)
    attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

    attn_out = tf.keras.layers.Dense(attn_vec, d_model, use_bias=False,
                                     kernel_initializer=kernel_initializer, name='o')
    attn_out = tf.keras.layers.Dropout(attn_out, dropout, training=is_training)

    output = tf.keras.layers.LayerNormalization(attn_out + w, begin_norm_axis=-1)

    return output


def embedding_lookup(lookup_table, x, use_tpu=False):
    if use_tpu:
        n_token = tf.shape(lookup_table)[0]
        one_hot_idx = tf.one_hot(x, n_token)

        if one_hot_idx.shape.ndims == 2:
            return tf.einsum('nd,in->id', lookup_table, one_hot_idx)
        else:
            return tf.einsum('nd,ibn->ibd', lookup_table, one_hot_idx)
    else:
        return tf.nn.embedding_lookup(lookup_table, x)


def mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer,
                                   proj_initializer, div_val=1,
                                   proj_same_dim=True, **kwargs):
    emb_scale = d_proj ** 0.5

    if div_val == 1:
        lookup_table = tf.get
        # embedding_layer = tf.keras.layers.Embedding(n_token, d_embed)
