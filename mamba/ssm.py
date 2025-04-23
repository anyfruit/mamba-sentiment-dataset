import tensorflow as tf

def dynamic_state_update(input_seq, delta, mat_A, mat_B, mat_C, gain):

    # batch size b
    # sequence length l
    # feature dimension d
    # number of states n

    # input_seq: [b, l, d]
    # delta: [b, l, d]
    # mat_A: [d, n]
    # mat_B: [b, l, n]
    # mat_C: [b, l, n]
    # gain: [d]

    update = tf.einsum('bld,dn->bldn', delta, mat_A)
    input_mix = tf.einsum('bld,bld,bln->bldn', delta, input_seq, mat_B)

    shifted_update = tf.pad(update[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]
    shifted_update = tf.reverse(shifted_update, axis=[1])
    cum_sum = tf.math.cumsum(shifted_update, axis=1)
    exp_update = tf.exp(cum_sum)
    exp_update = tf.reverse(exp_update, axis=[1])

    # Apply kernel
    inter = input_mix * exp_update
    normed = tf.math.cumsum(inter, axis=1) / (exp_update + 1e-12)

    # Residual
    readout = tf.einsum('bldn,bln->bld', normed, mat_C)

    return readout + input_seq * gain