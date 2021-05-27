# -*-coding:utf-8-*-
__author__ = 'zhangqi49'

import tensorflow as tf
import numpy as np
from tcn_2 import  *


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]


def multihead_attn_ori(w, attn_mask, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, head_attention_flag=False, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]
        w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
                                  kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        if head_attention_flag:
            cross_w = tf.transpose(w, [1, 0, 2])
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(d_model) for i in range(2)])
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, cross_w, dtype=tf.float32)
            output_rnn = output_rnn[:, -1, :]
            cross_head_attention = tf.layers.dense(output_rnn, n_head * n_head, use_bias=False,
                                                   kernel_initializer=kernel_initializer, name='cross_att')
            cross_head_attention = tf.reshape(cross_head_attention, [bsz, n_head, n_head])
            cross_head_attention = tf.nn.softmax(cross_head_attention)

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [qlen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [qlen, bsz, n_head, d_head])

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k)

        attn_score = attn_score * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        tf.add_to_collection('attentions', attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        if head_attention_flag:
            attn_vec = tf.einsum('ibnd,bmn->ibmd', attn_vec, cross_head_attention)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output


def multihead_attn(w, c_m, m_p, side, attn_mask, d_model,
                   n_head, d_head, dropout, dropatt, is_training,
                   kernel_initializer, head_attention_flag=False, side_flag=True, cp_flag=True, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]
        qk_w = w
        if cp_flag:
            qk_w = tf.concat([qk_w, c_m, m_p], axis=2)
        if side_flag:
            qk_w = tf.concat([qk_w, side], axis=2)

        w_head_v = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='v')
        w_heads_qk = tf.layers.dense(w, 2 * n_head * d_head, use_bias=False,
                                     kernel_initializer=kernel_initializer, name='qk')

        #######################################
        # c_m=positionwise_FF(c_m,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='c_m_1',is_training=is_training)
        # m_p=positionwise_FF(m_p,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='m_p_1',is_training=is_training)
        # side=positionwise_FF(side,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='side_1',is_training=is_training)
        # qk_w = tf.layers.dense(qk_w, d_model,kernel_initializer=kernel_initializer, name='qk_r')
        # qk_w = tf.concat([w, qk_w], axis=2)
        #
        # w_heads_qk_2 = tf.layers.dense(qk_w, 2 * n_head * d_head, use_bias=False,
        #                                kernel_initializer=kernel_initializer, name='qk_2')
        ########################################

        # w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        # if head_attention_flag:
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q, w_head_k = tf.split(w_heads_qk, 2, -1)

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [qlen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [qlen, bsz, n_head, d_head])

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k)

        attn_score = attn_score * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        tf.add_to_collection('attentions', attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        if head_attention_flag:
            cross_w = tf.transpose(w, [1, 0, 2])
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(d_model) for i in range(2)])
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, cross_w, dtype=tf.float32)
            output_rnn = output_rnn[:, -1, :]
            cross_head_attention = tf.layers.dense(output_rnn, n_head * n_head, use_bias=False,
                                                   kernel_initializer=kernel_initializer, name='cross_att')
            cross_head_attention = tf.reshape(cross_head_attention, [bsz, n_head, n_head])
            cross_head_attention = tf.nn.softmax(cross_head_attention)
            attn_vec = tf.einsum('ibnd,bmn->ibmd', attn_vec, cross_head_attention)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output,c_m,m_p,side,

def multihead_attn_tpe(w, c_m, m_p, side, attn_mask, d_model,
                   n_head, d_head, dropout, dropatt, is_training,
                   kernel_initializer, head_attention_flag=False, side_flag=False, cp_flag=False, qlen_input=12 ,scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]

        #############TPE##################
        # def rel_shift(x):
        #     x_size = tf.shape(x)
        #
        #     x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        #     x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        #     x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        #     x = tf.reshape(x, x_size)
        #
        #     return x
        #
        # qlen = tf.shape(w)[0]
        # rlen = tf.shape(r)[0]
        # bsz = tf.shape(w)[1]
        #
        # cat = tf.concat([mems, w],
        #                 0) if mems is not None and mems.shape.ndims > 1 else w
        # w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        #
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        # w_head_q = w_head_q[-qlen:]
        #
        # klen = tf.shape(w_head_k)[0]
        #
        # w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        # w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        # w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])
        #
        # r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])
        #
        # rw_head_q = w_head_q + r_w_bias
        # rr_head_q = w_head_q + r_r_bias
        #
        # AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        # BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        # BD = rel_shift(BD)
        #
        # attn_score = (AC + BD) * scale
        # attn_mask_t = attn_mask[:, :, None, None]
        # attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t
        #
        # attn_prob = tf.nn.softmax(attn_score, 1)
        # tf.add_to_collection('attentions', attn_prob)
        # attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        #
        # attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)

        ###########################1
        AR = spcific_nets_simple(side[0, :, :], is_training=is_training, d_original=12 * 4 * 4, d_inner=64, dropout=0,
                                 initializer=kernel_initializer, scope='side_ar1')
        AR = tf.reshape(AR, [bsz, qlen, n_head, d_head])
        # AR = tf.reshape(side[0, :, :], [bsz, qlen, n_head, d_head])
        AR_ATT = tf.concat(
            [tf.slice(tf.pad(tf.expand_dims(AR, axis=1), [[0, 0], [0, 0], [0, qlen_input - i - 1],[0,0],[0,0]]),
                      [0, 0, qlen_input - i - 1,0,0],
                      [-1, 1, qlen_input,-1,-1]) for i in
             range(qlen_input)], axis=1)#[bllnd]

        ############################2
        # AR = spcific_nets_simple(side[0,:,:],is_training=is_training,d_original=12*4*4,d_inner=64,dropout=0,initializer=kernel_initializer,scope='side_ar2')
        # AR = tf.reshape(AR, [bsz, qlen, n_head, d_head])
        # AR = tf.reshape(side[0, :, :], [bsz, qlen, n_head, d_head])
        # AR_ATT2=AR
        ############################

        # AR = tf.get_variable('AR', [qlen_input],initializer=kernel_initializer)

        # head_all = tf.concat([tf.transpose(batch_inp[:, :, -1], [1, 0]), tf.transpose(c_m_batch_inp[:, :, -1], [1, 0]),
        #                       tf.transpose(m_p_batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # att_head = spcific_nets_simple(head_all, is_training=is_training, d_original=3, d_inner=32, dropout=dropout,
        #                                initializer=initializer, scope='head_1')
        # att_score = tf.nn.softmax(att_head)


        # AR = spcific_nets_simple(side[0,:,:],is_training=is_training,d_original=qlen_input,d_inner=32,dropout=dropout,initializer=kernel_initializer,scope='side_ar')
        # # AR=side[0,:,:]
        # print(side)#(12, ?, 32)
        # print(AR)
        # print(w)#(12,?,16)
        # AR_ATT = tf.concat(
        #     [tf.slice(tf.pad(tf.expand_dims(AR, axis=1), [[0,0],[0, 0], [0, qlen_input - i - 1]]), [0,0, qlen_input - i - 1],
        #               [-1,1, qlen_input]) for i in
        #      range(qlen_input)], axis=1)
        #
        # print(AR_ATT)#bll
        # print(side)
        # print(w)

        # AR=tf.nn.l2_normalize(AR)
        # AR_2=tf.nn.softmax(AR)
        # AR_ATT = tf.concat(
        #     [tf.slice(tf.pad(tf.expand_dims(AR, axis=0), [[0, 0], [0, qlen_input - i - 1]]), [0, qlen_input - i - 1], [1, qlen_input]) for i in
        #      range(qlen_input)], axis=0)

        qk_w = w
        if cp_flag:
            qk_w = tf.concat([qk_w, c_m, m_p], axis=2)
        if side_flag:
            qk_w = tf.concat([qk_w, side], axis=2)

        w_head_v = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='v')
        w_heads_qk = tf.layers.dense(w, 2 * n_head * d_head, use_bias=False,
                                     kernel_initializer=kernel_initializer, name='qk')

        #######################################
        # c_m=positionwise_FF(c_m,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='c_m_1',is_training=is_training)
        # m_p=positionwise_FF(m_p,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='m_p_1',is_training=is_training)
        # side=positionwise_FF(side,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='side_1',is_training=is_training)
        # qk_w = tf.layers.dense(qk_w, d_model,kernel_initializer=kernel_initializer, name='qk_r')
        # qk_w = tf.concat([w, qk_w], axis=2)
        #
        # w_heads_qk_2 = tf.layers.dense(qk_w, 2 * n_head * d_head, use_bias=False,
        #                                kernel_initializer=kernel_initializer, name='qk_2')
        ########################################

        # w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        # if head_attention_flag:
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q, w_head_k = tf.split(w_heads_qk, 2, -1)

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [qlen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [qlen, bsz, n_head, d_head])

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k)

        ########1
        attn_score_tpe=tf.einsum('ibnd,bijnd->ijbn', w_head_q, AR_ATT)
        ori_att=attn_score[:,:,0,0]
        tpe_att=attn_score_tpe[:,:,0,0]
        # attn_score=attn_score+attn_score_tpe
        #########2
        # attn_score_tpe = tf.einsum('ibnd,bjnd->ijbn', w_head_q, AR_ATT)
        # attn_score = attn_score + attn_score_tpe


        attn_score = attn_score * scale
        # attn_score = tf.einsum('ijbn,ij->ijbn',attn_score,AR_ATT)
        # attn_score = tf.einsum('ijbn,bij->ijbn',attn_score,AR_ATT)
        # attn_score = attn_score+tf.expand_dims(tf.transpose(AR_ATT,[1,2,0]),axis=3)
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        #############1
        attn_score_tpe = attn_score_tpe * (1 - attn_mask_t) - 1e30 * attn_mask_t
        attn_prob_tpe = tf.nn.softmax(attn_score_tpe, 1)
        attn_prob=(attn_prob+attn_prob_tpe)/2.0

        tf.add_to_collection('attentions', attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        ###########2.1
        # attn_vec_tpe = tf.einsum('ijbn,bjnd->ibnd', attn_prob, AR_ATT2)
        # attn_vec=attn_vec+attn_vec_tpe

        if head_attention_flag:
            cross_w = tf.transpose(w, [1, 0, 2])
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(d_model) for i in range(2)])
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, cross_w, dtype=tf.float32)
            output_rnn = output_rnn[:, -1, :]
            cross_head_attention = tf.layers.dense(output_rnn, n_head * n_head, use_bias=False,
                                                   kernel_initializer=kernel_initializer, name='cross_att')
            cross_head_attention = tf.reshape(cross_head_attention, [bsz, n_head, n_head])
            cross_head_attention = tf.nn.softmax(cross_head_attention)
            attn_vec = tf.einsum('ibnd,bmn->ibmd', attn_vec, cross_head_attention)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output,c_m,m_p,side,[ori_att,tpe_att]


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)
    return x

def rel_multihead_attn(w, r, r_w_bias, r_r_bias, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
                                  kernel_initializer=kernel_initializer, name='qkv')
        r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='r')

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        # w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        # attn_mask_t = attn_mask[:, :, None, None]
        # attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        tf.add_to_collection('attentions',attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output,[AC[:,:,0,0],BD[:,:,0,0],attn_prob[:,:,0,0]]

def multihead_attn_tpe4_XL(w, w_o, c_m, m_p, side, attn_mask, d_model,
                   n_head, d_head, dropout, dropatt, is_training,
                   kernel_initializer, head_attention_flag=False, side_flag=True, cp_flag=True, qlen_input=6,scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]
        #############TPE##################

        #
        # qlen = tf.shape(w)[0]
        # rlen = tf.shape(r)[0]
        # bsz = tf.shape(w)[1]
        #
        # cat = tf.concat([mems, w],
        #                 0) if mems is not None and mems.shape.ndims > 1 else w
        # w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        #
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        # w_head_q = w_head_q[-qlen:]
        #
        # klen = tf.shape(w_head_k)[0]
        #
        # w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        # w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        # w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])
        #
        # r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])
        #
        # rw_head_q = w_head_q + r_w_bias
        # rr_head_q = w_head_q + r_r_bias
        #
        # AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        # BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        # BD = rel_shift(BD)
        #
        # attn_score = (AC + BD) * scale
        # attn_mask_t = attn_mask[:, :, None, None]
        # attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t
        #
        # attn_prob = tf.nn.softmax(attn_score, 1)
        # tf.add_to_collection('attentions', attn_prob)
        # attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        #
        # attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)

        ###########################1

        ###########################1
        # AR = spcific_nets_simple(side, is_training=is_training, d_original=qlen_input * 4 * 4, d_inner=64, dropout=0.1,
        #                          initializer=kernel_initializer,layer=2, scope='side_ar1')
        # AR = tf.reshape(AR, [bsz, qlen, n_head, d_head])

        # AR = tf.reshape(side, [bsz, qlen, n_head, d_head])
        # AR_ATT = tf.concat(
        #     [tf.slice(tf.pad(tf.expand_dims(AR, axis=1), [[0, 0], [0, 0], [0, qlen_input - i - 1], [0, 0], [0, 0]]),
        #               [0, 0, qlen_input - i - 1, 0, 0],
        #               [-1, 1, qlen_input, -1, -1]) for i in
        #      range(qlen_input)], axis=1)  # [bllnd]
        ########################
        AR = tf.reshape(side, [bsz, 2*qlen, n_head, d_head])
        # AR = tf.tile(tf.get_variable('AR',shape=[1,2*qlen_input, n_head, d_head],initializer=kernel_initializer),[32,1,1,1])
        AR_ATT = tf.concat(
            [tf.slice(tf.expand_dims(AR, axis=1),
                      [0, 0, qlen_input - i - 1, 0, 0],
                      [-1, 1, qlen_input, -1, -1]) for i in
             range(qlen_input)], axis=1)  # [bllnd]
        ########################


        w_head_v = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='v')
        # w_heads_qk = tf.layers.dense(w, 2 * n_head * d_head, use_bias=False,
        #                              kernel_initializer=kernel_initializer, name='qk')
        w_head_k = tf.layers.dense(w_o, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='k')
        w_head_q = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='q')

        #######################################
        # c_m=positionwise_FF(c_m,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='c_m_1',is_training=is_training)
        # m_p=positionwise_FF(m_p,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='m_p_1',is_training=is_training)
        # side=positionwise_FF(side,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='side_1',is_training=is_training)
        # qk_w = tf.layers.dense(qk_w, d_model,kernel_initializer=kernel_initializer, name='qk_r')
        # qk_w = tf.concat([w, qk_w], axis=2)
        #
        # w_heads_qk_2 = tf.layers.dense(qk_w, 2 * n_head * d_head, use_bias=False,
        #                                kernel_initializer=kernel_initializer, name='qk_2')
        ########################################

        # w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        # if head_attention_flag:
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        # w_head_q, w_head_k = tf.split(w_heads_qk, 2, -1)

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [qlen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [qlen, bsz, n_head, d_head])

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k)

        attn_score = attn_score * scale
        # attn_mask_t = attn_mask[:, :, None, None]
        # attn_score = attn_score * (1 - attn_mask_t) - 1e10 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)

        ############################
        # print(w_head_q)
        # print(AR_ATT)
        # exit()
        attn_score_tpe = tf.einsum('ibnd,bijnd->ijbn', w_head_q, AR_ATT)*scale
        # attn_score_tpe = attn_score_tpe - 1e10 * attn_mask_t
        # attn_prob_tpe = tf.nn.softmax(attn_score_tpe, 1)
        # attn_prob_tpe = tf.nn.softmax(tf.nn.l2_normalize(attn_score_tpe,axis=1)+tf.nn.l2_normalize(attn_score,axis=1), 1)
        attn_prob_tpe = tf.nn.softmax(attn_score_tpe+attn_score, 1)

        ori_att = attn_prob[:, :, 0, 0]
        # ori_att = attn_score[:, :, 0, 0]
        # ori_att = 1e30 * attn_mask_t[:, :, 0, 0]
        # tpe_att = attn_score_tpe[:, :, 0, 0]
        tpe_att = attn_prob_tpe[:, :, 0, 0]

        alph=0.5
        # attn_prob=((1-alph)*attn_prob_tpe+alph*attn_prob)
        attn_prob=attn_prob_tpe
        # attn_prob=attn_prob
        # attn_prob=((1-alph)*attn_score_tpe+alph*attn_prob)
        #####################################

        tf.add_to_collection('attentions', attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        #####################################
        # rpe=tf.reshape(side, [bsz, 2*qlen, n_head, d_head])
        # rpe = tf.tile(tf.get_variable('rpe', shape=[1, 2 * qlen_input, n_head, d_head], initializer=kernel_initializer),
        #              [32, 1, 1, 1])
        # w_head_v_rpe = tf.concat(
        #     [tf.slice(tf.expand_dims(rpe, axis=1),
        #               [0, 0, qlen_input - i - 1, 0, 0],
        #               [-1, 1, qlen_input, -1, -1]) for i in
        #      range(qlen_input)], axis=1)  # [bllnd]
        # w_head_v_rpe=tf.transpose(w_head_v_rpe,[1,2,0,3,4])
        # attn_vec_rpe=tf.einsum('ijbn,ijbnd->ibnd', attn_prob, w_head_v_rpe)
        # attn_vec=attn_vec+attn_vec_rpe
        #######################################
        if head_attention_flag:
            cross_w = tf.transpose(w, [1, 0, 2])
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(d_model) for i in range(2)])
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, cross_w, dtype=tf.float32)
            output_rnn = output_rnn[:, -1, :]
            cross_head_attention = tf.layers.dense(output_rnn, n_head * n_head, use_bias=False,
                                                   kernel_initializer=kernel_initializer, name='cross_att')
            cross_head_attention = tf.reshape(cross_head_attention, [bsz, n_head, n_head])
            cross_head_attention = tf.nn.softmax(cross_head_attention)
            attn_vec = tf.einsum('ibnd,bmn->ibmd', attn_vec, cross_head_attention)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output,c_m,m_p,side,[ori_att,tpe_att]


def multihead_attn_tpe3(w, w_o, c_m, m_p, side, attn_mask, d_model,
                   n_head, d_head, dropout, dropatt, is_training,
                   kernel_initializer, head_attention_flag=False, side_flag=True, cp_flag=True, qlen_input=6,scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]
        ###########################1
        # AR = spcific_nets_simple(side, is_training=is_training, d_original=qlen_input * 4 * 4, d_inner=64, dropout=0.1,
        #                          initializer=kernel_initializer,layer=2, scope='side_ar1')
        # AR = tf.reshape(AR, [bsz, qlen, n_head, d_head])

        # AR = tf.reshape(side, [bsz, qlen, n_head, d_head])
        # AR_ATT = tf.concat(
        #     [tf.slice(tf.pad(tf.expand_dims(AR, axis=1), [[0, 0], [0, 0], [0, qlen_input - i - 1], [0, 0], [0, 0]]),
        #               [0, 0, qlen_input - i - 1, 0, 0],
        #               [-1, 1, qlen_input, -1, -1]) for i in
        #      range(qlen_input)], axis=1)  # [bllnd]
        ########################
        AR = tf.reshape(side, [bsz, 2*qlen, n_head, d_head])
        # AR = tf.tile(tf.get_variable('AR',shape=[1,2*qlen_input, n_head, d_head],initializer=kernel_initializer),[32,1,1,1])
        AR_ATT = tf.concat(
            [tf.slice(tf.expand_dims(AR, axis=1),
                      [0, 0, qlen_input - i - 1, 0, 0],
                      [-1, 1, qlen_input, -1, -1]) for i in
             range(qlen_input)], axis=1)  # [bllnd]
        ########################



        # w_heads_qk = tf.layers.dense(w, 2 * n_head * d_head, use_bias=False,
        #                              kernel_initializer=kernel_initializer, name='qk')
        w_head_k = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='k')

        w_head_v = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                    kernel_initializer = kernel_initializer, name = 'v')
        # w_head_q_o = tf.layers.dense(w, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='q_o')
        # w_head_k = tf.layers.dense(w, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='k')
        w_head_q = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='q')

        # w_heads_qkv = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
        #                              kernel_initializer=kernel_initializer, name='qkv')

        #######################################
        # c_m=positionwise_FF(c_m,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='c_m_1',is_training=is_training)
        # m_p=positionwise_FF(m_p,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='m_p_1',is_training=is_training)
        # side=positionwise_FF(side,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='side_1',is_training=is_training)
        # qk_w = tf.layers.dense(qk_w, d_model,kernel_initializer=kernel_initializer, name='qk_r')
        # qk_w = tf.concat([w, qk_w], axis=2)
        #
        # w_heads_qk_2 = tf.layers.dense(qk_w, 2 * n_head * d_head, use_bias=False,
        #                                kernel_initializer=kernel_initializer, name='qk_2')
        ########################################

        # w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        # if head_attention_flag:
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads_qkv, 3, -1)
        # w_head_q, w_head_k = tf.split(w_heads_qk, 2, -1)

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [qlen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [qlen, bsz, n_head, d_head])
        # w_head_q_o = tf.reshape(w_head_q_o, [qlen, bsz, n_head, d_head])

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k)

        attn_score = attn_score * scale
        # attn_mask_t = attn_mask[:, :, None, None]
        # attn_score = attn_score * (1 - attn_mask_t) - 1e10 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)

        ############################
        # print(w_head_q)
        # print(AR_ATT)
        # exit()
        attn_score_tpe = tf.einsum('ibnd,bijnd->ijbn', w_head_q, AR_ATT)*scale
        # attn_score_tpe = tf.einsum('ibnd,bijnd->ijbn', w_head_q_o, AR_ATT)*scale
        # attn_score_tpe = attn_score_tpe - 1e10 * attn_mask_t
        attn_prob_tpe = tf.nn.softmax(attn_score_tpe, 1)
        # attn_prob_tpe = tf.nn.softmax(tf.nn.l2_normalize(attn_score_tpe,axis=1)+tf.nn.l2_normalize(attn_score,axis=1), 1)
        # attn_prob_tpe = tf.nn.softmax(attn_score_tpe+attn_score, 1)

        # ori_att = attn_prob[:, :, 0, 0]
        # ori_att = attn_score[:, :, 0, 0]
        ori_att = attn_prob
        # ori_att = 1e30 * attn_mask_t[:, :, 0, 0]
        # tpe_att = attn_score_tpe[:, :, 0, 0]
        tpe_att = attn_prob_tpe
        # tpe_att = attn_prob_tpe[:, :, 0, 0]

        alph=0.5
        # alph=0.75
        attn_prob=((1-alph)*attn_prob_tpe+alph*attn_prob)
        # attn_prob=attn_prob_tpe
        # attn_prob=attn_prob
        # attn_prob=((1-alph)*attn_score_tpe+alph*attn_prob)
        #####################################

        tf.add_to_collection('attentions', attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        #####################################
        # rpe=tf.reshape(side, [bsz, 2*qlen, n_head, d_head])
        # rpe = tf.tile(tf.get_variable('rpe', shape=[1, 2 * qlen_input, n_head, d_head], initializer=kernel_initializer),
        #              [32, 1, 1, 1])
        # w_head_v_rpe = tf.concat(
        #     [tf.slice(tf.expand_dims(rpe, axis=1),
        #               [0, 0, qlen_input - i - 1, 0, 0],
        #               [-1, 1, qlen_input, -1, -1]) for i in
        #      range(qlen_input)], axis=1)  # [bllnd]
        # w_head_v_rpe=tf.transpose(w_head_v_rpe,[1,2,0,3,4])
        # attn_vec_rpe=tf.einsum('ijbn,ijbnd->ibnd', attn_prob, w_head_v_rpe)
        # attn_vec=attn_vec+attn_vec_rpe
        #######################################
        if head_attention_flag:
            cross_w = tf.transpose(w, [1, 0, 2])
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(d_model) for i in range(2)])
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, cross_w, dtype=tf.float32)
            output_rnn = output_rnn[:, -1, :]
            cross_head_attention = tf.layers.dense(output_rnn, n_head * n_head, use_bias=False,
                                                   kernel_initializer=kernel_initializer, name='cross_att')
            cross_head_attention = tf.reshape(cross_head_attention, [bsz, n_head, n_head])
            cross_head_attention = tf.nn.softmax(cross_head_attention)
            attn_vec = tf.einsum('ibnd,bmn->ibmd', attn_vec, cross_head_attention)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output,c_m,m_p,side,[ori_att,tpe_att]

def multihead_attn_tpe2(w, w_o, c_m, m_p, side, attn_mask, d_model,
                   n_head, d_head, dropout, dropatt, is_training,
                   kernel_initializer, head_attention_flag=False, side_flag=True, cp_flag=True, qlen_input=6,scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]
        ###########################1
        # AR = spcific_nets_simple(side, is_training=is_training, d_original=qlen_input * 4 * 4, d_inner=64, dropout=0.1,
        #                          initializer=kernel_initializer,layer=2, scope='side_ar1')
        # AR = tf.reshape(AR, [bsz, qlen, n_head, d_head])

        # AR = tf.reshape(side, [bsz, qlen, n_head, d_head])
        # AR_ATT = tf.concat(
        #     [tf.slice(tf.pad(tf.expand_dims(AR, axis=1), [[0, 0], [0, 0], [0, qlen_input - i - 1], [0, 0], [0, 0]]),
        #               [0, 0, qlen_input - i - 1, 0, 0],
        #               [-1, 1, qlen_input, -1, -1]) for i in
        #      range(qlen_input)], axis=1)  # [bllnd]
        ########################
        AR = tf.reshape(side, [bsz, 2*qlen, n_head, d_head])
        # AR = tf.tile(tf.get_variable('AR',shape=[1,2*qlen_input, n_head, d_head],initializer=kernel_initializer),[32,1,1,1])
        AR_ATT = tf.concat(
            [tf.slice(tf.expand_dims(AR, axis=1),
                      [0, 0, qlen_input - i - 1, 0, 0],
                      [-1, 1, qlen_input, -1, -1]) for i in
             range(qlen_input)], axis=1)  # [bllnd]
        ########################


        w_head_v = tf.layers.dense(w, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='v')
        w_heads_qk = tf.layers.dense(w, 2 * n_head * d_head, use_bias=False,
                                     kernel_initializer=kernel_initializer, name='qk')

        #######################################
        # c_m=positionwise_FF(c_m,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='c_m_1',is_training=is_training)
        # m_p=positionwise_FF(m_p,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='m_p_1',is_training=is_training)
        # side=positionwise_FF(side,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='side_1',is_training=is_training)
        # qk_w = tf.layers.dense(qk_w, d_model,kernel_initializer=kernel_initializer, name='qk_r')
        # qk_w = tf.concat([w, qk_w], axis=2)
        #
        # w_heads_qk_2 = tf.layers.dense(qk_w, 2 * n_head * d_head, use_bias=False,
        #                                kernel_initializer=kernel_initializer, name='qk_2')
        ########################################

        # w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        # if head_attention_flag:
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q, w_head_k = tf.split(w_heads_qk, 2, -1)

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [qlen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [qlen, bsz, n_head, d_head])

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k)

        attn_score = attn_score * scale
        # attn_mask_t = attn_mask[:, :, None, None]
        # attn_score = attn_score * (1 - attn_mask_t) - 1e10 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)

        ############################
        # print(w_head_q)
        # print(AR_ATT)
        # exit()
        attn_score_tpe = tf.einsum('ibnd,bijnd->ijbn', w_head_q, AR_ATT)*scale
        # attn_score_tpe = attn_score_tpe - 1e10 * attn_mask_t
        attn_prob_tpe = tf.nn.softmax(attn_score_tpe, 1)
        # attn_prob_tpe = tf.nn.softmax(tf.nn.l2_normalize(attn_score_tpe,axis=1)+tf.nn.l2_normalize(attn_score,axis=1), 1)
        # attn_prob_tpe = tf.nn.softmax(attn_score_tpe+attn_score, 1)

        ori_att = attn_prob[:, :, 0, 0]
        # ori_att = attn_score[:, :, 0, 0]
        # ori_att = 1e30 * attn_mask_t[:, :, 0, 0]
        # tpe_att = attn_score_tpe[:, :, 0, 0]
        tpe_att = attn_prob_tpe[:, :, 0, 0]

        alph=0.5
        attn_prob=((1-alph)*attn_prob_tpe+alph*attn_prob)
        # attn_prob=attn_prob_tpe
        # attn_prob=attn_prob
        # attn_prob=((1-alph)*attn_score_tpe+alph*attn_prob)
        #####################################

        tf.add_to_collection('attentions', attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        #####################################
        # rpe=tf.reshape(side, [bsz, 2*qlen, n_head, d_head])
        # rpe = tf.tile(tf.get_variable('rpe', shape=[1, 2 * qlen_input, n_head, d_head], initializer=kernel_initializer),
        #              [32, 1, 1, 1])
        # w_head_v_rpe = tf.concat(
        #     [tf.slice(tf.expand_dims(rpe, axis=1),
        #               [0, 0, qlen_input - i - 1, 0, 0],
        #               [-1, 1, qlen_input, -1, -1]) for i in
        #      range(qlen_input)], axis=1)  # [bllnd]
        # w_head_v_rpe=tf.transpose(w_head_v_rpe,[1,2,0,3,4])
        # attn_vec_rpe=tf.einsum('ijbn,ijbnd->ibnd', attn_prob, w_head_v_rpe)
        # attn_vec=attn_vec+attn_vec_rpe
        #######################################
        if head_attention_flag:
            cross_w = tf.transpose(w, [1, 0, 2])
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(d_model) for i in range(2)])
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, cross_w, dtype=tf.float32)
            output_rnn = output_rnn[:, -1, :]
            cross_head_attention = tf.layers.dense(output_rnn, n_head * n_head, use_bias=False,
                                                   kernel_initializer=kernel_initializer, name='cross_att')
            cross_head_attention = tf.reshape(cross_head_attention, [bsz, n_head, n_head])
            cross_head_attention = tf.nn.softmax(cross_head_attention)
            attn_vec = tf.einsum('ibnd,bmn->ibmd', attn_vec, cross_head_attention)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output,c_m,m_p,side,[ori_att,tpe_att]


def multihead_attn_new(w, c_m, m_p, side, attn_mask, d_model,
                   n_head, d_head, dropout, dropatt, is_training,
                   kernel_initializer, head_attention_flag=False, side_flag=True, cp_flag=True, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]
        qk_w = w
        if cp_flag:
            qk_w = tf.concat([qk_w, c_m, m_p], axis=2)
        if side_flag:
            qk_w = tf.concat([qk_w, side], axis=2)

        w_head_v = tf.layers.dense(w, 2*n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='v')
        w_heads_qk = tf.layers.dense(w, 2 * n_head * d_head, use_bias=False,
                                     kernel_initializer=kernel_initializer, name='qk')

        #######################################
        new_c_m=positionwise_FF(c_m,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='c_m_1',is_training=is_training)
        new_m_p=positionwise_FF(m_p,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='m_p_1',is_training=is_training)
        new_side=positionwise_FF(side,d_model=d_model,d_inner=2*d_model,dropout=dropout,kernel_initializer=kernel_initializer,scope='side_1',is_training=is_training)
        qk_w = tf.layers.dense(qk_w, d_model,kernel_initializer=kernel_initializer, name='qk_r')
        qk_w = tf.concat([w, qk_w], axis=2)

        w_heads_qk_2 = tf.layers.dense(qk_w, 2 * n_head * d_head, use_bias=False,
                                       kernel_initializer=kernel_initializer, name='qk_2')
        w_heads_qk = tf.concat([w_heads_qk, w_heads_qk_2], axis=2)
        ########################################

        # w_heads = tf.layers.dense(w, 3 * n_head * d_head, use_bias=False,
        #                           kernel_initializer=kernel_initializer, name='qkv')
        # r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
        #                            kernel_initializer=kernel_initializer, name='r')
        # if head_attention_flag:
        # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q, w_head_k = tf.split(w_heads_qk, 2, -1)

        # w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_q = tf.reshape(w_head_q, [qlen, bsz, 2*n_head, d_head])
        # w_head_k = tf.reshape(w_head_k, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [qlen, bsz, 2*n_head, d_head])
        # w_head_v = tf.reshape(w_head_v, [qlen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [qlen, bsz, 2*n_head, d_head])

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k)

        attn_score = attn_score * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        tf.add_to_collection('attentions', attn_prob)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        if head_attention_flag:
            cross_w = tf.transpose(w, [1, 0, 2])
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(d_model) for i in range(2)])
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, cross_w, dtype=tf.float32)
            output_rnn = output_rnn[:, -1, :]
            cross_head_attention = tf.layers.dense(output_rnn, n_head * n_head, use_bias=False,
                                                   kernel_initializer=kernel_initializer, name='cross_att')
            cross_head_attention = tf.reshape(cross_head_attention, [bsz, n_head, n_head])
            cross_head_attention = tf.nn.softmax(cross_head_attention)
            attn_vec = tf.einsum('ibnd,bmn->ibmd', attn_vec, cross_head_attention)
        size_t = tf.shape(attn_vec)
        # attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], 2*n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
        # print(attn_out)
        # print(w)
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output,new_c_m,new_m_p,new_side


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True):
    with tf.variable_scope(scope):
        output = tf.layers.dense(inp, d_inner, activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_1')
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_1')
        output = tf.layers.dense(output, d_model,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_2')
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_2')
        output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
    return output


def Transformer_nets(dec_inp, is_training, d_step, d_original, d_model, n_layer, n_head, d_head, d_inner, dropout,
                     dropatt, initializer, init_embeding=True,
                     mask_same_length=False, scope='transformer_nets'):
    with tf.variable_scope(scope):
        if init_embeding:
            dec_inp = tf.layers.dense(dec_inp, d_model, use_bias=False, kernel_initializer=initializer,
                                      name='embedding')
            # d_model = tf.shape(dec_inp)[-1]
        else:
            d_model = d_original

        qlen = tf.shape(dec_inp)[0]

        embeddings = dec_inp

        attn_mask = _create_mask(qlen, 0, mask_same_length)

        pos_seq = tf.range(qlen - 1, -1, -1.0)
        inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        # inv_freq=[1/(10000.0**(i/d_model)) for i in range(0,d_model,2)]
        inv_freq = tf.constant(inv_freq)
        # inv_freq = 1.0 / (10000.0 ** (tf.range(0, d_model, 2.0,dtype=tf.float32) / (tf.cast(d_model,tf.float32))))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = tf.layers.dropout(embeddings, dropout, training=is_training)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

        output = output + pos_emb

        for i in range(3):
            with tf.variable_scope('layer_{}'.format(i)):
                output = multihead_attn(
                    w=output,
                    attn_mask=attn_mask,
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer)
                output = positionwise_FF(
                    inp=output,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)

        # output = tf.layers.dropout(output, dropout, training=is_training)
        output = spcific_nets_simple(output, is_training=is_training, d_original=5,
                                         d_inner=32, dropout=dropout, initializer=initializer,
                                         scope='com_final_spcific')
        print(output)
        print('-------------------')

        # output = output
        return output


def dense_interpolation(input, t_dense, m_dense):
    # T = tf.shape(input)[0]
    T = t_dense
    u = []
    for t in range(T):
        s = t * (m_dense / T)
        row = []
        for m in range(m_dense):
            w = pow(1 - (abs(s - m) / m_dense), 2)
            row.append(w)
        u.append(row)
    u = tf.constant(u)
    output = tf.einsum('im,ibd->mbd', u, input)
    return output


def flat(input, only_last_one=False):
    if only_last_one:
        output = input[-1]
    else:
        output = tf.transpose(input, [1, 0, 2])
        output = tf.layers.flatten(output)
        # output = tf.reshape(output, [-1, tf.shape(output)[1]*tf.shape(output)[2]])
    return output


def spcific_nets_simple(dec_inp, is_training, d_original, d_inner, dropout, initializer, layer=1,scope='spcific_nets'):
    with tf.variable_scope(scope):
        output=dec_inp
        for i in range(layer):
            output = tf.layers.dense(output, d_inner, activation=tf.nn.relu, kernel_initializer=initializer,
                                    name='layer_{}'.format(i+1))
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_1')
        output = tf.layers.dense(output, d_original, kernel_initializer=initializer, name='layer_f')
        # output = tf.layers.dense(output, d_original, activation=tf.nn.relu,kernel_initializer=initializer, name='layer_f')

        return output


def spcific_nets_simple_dense(dec_inp, is_training, len_time, m_dense, d_original, d_inner, dropout, initializer,
                              only_last_one=False, scope='spcific_nets'):
    with tf.variable_scope(scope):
        output = dense_interpolation(dec_inp, t_dense=len_time, m_dense=m_dense)
        output = flat(output, only_last_one=only_last_one)
        output = tf.layers.dense(output, d_inner, activation=tf.nn.relu, kernel_initializer=initializer, name='layer_1')
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_1')
        output = tf.layers.dense(output, d_original, kernel_initializer=initializer, name='layer_2')
        return output


def spcific_net_ff_dense(inp, d_model, d_inner, dropout, initializer, len_time, m_dense, is_training,
                         only_last_one=True, scope='ff_dense'):
    with tf.variable_scope(scope):
        inp = positionwise_FF(inp=inp,
                                 d_model=d_model,
                                 d_inner=d_inner,
                                 dropout=dropout,
                                 kernel_initializer=initializer,
                                 is_training=is_training,
                                 scope='ff')
        output = dense_interpolation(inp, t_dense=len_time, m_dense=m_dense)
        output = flat(output, only_last_one=only_last_one)
        return output


def spcific_net_attention(inp, c_m, m_p, side, attn_mask, n_head, d_head, dropatt, d_model, d_inner, dropout,
                          initializer, len_time,
                          m_dense, is_training,
                          only_last_one=True, head_attention_flag=False, side_flag=False, cp_flag=False,
                          scope='ff_dense'):
    with tf.variable_scope(scope):
        output = multihead_attn(
            w=inp,
            c_m=c_m,
            m_p=m_p,
            side=side,
            attn_mask=attn_mask,
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            is_training=is_training,
            kernel_initializer=initializer,
            head_attention_flag=head_attention_flag, side_flag=side_flag, cp_flag=cp_flag)

        output = positionwise_FF(inp=output,
                                 d_model=d_model,
                                 d_inner=d_inner,
                                 dropout=dropout,
                                 kernel_initializer=initializer,
                                 is_training=is_training,
                                 scope='ff')
        output = dense_interpolation(output, t_dense=len_time, m_dense=m_dense)
        output = flat(output, only_last_one=only_last_one)
        return output


def pre_lstm(input, size=5, layers=3, scope='pre_lstm'):
    with tf.variable_scope(scope):
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for i in range(layers)])
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        return output_rnn




def TDAN(classify_num, batch_inp, c_m_batch_inp, m_p_batch_inp, side_inform, past_len, future_len, is_training,
                  d_step,
                  d_model,
                  n_layer, n_head, d_head, d_inner, d_spcific_inner, d_input_inner, d_side_inner,
                  d_side_out, dropout, dropatt, m_dense, initializer, init_embeding=True,
                  mask_same_length=False, side_flag=False, head_attention_flag=False, side_input=True, cp_input=True,
                  Bi_attention=True, scope='level3_nets'):
    with tf.variable_scope(scope):
        #########################################
        ori_inp=batch_inp
        head_all=tf.concat([tf.transpose(batch_inp[:,:,-1],[1,0]),tf.transpose(c_m_batch_inp[:,:,-1],[1,0]),tf.transpose(m_p_batch_inp[:,:,-1],[1,0]),side_inform],axis=1)
        att_head=spcific_nets_simple(head_all,is_training=is_training,d_original=3,d_inner=32,dropout=dropout,initializer=initializer,scope='head_1')
        att_score=tf.nn.softmax(att_head)
        MIX_att = att_score
        head_all_v=tf.concat([tf.expand_dims(batch_inp,-1),tf.expand_dims(c_m_batch_inp,-1),tf.expand_dims(m_p_batch_inp,-1)],axis=3)
        batch_inp=tf.einsum('lbdn,bn->lbd',head_all_v,att_score)
        # MIX_att = tf.constant(0)
        ######################################
        side_inform = tf.concat(
            [tf.transpose(batch_inp[:, :, -1], [1, 0]), tf.transpose(c_m_batch_inp[:, :, -1], [1, 0]),
             tf.transpose(m_p_batch_inp[:, :, -1], [1, 0]), side_inform+1e-20], axis=1)

        # side_inform = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
                                                    d_inner=d_side_inner, dropout=0.1, initializer=initializer,
                                                    layer=1,scope='side_imbedding')
        # context_pe=spcific_nets_simple(side_inform, is_training=is_training, d_original=d_model*past_len,
        #                                             d_inner=d_side_inner, dropout=0.1, initializer=initializer,
        #                                             scope='pe_imbedding')
        # context_pe=tf.reshape(context_pe,[-1,past_len,d_model])
        # context_pe=tf.transpose(context_pe,[1,0,2])
        ##########################################
        # pos_seq_past = tf.range(past_len - 1, -1, -1.0)
        # inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        # inv_freq = tf.constant(inv_freq)
        # pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        # pos_emb_past=tf.reshape(tf.transpose(pos_emb_past,[1,0,2]),[1,past_len*d_model])
        # pos_emb_past=tf.tile(pos_emb_past,[32,1])
        # side_inform_pe = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform,pos_emb_past], axis=1)
        # # side_inform = tf.concat(
        # #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # context_pe = spcific_nets_simple(side_inform_pe, is_training=is_training, d_original=d_model * past_len,
        #                                  d_inner=d_side_inner*2, dropout=0.1, initializer=initializer,
        #                                  scope='pe_imbedding')
        # context_pe = tf.reshape(context_pe, [-1, past_len, d_model])
        # context_pe = tf.transpose(context_pe, [1, 0, 2])
        ###########################################


        dec_inp_past = batch_inp

        c_m_past = c_m_batch_inp

        m_p_past = m_p_batch_inp

        # if side_input:
        # side_inform_imbedding = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        #                                             d_inner=d_side_inner, dropout=dropout, initializer=initializer,
        #                                             scope='side_imbedding')
        # side_inform_imbedding = tf.tile(tf.expand_dims(side_inform_imbedding, 0), [past_len, 1, 1])
        # print(side_inform_imbedding)
        # print(dec_inp_past)
        # exit()
        # dec_inp_past = tf.concat([dec_inp_past, side_inform_imbedding], axis=2)
        # dec_inp_future = tf.concat([dec_inp_future, side_inform_imbedding], axis=2)
        if init_embeding:
            c_m_past = spcific_nets_simple(c_m_past, is_training=is_training, d_original=d_model,
                                           d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                           scope='embedding_past_c_m')
            m_p_past = spcific_nets_simple(m_p_past, is_training=is_training, d_original=d_model,
                                           d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                           scope='embedding_past_m_p')
        if init_embeding:
            # dec_inp_past = spcific_nets_simple(dec_inp_past, is_training=is_training, d_original=d_model,
            #                                    d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
            #                                    scope='imbedding_past')
            dec_inp_past = tf.layers.dense(dec_inp_past, d_model, use_bias=False, kernel_initializer=initializer,
                                      name='embedding')
        else:
            d_model = d_step

        qlen_past = tf.shape(dec_inp_past)[0]

        embeddings_past = dec_inp_past

        attn_mask_past = _create_mask(qlen_past, 0, mask_same_length)

        pos_seq_past = tf.range(qlen_past - 1, -1, -1.0)
        inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        inv_freq = tf.constant(inv_freq)
        pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        print(pos_emb_past)
        # print(context_pe)
        # exit()

        output_past = tf.layers.dropout(embeddings_past, dropout, training=is_training)

        c_m_past = tf.layers.dropout(c_m_past, dropout, training=is_training)
        m_p_past = tf.layers.dropout(m_p_past, dropout, training=is_training)

        pos_emb_past = tf.layers.dropout(pos_emb_past, dropout, training=is_training)

        output_past_ori=output_past
        output_past = output_past + pos_emb_past
        # output_past = output_past + context_pe
        # output_past = output_past + tf.nn.l2_normalize(context_pe,axis=0)
        # output_past = tf.concat([output_past,context_pe],axis=2)
        # output_past = tf.concat([output_past,tf.nn.l2_normalize(context_pe,axis=0)],axis=2)
        # print(output_past)
        # exit()
        #
        # c_m_past = c_m_past + pos_emb_past
        # m_p_past = m_p_past + pos_emb_past

        # r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
        #                            initializer=initializer)
        # r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
        #                            initializer=initializer)

        #####past
        AR_layer = []
        for i in range(n_layer):
            with tf.variable_scope('past_layer_{}'.format(i)):
                # output_past,AR=rel_multihead_attn(
                #     w=output_past_ori,
                #     r=pos_emb_past,
                #     r_w_bias=r_w_bias,
                #     r_r_bias=r_r_bias,
                #     d_model=d_model,
                #     n_head=n_head,
                #     d_head=d_head,
                #     dropout=dropout,
                #     dropatt=dropatt,
                #     is_training=is_training,
                #     kernel_initializer=initializer)
                output_past,c_m_past,m_p_past,side_inform,AR = multihead_attn_tpe3(
                    w=output_past,
                    w_o=output_past_ori,
                    c_m=c_m_past,
                    m_p=m_p_past,
                    side=side_inform,
                    attn_mask=attn_mask_past,
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer,
                    head_attention_flag=head_attention_flag,
                    side_flag=side_input,
                    cp_flag=cp_input,qlen_input=past_len)
                print(output_past)
                AR_layer.append(AR)
                output_past = positionwise_FF(
                    inp=output_past,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)
        print(output_past)
        # com_past = tf.slice(output_past, [0, 0, 0], [past_len, com_len, d_model])
        # mk_past = tf.slice(output_past, [0, com_len, 0], [past_len, 1, d_model])
        # print(com_past)
        # print(d_model)
        # inp, d_model, d_inner, dropout, initializer, len_time, m_dense, is_training,
        # only_last_one = True, scope = 'ff_dense'):

        com_past = spcific_net_ff_dense(inp=output_past,
                                        d_model=d_model,
                                        d_inner=d_spcific_inner,
                                        dropout=dropout, initializer=initializer, len_time=past_len, m_dense=m_dense,
                                        is_training=is_training,
                                        only_last_one=True,
                                        scope='spcific_ff_past'
                                        )
        com_output = com_past
        mtl_l2_reg=tf.constant(0.0)

        if side_flag:
            side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
                                              d_inner=d_side_inner, dropout=dropout, initializer=initializer,
                                              scope='side_spcific')
            com_output = tf.concat([com_output, side_inform], axis=1)
        else:
            pass
        com_output = spcific_nets_simple(com_output, is_training=is_training, d_original=classify_num,
                                         d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                         scope='com_final_spcific')

        print(com_output)
        # exit()

        # return com_output, mtl_l2_reg
        return com_output, mtl_l2_reg,MIX_att,AR_layer

def trans_nets(classify_num, batch_inp, c_m_batch_inp, m_p_batch_inp, side_inform, past_len, future_len, is_training,
                  d_step,
                  d_model,
                  n_layer, n_head, d_head, d_inner, d_spcific_inner, d_input_inner, d_side_inner,
                  d_side_out, dropout, dropatt, m_dense, initializer, init_embeding=True,
                  mask_same_length=False, side_flag=False, head_attention_flag=False, side_input=True, cp_input=True,
                  Bi_attention=True, scope='level3_nets'):
    with tf.variable_scope(scope):
        #########################################
        # ori_inp=batch_inp
        # head_all=tf.concat([tf.transpose(batch_inp[:,:,-1],[1,0]),tf.transpose(c_m_batch_inp[:,:,-1],[1,0]),tf.transpose(m_p_batch_inp[:,:,-1],[1,0]),side_inform],axis=1)
        # att_head=spcific_nets_simple(head_all,is_training=is_training,d_original=3,d_inner=32,dropout=dropout,initializer=initializer,scope='head_1')
        # att_score=tf.nn.softmax(att_head)
        # MIX_att = att_score
        # head_all_v=tf.concat([tf.expand_dims(batch_inp,-1),tf.expand_dims(c_m_batch_inp,-1),tf.expand_dims(m_p_batch_inp,-1)],axis=3)
        # batch_inp=tf.einsum('lbdn,bn->lbd',head_all_v,att_score)
        MIX_att = tf.constant(0)
        #####################################
        # side_inform = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), tf.transpose(c_m_batch_inp[:, :, -1], [1, 0]),
        #      tf.transpose(m_p_batch_inp[:, :, -1], [1, 0]), side_inform+1e-20], axis=1)

        # side_inform = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        #                                             d_inner=d_side_inner, dropout=0.1, initializer=initializer,
        #                                             layer=1,scope='side_imbedding')

        # context_pe=spcific_nets_simple(side_inform, is_training=is_training, d_original=d_model*past_len,
        #                                             d_inner=d_side_inner, dropout=0.1, initializer=initializer,
        #                                             scope='pe_imbedding')
        # context_pe=tf.reshape(context_pe,[-1,past_len,d_model])
        # context_pe=tf.transpose(context_pe,[1,0,2])
        ##########################################
        # pos_seq_past = tf.range(past_len - 1, -1, -1.0)
        # inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        # inv_freq = tf.constant(inv_freq)
        # pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        # pos_emb_past=tf.reshape(tf.transpose(pos_emb_past,[1,0,2]),[1,past_len*d_model])
        # pos_emb_past=tf.tile(pos_emb_past,[32,1])
        # side_inform_pe = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform,pos_emb_past], axis=1)
        # # side_inform = tf.concat(
        # #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # context_pe = spcific_nets_simple(side_inform_pe, is_training=is_training, d_original=d_model * past_len,
        #                                  d_inner=d_side_inner*2, dropout=0.1, initializer=initializer,
        #                                  scope='pe_imbedding')
        # context_pe = tf.reshape(context_pe, [-1, past_len, d_model])
        # context_pe = tf.transpose(context_pe, [1, 0, 2])
        ###########################################


        dec_inp_past = batch_inp

        c_m_past = c_m_batch_inp

        m_p_past = m_p_batch_inp

        # if side_input:
        # side_inform_imbedding = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        #                                             d_inner=d_side_inner, dropout=dropout, initializer=initializer,
        #                                             scope='side_imbedding')
        # side_inform_imbedding = tf.tile(tf.expand_dims(side_inform_imbedding, 0), [past_len, 1, 1])
        # print(side_inform_imbedding)
        # print(dec_inp_past)
        # exit()
        # dec_inp_past = tf.concat([dec_inp_past, side_inform_imbedding], axis=2)
        # dec_inp_future = tf.concat([dec_inp_future, side_inform_imbedding], axis=2)
        if init_embeding:
            c_m_past = spcific_nets_simple(c_m_past, is_training=is_training, d_original=d_model,
                                           d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                           scope='embedding_past_c_m')
            m_p_past = spcific_nets_simple(m_p_past, is_training=is_training, d_original=d_model,
                                           d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                           scope='embedding_past_m_p')
        if init_embeding:
            # dec_inp_past = spcific_nets_simple(dec_inp_past, is_training=is_training, d_original=d_model,
            #                                    d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
            #                                    scope='imbedding_past')
            dec_inp_past = tf.layers.dense(dec_inp_past, d_model, use_bias=False, kernel_initializer=initializer,
                                      name='embedding')
        else:
            d_model = d_step

        qlen_past = tf.shape(dec_inp_past)[0]

        embeddings_past = dec_inp_past

        attn_mask_past = _create_mask(qlen_past, 0, mask_same_length)

        pos_seq_past = tf.range(qlen_past - 1, -1, -1.0)
        inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        inv_freq = tf.constant(inv_freq)
        pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        print(pos_emb_past)
        # print(context_pe)
        # exit()

        output_past = tf.layers.dropout(embeddings_past, dropout, training=is_training)

        c_m_past = tf.layers.dropout(c_m_past, dropout, training=is_training)
        m_p_past = tf.layers.dropout(m_p_past, dropout, training=is_training)

        pos_emb_past = tf.layers.dropout(pos_emb_past, dropout, training=is_training)

        output_past_ori=output_past
        output_past = output_past + pos_emb_past
        # output_past = output_past + context_pe
        # output_past = output_past + tf.nn.l2_normalize(context_pe,axis=0)
        # output_past = tf.concat([output_past,context_pe],axis=2)
        # output_past = tf.concat([output_past,tf.nn.l2_normalize(context_pe,axis=0)],axis=2)
        # print(output_past)
        # exit()
        #
        # c_m_past = c_m_past + pos_emb_past
        # m_p_past = m_p_past + pos_emb_past

        # r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
        #                            initializer=initializer)
        # r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
        #                            initializer=initializer)

        #####past
        AR_layer = []
        for i in range(n_layer):
            with tf.variable_scope('past_layer_{}'.format(i)):
                # output_past,AR=rel_multihead_attn(
                #     w=output_past_ori,
                #     r=pos_emb_past,
                #     r_w_bias=r_w_bias,
                #     r_r_bias=r_r_bias,
                #     d_model=d_model,
                #     n_head=n_head,
                #     d_head=d_head,
                #     dropout=dropout,
                #     dropatt=dropatt,
                #     is_training=is_training,
                #     kernel_initializer=initializer)
                output_past, c_m_past, m_p_past, side_inform = multihead_attn(
                    w=output_past,
                    c_m=c_m_past,
                    m_p=m_p_past,
                    side=side_inform,
                    attn_mask=attn_mask_past,
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer,
                    head_attention_flag=False,
                    side_flag=False,
                    cp_flag=False)
                # output_past,c_m_past,m_p_past,side_inform,AR = multihead_attn_tpe3(
                #     w=output_past,
                #     w_o=output_past_ori,
                #     c_m=c_m_past,
                #     m_p=m_p_past,
                #     side=side_inform,
                #     attn_mask=attn_mask_past,
                #     d_model=d_model,
                #     n_head=n_head,
                #     d_head=d_head,
                #     dropout=dropout,
                #     dropatt=dropatt,
                #     is_training=is_training,
                #     kernel_initializer=initializer,
                #     head_attention_flag=head_attention_flag,
                #     side_flag=side_input,
                #     cp_flag=cp_input,qlen_input=past_len)
                # print(output_past)
                AR=tf.constant(0)
                AR_layer.append(AR)
                output_past = positionwise_FF(
                    inp=output_past,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)
        print(output_past)
        # com_past = tf.slice(output_past, [0, 0, 0], [past_len, com_len, d_model])
        # mk_past = tf.slice(output_past, [0, com_len, 0], [past_len, 1, d_model])
        # print(com_past)
        # print(d_model)
        # inp, d_model, d_inner, dropout, initializer, len_time, m_dense, is_training,
        # only_last_one = True, scope = 'ff_dense'):

        com_output = flat(output_past, only_last_one=False)

        # com_past = spcific_net_ff_dense(inp=output_past,
        #                                 d_model=d_model,
        #                                 d_inner=d_spcific_inner,
        #                                 dropout=dropout, initializer=initializer, len_time=past_len, m_dense=m_dense,
        #                                 is_training=is_training,
        #                                 only_last_one=True,
        #                                 scope='spcific_ff_past'
        #                                 )
        # com_output = com_past
        mtl_l2_reg=tf.constant(0.0)

        if side_flag:
            side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
                                              d_inner=d_side_inner, dropout=dropout, initializer=initializer,
                                              scope='side_spcific')
            com_output = tf.concat([com_output, side_inform], axis=1)
        else:
            pass
        com_output = spcific_nets_simple(com_output, is_training=is_training, d_original=classify_num,
                                         d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                         scope='com_final_spcific')

        print(com_output)
        # exit()

        # return com_output, mtl_l2_reg
        return com_output, mtl_l2_reg,MIX_att,AR_layer

def LSTNets(classify_num, batch_inp, c_m_batch_inp, m_p_batch_inp, side_inform, past_len, future_len, is_training,
                  d_step,
                  d_model,
                  n_layer, n_head, d_head, d_inner, d_spcific_inner, d_input_inner, d_side_inner,
                  d_side_out, dropout, dropatt, m_dense, initializer, init_embeding=True,
                  mask_same_length=False, side_flag=False, head_attention_flag=False, side_input=True, cp_input=True,
                  Bi_attention=True, scope='level3_nets'):
    with tf.variable_scope(scope):
        #########################################
        # ori_inp=batch_inp
        # head_all=tf.concat([tf.transpose(batch_inp[:,:,-1],[1,0]),tf.transpose(c_m_batch_inp[:,:,-1],[1,0]),tf.transpose(m_p_batch_inp[:,:,-1],[1,0]),side_inform],axis=1)
        # att_head=spcific_nets_simple(head_all,is_training=is_training,d_original=3,d_inner=32,dropout=dropout,initializer=initializer,scope='head_1')
        # att_score=tf.nn.softmax(att_head)
        # MIX_att = att_score
        # head_all_v=tf.concat([tf.expand_dims(batch_inp,-1),tf.expand_dims(c_m_batch_inp,-1),tf.expand_dims(m_p_batch_inp,-1)],axis=3)
        # batch_inp=tf.einsum('lbdn,bn->lbd',head_all_v,att_score)
        MIX_att = tf.constant(0)
        #####################################
        # side_inform = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), tf.transpose(c_m_batch_inp[:, :, -1], [1, 0]),
        #      tf.transpose(m_p_batch_inp[:, :, -1], [1, 0]), side_inform+1e-20], axis=1)

        # side_inform = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        #                                             d_inner=d_side_inner, dropout=0.1, initializer=initializer,
        #                                             layer=1,scope='side_imbedding')

        # context_pe=spcific_nets_simple(side_inform, is_training=is_training, d_original=d_model*past_len,
        #                                             d_inner=d_side_inner, dropout=0.1, initializer=initializer,
        #                                             scope='pe_imbedding')
        # context_pe=tf.reshape(context_pe,[-1,past_len,d_model])
        # context_pe=tf.transpose(context_pe,[1,0,2])
        ##########################################
        # pos_seq_past = tf.range(past_len - 1, -1, -1.0)
        # inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        # inv_freq = tf.constant(inv_freq)
        # pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        # pos_emb_past=tf.reshape(tf.transpose(pos_emb_past,[1,0,2]),[1,past_len*d_model])
        # pos_emb_past=tf.tile(pos_emb_past,[32,1])
        # side_inform_pe = tf.concat(
        #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform,pos_emb_past], axis=1)
        # # side_inform = tf.concat(
        # #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # context_pe = spcific_nets_simple(side_inform_pe, is_training=is_training, d_original=d_model * past_len,
        #                                  d_inner=d_side_inner*2, dropout=0.1, initializer=initializer,
        #                                  scope='pe_imbedding')
        # context_pe = tf.reshape(context_pe, [-1, past_len, d_model])
        # context_pe = tf.transpose(context_pe, [1, 0, 2])
        ###########################################


        dec_inp_past = batch_inp

        c_m_past = c_m_batch_inp

        m_p_past = m_p_batch_inp

        # if side_input:
        # side_inform_imbedding = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        #                                             d_inner=d_side_inner, dropout=dropout, initializer=initializer,
        #                                             scope='side_imbedding')
        # side_inform_imbedding = tf.tile(tf.expand_dims(side_inform_imbedding, 0), [past_len, 1, 1])
        # print(side_inform_imbedding)
        # print(dec_inp_past)
        # exit()
        # dec_inp_past = tf.concat([dec_inp_past, side_inform_imbedding], axis=2)
        # dec_inp_future = tf.concat([dec_inp_future, side_inform_imbedding], axis=2)
        if init_embeding:
            c_m_past = spcific_nets_simple(c_m_past, is_training=is_training, d_original=d_model,
                                           d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                           scope='embedding_past_c_m')
            m_p_past = spcific_nets_simple(m_p_past, is_training=is_training, d_original=d_model,
                                           d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                           scope='embedding_past_m_p')
        if init_embeding:
            # dec_inp_past = spcific_nets_simple(dec_inp_past, is_training=is_training, d_original=d_model,
            #                                    d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
            #                                    scope='imbedding_past')
            dec_inp_past = tf.layers.dense(dec_inp_past, d_model, use_bias=False, kernel_initializer=initializer,
                                      name='embedding')
        else:
            d_model = d_step

        qlen_past = tf.shape(dec_inp_past)[0]

        embeddings_past = dec_inp_past

        attn_mask_past = _create_mask(qlen_past, 0, mask_same_length)

        pos_seq_past = tf.range(qlen_past - 1, -1, -1.0)
        inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        inv_freq = tf.constant(inv_freq)
        pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        print(pos_emb_past)
        # print(context_pe)
        # exit()

        output_past = tf.layers.dropout(embeddings_past, dropout, training=is_training)

        c_m_past = tf.layers.dropout(c_m_past, dropout, training=is_training)
        m_p_past = tf.layers.dropout(m_p_past, dropout, training=is_training)

        pos_emb_past = tf.layers.dropout(pos_emb_past, dropout, training=is_training)

        output_past_ori=output_past
        output_past = output_past + pos_emb_past
        # output_past = output_past + context_pe
        # output_past = output_past + tf.nn.l2_normalize(context_pe,axis=0)
        # output_past = tf.concat([output_past,context_pe],axis=2)
        # output_past = tf.concat([output_past,tf.nn.l2_normalize(context_pe,axis=0)],axis=2)
        # print(output_past)
        # exit()
        #
        # c_m_past = c_m_past + pos_emb_past
        # m_p_past = m_p_past + pos_emb_past

        # r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
        #                            initializer=initializer)
        # r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
        #                            initializer=initializer)

        #####past
        AR_layer = []
        for i in range(n_layer):
            with tf.variable_scope('past_layer_{}'.format(i)):
                # output_past,AR=rel_multihead_attn(
                #     w=output_past_ori,
                #     r=pos_emb_past,
                #     r_w_bias=r_w_bias,
                #     r_r_bias=r_r_bias,
                #     d_model=d_model,
                #     n_head=n_head,
                #     d_head=d_head,
                #     dropout=dropout,
                #     dropatt=dropatt,
                #     is_training=is_training,
                #     kernel_initializer=initializer)
                output_past, c_m_past, m_p_past, side_inform = multihead_attn(
                    w=output_past,
                    c_m=c_m_past,
                    m_p=m_p_past,
                    side=side_inform,
                    attn_mask=attn_mask_past,
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer,
                    head_attention_flag=False,
                    side_flag=False,
                    cp_flag=False)
                # output_past,c_m_past,m_p_past,side_inform,AR = multihead_attn_tpe3(
                #     w=output_past,
                #     w_o=output_past_ori,
                #     c_m=c_m_past,
                #     m_p=m_p_past,
                #     side=side_inform,
                #     attn_mask=attn_mask_past,
                #     d_model=d_model,
                #     n_head=n_head,
                #     d_head=d_head,
                #     dropout=dropout,
                #     dropatt=dropatt,
                #     is_training=is_training,
                #     kernel_initializer=initializer,
                #     head_attention_flag=head_attention_flag,
                #     side_flag=side_input,
                #     cp_flag=cp_input,qlen_input=past_len)
                # print(output_past)
                AR=tf.constant(0)
                AR_layer.append(AR)
                output_past = positionwise_FF(
                    inp=output_past,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)
        print(output_past)
        # com_past = tf.slice(output_past, [0, 0, 0], [past_len, com_len, d_model])
        # mk_past = tf.slice(output_past, [0, com_len, 0], [past_len, 1, d_model])
        # print(com_past)
        # print(d_model)
        # inp, d_model, d_inner, dropout, initializer, len_time, m_dense, is_training,
        # only_last_one = True, scope = 'ff_dense'):

        com_past = spcific_net_ff_dense(inp=output_past,
                                        d_model=d_model,
                                        d_inner=d_spcific_inner,
                                        dropout=dropout, initializer=initializer, len_time=past_len, m_dense=m_dense,
                                        is_training=is_training,
                                        only_last_one=True,
                                        scope='spcific_ff_past'
                                        )
        com_output = com_past
        mtl_l2_reg=tf.constant(0.0)

        if side_flag:
            side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
                                              d_inner=d_side_inner, dropout=dropout, initializer=initializer,
                                              scope='side_spcific')
            com_output = tf.concat([com_output, side_inform], axis=1)
        else:
            pass
        com_output = spcific_nets_simple(com_output, is_training=is_training, d_original=classify_num,
                                         d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                         scope='com_final_spcific')

        print(com_output)
        # exit()

        # return com_output, mtl_l2_reg
        return com_output, mtl_l2_reg,MIX_att,AR_layer

def TCN(classify_num, batch_inp, c_m_batch_inp, m_p_batch_inp, side_inform, past_len, future_len, is_training,
                  d_step,
                  d_model,
                  n_layer, n_head, d_head, d_inner, d_spcific_inner, d_input_inner, d_side_inner,
                  d_side_out, dropout, dropatt, m_dense, initializer, init_embeding=True,
                  mask_same_length=False, side_flag=False, head_attention_flag=False, side_input=True, cp_input=True,
                  Bi_attention=True, scope='level3_nets'):
    with tf.variable_scope(scope):
        x=tf.transpose(batch_inp, [1, 0,2])
        # x = tf.convert_to_tensor(np.random.random((100, 10, 50)), dtype=tf.float32)  # batch, seq_len, dim
        # y = TemporalConvNet(x, num_channels=[8, 16, 8], sequence_length=6)
        # y = TemporalConvNet(x, num_channels=[8, 16,16, 8], sequence_length=6)
        y = TemporalConvNet(x, num_channels=[16,32, 16], sequence_length=6)
        # y = TemporalConvNet(x, num_channels=[16,32,32, 16], sequence_length=6)
        y=tf.layers.flatten(y)
        # print(y.shape)
        com_output = spcific_nets_simple(y, is_training=is_training, d_original=classify_num,
                                         d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
                                         scope='com_final_spcific')
        # print(com_output.shape)
        # exit()
        # # y = model(x,training=True)
        # print('-----')
        # print(y.shape)
        mtl_l2_reg = tf.constant(0.0)
        MIX_att = tf.constant(0.0)
        AR_layer = []

        # #########################################
        # # ori_inp=batch_inp
        # # head_all=tf.concat([tf.transpose(batch_inp[:,:,-1],[1,0]),tf.transpose(c_m_batch_inp[:,:,-1],[1,0]),tf.transpose(m_p_batch_inp[:,:,-1],[1,0]),side_inform],axis=1)
        # # att_head=spcific_nets_simple(head_all,is_training=is_training,d_original=3,d_inner=32,dropout=dropout,initializer=initializer,scope='head_1')
        # # att_score=tf.nn.softmax(att_head)
        # # MIX_att = att_score
        # # head_all_v=tf.concat([tf.expand_dims(batch_inp,-1),tf.expand_dims(c_m_batch_inp,-1),tf.expand_dims(m_p_batch_inp,-1)],axis=3)
        # # batch_inp=tf.einsum('lbdn,bn->lbd',head_all_v,att_score)
        # MIX_att = tf.constant(0)
        # #####################################
        # # side_inform = tf.concat(
        # #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), tf.transpose(c_m_batch_inp[:, :, -1], [1, 0]),
        # #      tf.transpose(m_p_batch_inp[:, :, -1], [1, 0]), side_inform+1e-20], axis=1)
        #
        # # side_inform = tf.concat(
        # #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # # side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        # #                                             d_inner=d_side_inner, dropout=0.1, initializer=initializer,
        # #                                             layer=1,scope='side_imbedding')
        #
        # # context_pe=spcific_nets_simple(side_inform, is_training=is_training, d_original=d_model*past_len,
        # #                                             d_inner=d_side_inner, dropout=0.1, initializer=initializer,
        # #                                             scope='pe_imbedding')
        # # context_pe=tf.reshape(context_pe,[-1,past_len,d_model])
        # # context_pe=tf.transpose(context_pe,[1,0,2])
        # ##########################################
        # # pos_seq_past = tf.range(past_len - 1, -1, -1.0)
        # # inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        # # inv_freq = tf.constant(inv_freq)
        # # pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        # # pos_emb_past=tf.reshape(tf.transpose(pos_emb_past,[1,0,2]),[1,past_len*d_model])
        # # pos_emb_past=tf.tile(pos_emb_past,[32,1])
        # # side_inform_pe = tf.concat(
        # #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform,pos_emb_past], axis=1)
        # # # side_inform = tf.concat(
        # # #     [tf.transpose(batch_inp[:, :, -1], [1, 0]), side_inform], axis=1)
        # # context_pe = spcific_nets_simple(side_inform_pe, is_training=is_training, d_original=d_model * past_len,
        # #                                  d_inner=d_side_inner*2, dropout=0.1, initializer=initializer,
        # #                                  scope='pe_imbedding')
        # # context_pe = tf.reshape(context_pe, [-1, past_len, d_model])
        # # context_pe = tf.transpose(context_pe, [1, 0, 2])
        # ###########################################
        #
        #
        # dec_inp_past = batch_inp
        #
        # c_m_past = c_m_batch_inp
        #
        # m_p_past = m_p_batch_inp
        #
        # # if side_input:
        # # side_inform_imbedding = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        # #                                             d_inner=d_side_inner, dropout=dropout, initializer=initializer,
        # #                                             scope='side_imbedding')
        # # side_inform_imbedding = tf.tile(tf.expand_dims(side_inform_imbedding, 0), [past_len, 1, 1])
        # # print(side_inform_imbedding)
        # # print(dec_inp_past)
        # # exit()
        # # dec_inp_past = tf.concat([dec_inp_past, side_inform_imbedding], axis=2)
        # # dec_inp_future = tf.concat([dec_inp_future, side_inform_imbedding], axis=2)
        # if init_embeding:
        #     c_m_past = spcific_nets_simple(c_m_past, is_training=is_training, d_original=d_model,
        #                                    d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
        #                                    scope='embedding_past_c_m')
        #     m_p_past = spcific_nets_simple(m_p_past, is_training=is_training, d_original=d_model,
        #                                    d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
        #                                    scope='embedding_past_m_p')
        # if init_embeding:
        #     # dec_inp_past = spcific_nets_simple(dec_inp_past, is_training=is_training, d_original=d_model,
        #     #                                    d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
        #     #                                    scope='imbedding_past')
        #     dec_inp_past = tf.layers.dense(dec_inp_past, d_model, use_bias=False, kernel_initializer=initializer,
        #                               name='embedding')
        # else:
        #     d_model = d_step
        #
        # qlen_past = tf.shape(dec_inp_past)[0]
        #
        # embeddings_past = dec_inp_past
        #
        # attn_mask_past = _create_mask(qlen_past, 0, mask_same_length)
        #
        # pos_seq_past = tf.range(qlen_past - 1, -1, -1.0)
        # inv_freq = [1 / (10000.0 ** (i / d_model)) for i in range(0, d_model, 2)]
        # inv_freq = tf.constant(inv_freq)
        # pos_emb_past = positional_embedding(pos_seq_past, inv_freq)
        # print(pos_emb_past)
        # # print(context_pe)
        # # exit()
        #
        # output_past = tf.layers.dropout(embeddings_past, dropout, training=is_training)
        #
        # c_m_past = tf.layers.dropout(c_m_past, dropout, training=is_training)
        # m_p_past = tf.layers.dropout(m_p_past, dropout, training=is_training)
        #
        # pos_emb_past = tf.layers.dropout(pos_emb_past, dropout, training=is_training)
        #
        # output_past_ori=output_past
        # output_past = output_past + pos_emb_past
        # # output_past = output_past + context_pe
        # # output_past = output_past + tf.nn.l2_normalize(context_pe,axis=0)
        # # output_past = tf.concat([output_past,context_pe],axis=2)
        # # output_past = tf.concat([output_past,tf.nn.l2_normalize(context_pe,axis=0)],axis=2)
        # # print(output_past)
        # # exit()
        # #
        # # c_m_past = c_m_past + pos_emb_past
        # # m_p_past = m_p_past + pos_emb_past
        #
        # # r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
        # #                            initializer=initializer)
        # # r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
        # #                            initializer=initializer)
        #
        # #####past
        # AR_layer = []
        # for i in range(n_layer):
        #     with tf.variable_scope('past_layer_{}'.format(i)):
        #         # output_past,AR=rel_multihead_attn(
        #         #     w=output_past_ori,
        #         #     r=pos_emb_past,
        #         #     r_w_bias=r_w_bias,
        #         #     r_r_bias=r_r_bias,
        #         #     d_model=d_model,
        #         #     n_head=n_head,
        #         #     d_head=d_head,
        #         #     dropout=dropout,
        #         #     dropatt=dropatt,
        #         #     is_training=is_training,
        #         #     kernel_initializer=initializer)
        #         output_past, c_m_past, m_p_past, side_inform = multihead_attn(
        #             w=output_past,
        #             c_m=c_m_past,
        #             m_p=m_p_past,
        #             side=side_inform,
        #             attn_mask=attn_mask_past,
        #             d_model=d_model,
        #             n_head=n_head,
        #             d_head=d_head,
        #             dropout=dropout,
        #             dropatt=dropatt,
        #             is_training=is_training,
        #             kernel_initializer=initializer,
        #             head_attention_flag=False,
        #             side_flag=False,
        #             cp_flag=False)
        #         # output_past,c_m_past,m_p_past,side_inform,AR = multihead_attn_tpe3(
        #         #     w=output_past,
        #         #     w_o=output_past_ori,
        #         #     c_m=c_m_past,
        #         #     m_p=m_p_past,
        #         #     side=side_inform,
        #         #     attn_mask=attn_mask_past,
        #         #     d_model=d_model,
        #         #     n_head=n_head,
        #         #     d_head=d_head,
        #         #     dropout=dropout,
        #         #     dropatt=dropatt,
        #         #     is_training=is_training,
        #         #     kernel_initializer=initializer,
        #         #     head_attention_flag=head_attention_flag,
        #         #     side_flag=side_input,
        #         #     cp_flag=cp_input,qlen_input=past_len)
        #         # print(output_past)
        #         AR=tf.constant(0)
        #         AR_layer.append(AR)
        #         output_past = positionwise_FF(
        #             inp=output_past,
        #             d_model=d_model,
        #             d_inner=d_inner,
        #             dropout=dropout,
        #             kernel_initializer=initializer,
        #             is_training=is_training)
        # print(output_past)
        # # com_past = tf.slice(output_past, [0, 0, 0], [past_len, com_len, d_model])
        # # mk_past = tf.slice(output_past, [0, com_len, 0], [past_len, 1, d_model])
        # # print(com_past)
        # # print(d_model)
        # # inp, d_model, d_inner, dropout, initializer, len_time, m_dense, is_training,
        # # only_last_one = True, scope = 'ff_dense'):
        #
        # com_past = spcific_net_ff_dense(inp=output_past,
        #                                 d_model=d_model,
        #                                 d_inner=d_spcific_inner,
        #                                 dropout=dropout, initializer=initializer, len_time=past_len, m_dense=m_dense,
        #                                 is_training=is_training,
        #                                 only_last_one=True,
        #                                 scope='spcific_ff_past'
        #                                 )
        # com_output = com_past
        # mtl_l2_reg=tf.constant(0.0)
        #
        # if side_flag:
        #     side_inform = spcific_nets_simple(side_inform, is_training=is_training, d_original=d_side_out,
        #                                       d_inner=d_side_inner, dropout=dropout, initializer=initializer,
        #                                       scope='side_spcific')
        #     com_output = tf.concat([com_output, side_inform], axis=1)
        # else:
        #     pass
        # com_output = spcific_nets_simple(com_output, is_training=is_training, d_original=classify_num,
        #                                  d_inner=d_spcific_inner, dropout=dropout, initializer=initializer,
        #                                  scope='com_final_spcific')
        #
        # print(com_output)
        # # exit()
        #
        # # return com_output, mtl_l2_reg
        return com_output, mtl_l2_reg,MIX_att,AR_layer
