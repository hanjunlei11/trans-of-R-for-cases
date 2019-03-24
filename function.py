#! -*- coding: utf-8 -*-
from config import *
import tensorflow as tf
from config import *
from tool import *
'''
inputs是一个形如(batch_size, seq_len, word_size)的张量；
函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
'''
def Position_Embedding(inputs, position_size):
    batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000., 2 * tf.range(position_size / 2, dtype=tf.float32) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(position_ij, 0) + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding

'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''
def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len,data_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

'''
普通的全连接
inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
'''
def Dense(inputs, ouput_size,keep_rate=None,activition='relu', bias=False, seq_len=None,mask_mode='add'):
    outputs = tf.layers.dense(inputs=inputs,units=ouput_size,use_bias=bias,kernel_regularizer=tf.contrib.layers.l2_regularizer(10e-6))
    if seq_len != None:
        outputs = Mask(outputs, seq_len,mask_mode)
    if activition is 'relu':
        outputs = tf.nn.relu(outputs)
    elif activition is 'leaky_relu':
        outputs = tf.nn.leaky_relu(outputs)
    elif activition is 'sigmoid':
        outputs = tf.nn.sigmoid(outputs)
    if keep_rate is not None:
        outputs = tf.nn.dropout(outputs,keep_prob=keep_rate)
    return outputs

'''
Multi-Head Attention的实现
'''
def multi_head_attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
    #对Q、K、V分别作线性映射
    query = Dense(Q, nb_head * size_per_head,bias= False)
    query = tf.reshape(query, (-1, tf.shape(query)[1], nb_head, size_per_head))
    query = tf.transpose(query, [0, 2, 1, 3])
    key = Dense(K, nb_head * size_per_head,bias= False)
    key = tf.reshape(key, (-1, tf.shape(key)[1], nb_head, size_per_head))
    key = tf.transpose(key, [0, 2, 1, 3])
    value = Dense(V, nb_head * size_per_head,bias= False)
    value = tf.reshape(value, (-1, tf.shape(value)[1], nb_head, size_per_head))
    value = tf.transpose(value, [0, 2, 1, 3])
    #计算内积，然后mask，然后softmax
    A = tf.matmul(query, key, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    #输出并mask
    output = tf.matmul(A, value)
    output = tf.transpose(output, [0, 2, 1, 3])
    output = tf.reshape(output, (-1, tf.shape(output)[1], nb_head * size_per_head))
    output = Mask(output, Q_len, mode='mul')
    return output

def feed_forward(inputs,seq_len=None,keep_rate=None,activition='relu'):
    shapes = int(inputs.shape[-1])
    dense = Dense(inputs=inputs,seq_len=seq_len, ouput_size=shapes*4,keep_rate=keep_rate,activition=activition)
    dense = Dense(inputs=dense,seq_len=seq_len, ouput_size=shapes,keep_rate=keep_rate,activition=activition)
    return dense

'''
前向传播encoder部分，输入是（batch_size*2, seq_len, word_size）形状，经过multi_head_attention
然后残差连接和norm，再经过两层全连接，最后残差连接和norm
输出是（batch_size, seq_len, word_size）形状
'''

def transformer(inputs,embedding_size,nb_layers,nb_head,size_per_head,Q_len=None,V_len=None,training=True,keep_rate=None,activition='relu',use_position = False):
    batch = inputs
    if use_position is True:
        position = Position_Embedding(inputs=inputs, position_size=embedding_size)
        batch = tf.concat([position,inputs],axis=-1)
    for i in range(nb_layers):
        output = multi_head_attention(Q=batch,K=batch,V=batch,nb_head=nb_head,size_per_head=size_per_head,Q_len=Q_len,V_len=V_len)
        norm = tf.layers.batch_normalization(inputs=output,training=training)
        dense_1 = feed_forward(inputs=norm,keep_rate=keep_rate,activition=activition)
        batch = tf.concat([batch,dense_1],axis=-1)
        batch = tf.layers.batch_normalization(inputs=batch,training=training)
    return batch

def train(Model,datafile,labelfile,epoch_number,keep_rate,is_trainning):
    data_all_train, data_all_test, data_r_train, data_r_test, data_label_train, data_label_test = read_file(datafile=datafile,labelflie=labelfile)
    print('读文件完成')
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt_op = tf.train.AdamOptimizer(learning_rate=learnning_rate).minimize(Model.losses)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        if is_trainning is True:
            sess.run(tf.global_variables_initializer())
        elif is_trainning is False:
            ckpt = tf.train.get_checkpoint_state('./ckpt_tans/')
            saver.restore(sess, save_path=ckpt.model_checkpoint_path)
        print('初始化完成')
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs_train/", sess.graph)
        print('开始训练')
        min_loss = 0
        for number in range(epoch_number):

            epoch_fzss, epoch_r, epoch_label, lenx = get_epoch(batch_size=batch_size, data_all=data_all_train, data_r=data_r_train,data_label=data_label_train)
            for i in range(lenx):
                batch_fzss,batch_r,batch_label = get_batch(epoch_fzss, epoch_r, epoch_label,i)
                feed_dic = {Model.batch_s:batch_fzss,Model.batch_r:batch_r,Model.batch_label:batch_label,Model.keep_rate:keep_rate,Model.is_trainning:is_trainning}
                _,loss,acc,rs = sess.run([opt_op,Model.losses,Model.accur,merged],feed_dict=feed_dic)
                writer.add_summary(rs, number*lenx+i)
                print('epopch',int(number+1),'batch:',i,'loss: ',loss,'acc: ',acc)

            epoch_fzss, epoch_r, epoch_label, lenx_test = get_epoch(batch_size=batch_size, data_all=data_all_test, data_r=data_r_test,data_label=data_label_test)
            losses = 0
            for i in range(lenx_test):
                batch_fzss, batch_r, batch_label = get_batch(epoch_fzss, epoch_r, epoch_label, i)
                feed_dic = {Model.batch_s: batch_fzss, Model.batch_r: batch_r, Model.batch_label: batch_label,
                            Model.keep_rate: 1.0, Model.is_trainning: False}
                loss, acc= sess.run([Model.losses, Model.accur], feed_dict=feed_dic)
                print('test_loss: ', loss, 'test_acc: ', acc)
                losses+=loss
            losses = losses/lenx_test
            if losses>=min_loss:
                min_loss = losses
                saver.save(sess, save_path='ckpt_test/model.ckpt', global_step=number + 1)




