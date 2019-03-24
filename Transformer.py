import tensorflow as tf
import numpy as np
from function import *
from config import *

class Model():
    def __init__(self):
        self.vocab_size = vocab_size
        self.data_len = data_len
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hiddin_size = hiddin_size

        with tf.name_scope('input'):
            self.batch_s = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.data_len))
            self.batch_r = tf.placeholder(dtype=tf.float32,shape=(self.batch_size,3))
            self.batch_label = tf.placeholder(dtype=tf.int64,shape=(self.batch_size,8))
            self.keep_rate = tf.placeholder(dtype=tf.float32,shape=(None))
            self.is_trainning = tf.placeholder(dtype=tf.bool,shape=(None))

        with tf.name_scope('embedding'):
            embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], dtype=tf.float32),
                                          trainable=True, name='word_embedding')
            self.batch = tf.nn.embedding_lookup(embedding_table,self.batch_s)
            self.batch = tf.nn.dropout(self.batch,keep_prob=self.keep_rate)

        with tf.name_scope('transformer'):
            self.batch_tran = transformer(inputs=self.batch,embedding_size=self.hiddin_size,nb_head=nb_head,size_per_head=size_per_head,nb_layers=nb_layers,training=self.is_trainning,keep_rate=self.keep_rate,use_position=True)

        with tf.name_scope('pooling_and_dense'):
            pooling = tf.layers.max_pooling1d(inputs=self.batch_tran, pool_size=180, strides=1)
            pooling = tf.reduce_mean(input_tensor=pooling, axis=1)
            pooling = tf.concat([pooling,self.batch_r],axis=-1)
            pre = []
            for i in range(8):
                dense = Dense(inputs=pooling,ouput_size=240,keep_rate=self.keep_rate)
                dense = tf.layers.batch_normalization(inputs=dense,training=self.is_trainning)
                dense = Dense(inputs=dense,ouput_size=24,keep_rate=self.keep_rate)
                dense = tf.layers.batch_normalization(inputs=dense, training=self.is_trainning)
                dense = Dense(inputs=dense,ouput_size=2)
                pre.append(dense)
        with tf.name_scope('loss_all'):
            for i in range(8):
                predction = pre[i]
                labbel = tf.reduce_mean(tf.slice(self.batch_label,[0,i],[-1,1]),axis=-1)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labbel,logits=predction)
                loss = tf.reduce_mean(loss,axis=-1)
                tf.add_to_collection('losses',loss)
            self.losses = tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('loss',self.losses)

        with tf.name_scope('acc'):
            for i in range(8):
                predction = pre[i]
                labbel = tf.reduce_mean(tf.slice(self.batch_label, [0, i], [-1, 1]), axis=-1)
                max_index = tf.argmax(predction, axis=1)
                cast_value = tf.cast(tf.equal(max_index, labbel), dtype=tf.float32)
                self.acc = tf.reduce_mean(cast_value,axis=-1)
                tf.add_to_collection('acc',self.acc)
                tf.summary.scalar('acc'+str(i),self.acc)
            self.accur = tf.get_collection('acc')




# model = Model()
# init_op=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     sess.run(model.batch_tran)