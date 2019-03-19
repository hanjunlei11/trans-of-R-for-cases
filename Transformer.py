import tensorflow as tf
from Multi_Head_Attention import *
from config import *

class Model():
    def __init__(self):
        self.truncate_l = truncate_l
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.zuiming_number = zuiming_number
        self.keep_rate = keep_rate
        self.L2_reregularizer = tf.contrib.layers.l2_regularizer(10e-6)
        self.L2_reregularizetion = 0.1

        with tf.name_scope("input"):
            self.test_zuiming_label = tf.placeholder(shape=(None,), dtype=tf.int64)
            self.fzss = tf.placeholder(shape=(None, self.truncate_l), dtype=tf.int64)
            self.add_inf = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            self.all_label_c = tf.placeholder(shape=(None, 21), dtype=tf.int64)
            self.input_length = tf.placeholder(tf.int64, (None))
            self.embedding_keep_rate = tf.placeholder(shape=(1,),dtype=tf.float32)
            self.is_traning = tf.placeholder(shape=(1,),dtype=tf.bool)

        with tf.name_scope('embedding_layer_train'):
            embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], dtype=tf.float32),
                                          trainable=True, name='word_embedding')
            s1_matrix_tr = tf.nn.embedding_lookup(embedding_table, self.fzss)
            s1_matrix_tr = tf.nn.dropout(s1_matrix_tr, keep_prob=self.embedding_keep_rate)


        with tf.name_scope('transformer'):
            self.batch_s1 = encoder(s1_matrix_tr, embedding_size=self.hidden_size, nb_layers=6,nb_head=4, size_per_head=64,
                                    keep_rate=self.keep_rate, training=self.is_traning)

        with tf.name_scope('Attention'):
            output_concate=tf.keras.layers.Dense(self.hidden_size*2)(self.batch_s1)
            attention_list=[]
            for i in range(20):
                att_vec=tf.get_variable(name='att_vec'+str(i),shape=(1,self.hidden_size*2),dtype=tf.float32,trainable=True)
                att_rate_vec=tf.reduce_sum(tf.multiply(output_concate,att_vec),keep_dims=True,axis=2)
                att_rate_vec=tf.nn.softmax(att_rate_vec)
                att_sum=tf.reduce_sum(tf.multiply(output_concate,att_rate_vec),axis=1)
                attention_list.append(att_sum)

        with tf.name_scope('classifier'):
            self.output_list = []
            for i in range(20):
                layer1 = tf.get_variable(name='c'+str(i+2)+'layer1', shape=(self.hidden_size*2, 40), dtype=tf.float32)
                b1 = tf.get_variable(name='c'+str(i+2)+'b1', shape=(40), dtype=tf.float32)
                output_layer1 = tf.nn.leaky_relu(tf.nn.xw_plus_b(attention_list[i], layer1, b1))
                layer2 = tf.get_variable(name='c'+str(i+2)+'layer2', shape=(40, 2), dtype=tf.float32)
                b2 = tf.get_variable(name='c'+str(i+2)+'b2' + str(i), shape=(2), dtype=tf.float32)
                output_layer2 = tf.nn.leaky_relu(tf.nn.xw_plus_b(output_layer1, layer2, b2))
                self.output_list.append(output_layer2)

        with tf.name_scope('loss'):
            for i in range(20):
                label=tf.slice(self.all_label_c,[0,i],[self.batch_size,1])
                label=tf.reshape(label,[label.shape[0]])
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_list[i],labels=label))
                tf.add_to_collection('losses',loss)
            losses=tf.add_n(tf.get_collection('losses'))
            self.loss=losses

        with tf.name_scope('acc'):
            max_index=tf.argmax(self.output_list[0],axis=1)
            equal_zhi = tf.cast(tf.equal(max_index, self.test_zuiming_label), dtype=tf.float32)
            self.acc =tf.reduce_mean(equal_zhi,axis=-1)


# model = Model()
# init_op=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     sess.run(model.s2_char_conv_out)