from Transformer import *
from tool import *
import tensorflow as tf
from config import *

model=Model()
print('1、构造模型完成')
opt_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
word2index, zm2index=get_dic(filename="train_all.txt")
train_data, test_data, zuiming_number =get_data('./train_all.txt',avaage=ava_age,devage=dev_age)
init_op=tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    print('2、初始化完成')
    mean_zm_acc = 0
    mean_xq_acc = 0
    print('3、开始训练')
    for i in range(10000):
        batch_date, batch_inf, batch_c,zm_train_label, input_length = get_batch(bachsize=batch_size,
                                                                                          data=train_data,
                                                                                          zm2index=zm2index,
                                                                                          word2index=word2index,
                                                                                          truncate_l=truncate_l)
        feed_dic = {model.fzss: batch_date, model.add_inf: batch_inf, model.all_label_c: batch_c,
                     model.test_zuiming_label: zm_train_label,
                    model.input_length: input_length}
        _, loss, acc= sess.run([opt_op, model.loss, model.acc], feed_dict=feed_dic)
        print('loss:',loss,'  ','zm_acc:',acc)
        saver.save(sess, './mymodel_3/MyModel')

        if (i+1)%100==0:
            batch_date, batch_inf, batch_c, batch_r, zm_test_label, input_length = get_batch(bachsize=batch_size,
                                                                                             data=test_data,
                                                                                             zm2index=zm2index,
                                                                                             word2index=word2index,
                                                                                             truncate_l=truncate_l)
            feed_dic = {model.fzss: batch_date, model.add_inf: batch_inf, model.all_label_c: batch_c,
                        model.test_zuiming_label: zm_test_label,
                        model.input_length: input_length}
            zm_acc, acc = sess.run([model.acc], feed_dict=feed_dic)
            print('zm_acc:',zm_acc,'   ','xq_acc:',acc)

    for i in range(600):
        batch_date, batch_inf, batch_c, batch_r, zm_test_label, input_length = get_test_data(bachsize=batch_size,
                                                                                             index=i,
                                                                                             data=test_data,
                                                                                             zm2index=zm2index,
                                                                                             word2index=word2index,
                                                                                             truncate_l=truncate_l)
        feed_dic = {model.fzss: batch_date, model.add_inf: batch_inf, model.all_label_c: batch_c,
                    model.test_zuiming_label: zm_test_label, model.input_length: input_length}
        zm_acc, xq_acc = sess.run([model.acc], feed_dict=feed_dic)
        mean_zm_acc+= zm_acc/600
        mean_xq_acc += xq_acc/600
    print('mean_zm_acc: ', mean_zm_acc,'   ','mean_xq_acc:',mean_xq_acc)
