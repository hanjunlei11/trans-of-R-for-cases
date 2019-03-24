from config import *
from tool import *
import tensorflow as tf
from function import *
from transformer import *
Model = Model()
print('构造模型完成')
train(Model=Model,datafile='./data.txt',labelfile='./shuxing.txt',epoch_number=20,keep_rate=0.8,is_trainning=True)