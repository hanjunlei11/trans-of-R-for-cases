import jieba
import math
from config import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_dic(filename):
    dic2count = {}
    file = open('./' + filename,encoding='UTF-8')
    s = file.readlines()
    ls = []
    count = 0
    count_zm = 0
    zm2index = {}
    for line in s:
        s11 =line.split('     ')
        s1 = s11[0].split()
        for word in s1:
            if word in dic2count:
                dic2count[word] += 1
            else:
                dic2count[word] = 1
        count+=1
        s2 = s11[1].split()
        if s2[3] in zm2index:
            count_zm+=0
        else:
            zm2index[s2[3]]=count_zm
            count_zm+=1
    for item in dic2count.items():
        if item[1] >= 5:
            ls.append(item[0])
    lsindex = list(range(len(ls)))
    word2index = dict(zip(ls, lsindex))
    return word2index,zm2index

def getlength(filename):
    file = open('./' + filename)
    s = file.read()
    ls = s.split(' ')
    ls=list(map(int,ls))
    return ls

def get_data(filename,avaage,devage):
    file=open('./'+filename, encoding='UTF-8')
    lines=file.readlines()
    data=[]
    zuiming_nunber = {}
    for i in lines:
        ss_attr_label=i.split('     ')
        ss=ss_attr_label[0].split()
        ls=len(ss)
        if ls>=400:
            ls=400
        attr_label=ss_attr_label[1].split()
        n_x_l = list(map(float,attr_label[:3]))
        n_x_l[0]=(n_x_l[0]-avaage)/devage
        n_x_l[1]= n_x_l[1]/1.0
        n_x_l[2]= n_x_l[2]/1.0
        if attr_label[3] in zuiming_nunber:
            zuiming_nunber[attr_label[3]]+=1
        else:
            zuiming_nunber[attr_label[3]]=1

        attrs=list(map(float,attr_label[3:24]))
        xq=[float(attr_label[-1])]
        data.append([ss,n_x_l,attrs,xq,ls])
    train_data, test_data = train_test_split(np.asarray(data),test_size=0.3,)
    # print(len(train_data))
    # print(len(test_data))
    return train_data, test_data, zuiming_nunber

def get_batch(bachsize,data,zm2index,word2index,truncate_l):
    random_int=np.random.randint(0,len(data),bachsize)
    data=np.asarray(data)
    batch_data=data[random_int]
    batch_index4ret=[]
    batch_inf = []
    batch_c = []
    batch_r = []
    zm_label = []
    for i in batch_data:
        sc=i[0]
        scc=i[1]
        sccc=i[2]
        scccc=i[3]
        tl4index=[0]*truncate_l
        batch_inf_1 = [0]*3
        batch_c_1 = [0]*21
        batch_r_1 = [0]
        zm = [0]
        count=0
        for j in sc:
            if j in word2index:
                tl4index[count]=word2index[j]
            count+=1
            if count==truncate_l:
                break
        for j in range(len(scc)):
            batch_inf_1[j]=scc[j]
        for j in range(len(sccc)):
            if j==0:
                zm = zm2index[str(sccc[j])]
                batch_c_1[j]=zm
            else:
                batch_c_1[j]=sccc[j]
        for j in range(len(scccc)):
            batch_r_1[j]=math.ceil(scccc[j]/3)
        batch_index4ret.append(tl4index)
        batch_inf.append(batch_inf_1)
        batch_c.append(batch_c_1)
        batch_r.append(batch_r_1)
        zm_label.append(zm)
    length = batch_data[:,4]
    return batch_index4ret,batch_inf,batch_c,batch_r,zm_label,length

def get_test_data(bachsize,index,data,zm2index,word2index,truncate_l):
    batch_data=data[bachsize*index:bachsize*index+127]
    batch_index4ret=[]
    batch_inf = []
    batch_c = []
    batch_r = []
    zm_label = []
    for i in batch_data:
        sc=i[0]
        scc=i[1]
        sccc=i[2]
        scccc=i[3]
        tl4index=[0]*truncate_l
        batch_inf_1 = [0]*3
        batch_c_1 = [0]*21
        batch_r_1 = [0]
        zm = [0]
        count=0
        for j in sc:
            if j in word2index:
                tl4index[count]=word2index[j]
            count+=1
            if count==truncate_l:
                break
        for j in range(len(scc)):
            batch_inf_1[j]=scc[j]
        for j in range(len(sccc)):
            if j==0:
                zm = zm2index[str(sccc[j])]
                batch_c_1[j]=zm
            else:
                batch_c_1[j]=sccc[j]
        for j in range(len(scccc)):
            batch_r_1[j]=math.ceil(scccc[j]/3)
        batch_index4ret.append(list(tl4index))
        batch_inf.append(batch_inf_1)
        batch_c.append(batch_c_1)
        batch_r.append(batch_r_1)
        zm_label.append(zm)
    length = batch_data[:,4]
    return batch_index4ret,batch_inf,batch_c,batch_r,zm_label,length


# train_data, test_data, zuiming_nunber = get_data('./train_all.txt',avaage=ava_age,devage=dev_age)