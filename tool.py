import numpy as np
import collections
from config import *
def get_data(filepath):
    with open(filepath,'r',encoding='utf-8') as data_all,open('./word2index.txt','w+',encoding='utf-8') as word2index,open('./data.txt','w+',encoding='utf-8') as data_s,open('./shuxing.txt','w+',encoding='utf-8') as shuxing:
        data = data_all.readlines()
        dic = collections.OrderedDict()
        for line in data:
            sx = []
            fzss, temp = line.split('     ')
            fzss = fzss.split(' ')
            label = temp.split(' ')
            if int(float(label[3]))!=216:
                continue
            for word in fzss:
                if word not in dic:
                    dic[word] = 1
                else:
                    dic[word]+=1
            sx.append(label[0])
            sx.append(label[1])
            sx.append(label[2])
            sx.append(label[6])
            sx.append(label[10])
            sx.append(label[11])
            sx.append(label[15])
            sx.append(label[16])
            sx.append(label[20])
            sx.append(label[21])
            sx.append(label[23])
            shuxing.write(' '.join(sx)+'\n')

        word_index = collections.OrderedDict()
        word_index['unk'] = 0
        for item in dic.items():
            if item[1]>deadline:
                word_index[item[0]] = len(word_index)

        for item in word_index.items():
            word2index.write(str(item[0])+': '+str(item[1])+'\n')
        print(len(dic))
        print(len(word_index))
        all_len = 0
        count = 0
        for line in data:
            case = []
            fzss, temp = line.split('     ')
            fzss = fzss.split(' ')
            label = temp.split(' ')
            if int(float(label[3])) != 216:
                continue
            all_len += len(fzss)
            count += 1
            for word in fzss:
                if word in word_index:
                    case.append(str(word_index[word]))
                else:
                    case.append(str(word_index['unk']))
            data_s.write(' '.join(case)+'\n')
        print(int(all_len/count))
        print(count)
        print('word_done')

def read_file(datafile,labelflie):
    with open(datafile,'r',encoding='utf-8') as data,open(labelflie,'r',encoding='utf-8') as label:
        data_lines = data.readlines()
        label_lines = label.readlines()
        data_all = []
        data_r = []
        data_label = []
        for i in range(len(label_lines)):
            line = [0]*data_len
            temp1 = data_lines[i].strip().split(' ')
            for j in range(len(temp1)):
                if j <data_len:
                    line[j]=int(temp1[j])
            data_all.append(line)
            temp2 = list(map(int,map(float,label_lines[i].strip().split(' '))))
            data_r.append(temp2[:3])
            data_label.append(temp2[3:])
        data_all = np.asarray(data_all)
        data_r = np.asarray(data_r)
        data_label = np.asarray(data_label)
        data_all_train = data_all[:int(0.8*len(data_all))]
        data_all_test = data_all[int(0.8 * len(data_all)):]
        data_r_train = data_r[:int(0.8*len(data_r))]
        data_r_test = data_r[int(0.8 * len(data_r)):]
        data_label_train = data_label[:int(0.8*len(data_label))]
        data_label_test = data_label[int(0.8 * len(data_label)):]
        return data_all_train,data_all_test,data_r_train,data_r_test,data_label_train,data_label_test

def get_epoch(batch_size,data_all,data_r,data_label):
    epoch_fzss = []
    epoch_r = []
    epoch_label = []
    for i in range(int(len(data_r)/batch_size)):
        epoch_fzss.append(data_all[batch_size*i:batch_size*(i+1)])
        epoch_r.append(data_r[batch_size*i:batch_size*(i+1)])
        epoch_label.append(data_label[batch_size*i:batch_size*(i+1)])
    epoch_fzss = np.asarray(epoch_fzss)
    epoch_r = np.asarray(epoch_r)
    epoch_label = np.asarray(epoch_label)
    return epoch_fzss,epoch_r,epoch_label,len(epoch_label)

def get_batch(epoch_fzss,epoch_r,epoch_label,i):
    return epoch_fzss[i],epoch_r[i],epoch_label[i]


# get_data('./data_all.txt')
# data_all_train,data_all_test,data_r_train,data_r_test,data_label_train,data_label_test = read_file('./data.txt','./shuxing.txt')
# epoch_fzss,epoch_r,epoch_label,len = get_epoch(batch_size,data_all_train,data_r_train,data_label_train)
# batch_fzss,batch_r,batch_label = get_batch(epoch_fzss,epoch_r,epoch_label,4)