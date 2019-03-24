import collections
import tensorflow as tf


def Permutation(ss):
    # write code here
    if len(ss) == 0:
        return []
    re = [ss]
    len_ss = len(ss)
    for i in range(len_ss):
        for j in range(len_ss - 1):
            if ss[i] != ss[j + 1]:
                temp = ss[i]
                ss[i] = ss[j + 1]
                ss[j + 1] = temp
            if ss not in re:
                re.append(ss)
    re.reverse()
    return re
print(Permutation(input()))
