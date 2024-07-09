import torch
import torch.nn.functional as F
import timeit
import numpy as np

def loss_maxcut_weighted(probs, C, weights, penalty_inc, penalty_c, hyper):
    #print(weights)
    x = probs.squeeze()
    loss = 0
    penalty=0
    #print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            # indices = [i - 1 for i in c]
            # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
            for index in c:
                temp_1s *= (1 - x[index-1])
                temp_0s *= (x[index-1])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[index-1])
                temp_0s *= (x[index-1])
        temp = (temp_1s + temp_0s)

        #print(temp_1s, temp_0s, w)
        #temp = (temp_1s + temp_0s - 1)

        #print(c, temp)
        loss += (temp * w)
        #print(loss)
    if penalty_inc:
        penalty = torch.sum(torch.min((1 - x), x))
        loss += penalty_c * penalty
    return loss

def loss_maxcut_weighted_anealed(probs, C, dct, weights,  temper, hyper=False):
    #print(weights)
    x = probs.squeeze()
    loss = 0
    #print('-----------------')
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            for index in c:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        #print(temp_1s, temp_0s, w)
        #temp = (temp_1s + temp_0s - 1)
        temp = (temp_1s + temp_0s)
        #print(c, temp)
        loss += (temp * w)
        #print(loss)
    Entropy=sum([item* torch.log2(item)+(1-item)*torch.log2(1-item) for item in x])
    loss+= temper*Entropy
    return loss