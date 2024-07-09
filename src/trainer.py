from src.model import single_node, single_node_xavier, HGNN_ATT
import timeit
from itertools import chain
import torch
from src.timer import Timer
from src.loss import loss_cal_and_update, maxcut_loss_func_helper, loss_maxcut_weighted, loss_sat_weighted, loss_maxind_weighted, loss_maxind_QUBO, loss_maxind_weighted2, loss_task_weighted, loss_maxcut_weighted_anealed, loss_task_weighted_vec, loss_mincut_weighted
from src.utils import mapping_algo, mapping_distribution, gen_q_mis,gen_q_maxcut, mapping_distribution_QUBO, get_normalized_G_from_con, mapping_distribution_vec_task
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from torch.autograd import grad
import pickle

def centralized_train(G, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    ####### fix seed to ensure consistent results ######
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG

    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    #### sometimes we want the number of epochs to grow with n #####
    # rounds = max(int(2 * n // 10), int(params['epoch']))
    rounds = int(params['epoch'])
    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]

    ### q_torch helps compute graph MIS and Maxcut loss faster
    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    elif params['mode'] == 'QUBO_maxcut':
        q_torch = gen_q_maxcut(C, n, torch_dtype=None, torch_device=None)

    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}

    ##### transfer learning: load and freeze the layers and only optimize on the input embeding ######
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        # embed = nn.Embedding(n, f)
        # embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # # for param in embed.parameters():
        #     param.requires_grad = False
        name=params["model_load_path"]+'conv1_'+file_name[:-4]+'.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"]+'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # conv1 = single_node(f, f//2)
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 1)
        # conv2 = single_node(f//2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())

    ###### if we want to initialize our model and input embedding by a pretrained model ####
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())

    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    #inputs = embed.weight
    #grad1=torch.zeros((int(params['epoch'])))
    #grad2 = torch.zeros((int(params['epoch'])))

    #### computes the distance between node features at each layer to detect oversmoothing ####
    dist=[]

    for i in range(rounds):

        ##### forward path ######
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        dis1=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        dis2=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.relu(temp)
        temp = conv2(temp)
        dis3 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        dis4 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.sigmoid(temp)
        #temp = torch.softmax(temp, dim=0)
        dist.append([dis1,dis2,dis3,dis4])

        ###### compute the loss #######
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])

        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], indicest, params['hyper'])

        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])

        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)

        elif params['mode'] == 'QUBO_maxcut':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            #loss2 = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
            # print(loss,loss2)

        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])

        elif params['mode'] == 'task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
            if loss==0:
                print("found zero loss")
                break

        elif params['mode'] == 'mincut':
            loss = loss_mincut_weighted(temp, C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'],
                                        indicest, params['hyper'])


        ###### optimization step #######
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()


        ##### decide if we want to stop based on tolerance (params['tol']) and patience (params['patience']) ######
        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')

                #### save the model #####
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        else:
            count = 0

        #### keep the best loss and result ####3
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')

            ##### the end of the epochs #####
            if i==int(params['epoch'])-1:
                ##### save the model #####
                name=params["model_save_path"]+'embed_'+file_name[:-4]+'.pt'
                torch.save(embed, name)
                name = params["model_save_path"]+'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"]+'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                ##### save the model #####
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss=loss

    # with open("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/dist_"+file_name[:-4]+".pkl", "wb") as fp:
    #     pickle.dump(dist, fp)


    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    all_weights = [1.0 for c in (C)]

    #### plot the histogram of the HyperGNN output (see if it's learning anything) #####
    name = params['plot_path']+ file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()
    train_time = timeit.default_timer() - temp_time

    ##### fine-tuning ######
    temp_time2 = timeit.default_timer()
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2

    # if params['mode'] != 'QUBO':
    #     res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'], params['hyper'])
    # else:
    #     res = mapping_distribution_QUBO(best_out, params, q_torch, n)
    # params2=params
    # params2['Niter_h']=100
    # params2['N_realize'] = 2
    # if params['mode'] != 'QUBO':
    #     res2 = mapping_distribution(best_out, params2, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    # else:
    #     res2 = mapping_distribution_QUBO(best_out, params2, q_torch, n)
    # return res, res2, best_out
    return res, best_out, train_time, map_time


##### for multi-gpu (distributed) training #####
def centralized_train_for(X, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}
    X = torch.cat([X[i] for i in X])
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        # embed = nn.Embedding(n, f)
        # embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # # for param in embed.parameters():
        #     param.requires_grad = False
        name=params["model_load_path"]+'conv1_'+file_name[:-4]+'.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"]+'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # conv1 = single_node(f, f//2)
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 1)
        # conv2 = single_node(f//2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    inputs = embed.weight
    #grad1=torch.zeros((int(params['epoch'])))
    #grad2 = torch.zeros((int(params['epoch'])))
    for i in range(int(params['epoch'])):
        print(i)
        temp = conv1(inputs)
        temp2=torch.zeros(temp.shape)
        for i in range(1,n+1):
            for cons in info[i]:
                if cons[0]==i:
                    temp2[i-1,:]+=temp[cons[1]-1,:]
                else:
                    temp2[i-1,:] += temp[cons[0]-1,:]
            temp2[i-1,:]=temp2[i-1,:]/len(info[i])
        temp=temp2
        #temp = G @ temp
        temp = torch.relu(temp)
        temp = conv2(temp)
        temp2 = torch.zeros(temp.shape)
        for i in range(1,n+1):
            for cons in info[i]:
                if cons[0] == i:
                    temp2[i-1,:] += temp[cons[1]-1,:]
                else:
                    temp2[i-1,:] += temp[cons[0]-1,:]
            temp2[i-1,:] = temp2[i-1,:] / len(info[i])
        temp = temp2
        #temp = G @ temp
        temp = torch.sigmoid(temp)
        #temp = torch.softmax(temp, dim=0)
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)
        elif params['mode']=='task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            if i==int(params['epoch'])-1:
                name=params["model_save_path"]+'embed_'+file_name[:-4]+'.pt'
                torch.save(embed, name)
                name = params["model_save_path"]+'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"]+'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss=loss

    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    name = './res/plots/Hist_HypOp_QUBO_Maxind/Hist_' + file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2
    # if params['mode'] != 'QUBO':
    #     res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'], params['hyper'])
    # else:
    #     res = mapping_distribution_QUBO(best_out, params, q_torch, n)
    # params2=params
    # params2['Niter_h']=100
    # params2['N_realize'] = 2
    # if params['mode'] != 'QUBO':
    #     res2 = mapping_distribution(best_out, params2, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    # else:
    #     res2 = mapping_distribution_QUBO(best_out, params2, q_torch, n)
    # return res, res2, best_out
    return res, best_out, train_time, map_time


##### gradient descent solver (no HyperGNN) ####
def GD_train(params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32
    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}


    embed = nn.Embedding(n, 1)

    parameters = embed.parameters()
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    for i in range(int(params['epoch'])):
        print(i)
        inputs = embed.weight
        temp = torch.sigmoid(inputs)
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C,  [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'],indicest, params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)

        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                break
        prev_loss=loss

    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    name = params["plot_path"]+'Hist_' + file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2
    # if params['mode'] != 'QUBO':
    #     res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'], params['hyper'])
    # else:
    #     res = mapping_distribution_QUBO(best_out, params, q_torch, n)
    # params2=params
    # params2['Niter_h']=100
    # params2['N_realize'] = 2
    # if params['mode'] != 'QUBO':
    #     res2 = mapping_distribution(best_out, params2, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    # else:
    #     res2 = mapping_distribution_QUBO(best_out, params2, q_torch, n)
    # return res, res2, best_out
    return res, best_out, train_time, map_time

def centralized_train_vec_task(G, params, f, C, n, info, weights, file_name, C_dic, lenc, leninfo):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32


    L=len(C)
    #f = n // 4
    f = 10

    # C_mat=np.zeros([n,L])
    # for c in C_dic.keys():
    #     for i in c:
    #         C_mat[i-1, C_dic[str(c)]]=1

    temper0 = 0.01
    p = 0
    count = 0
    prev_loss = 100
    patience = params['patience']
    best_loss = float('inf')
    dct = {x + 1: x for x in range(n)}
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        # embed = nn.Embedding(n, f)
        # embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # # for param in embed.parameters():
        #     param.requires_grad = False
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, L*f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # conv1 = single_node(f, f//2)
        conv1 = single_node_xavier(L*f, L*f // 2)
        conv2 = single_node_xavier(L*f // 2, L)
        # conv2 = single_node(f//2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr=params['lr'])
    # inputs = embed.weight
    # grad1=torch.zeros((int(params['epoch'])))
    # grad2 = torch.zeros((int(params['epoch'])))
    for i in range(int(params['epoch'])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        temp = G @ temp
        temp = torch.relu(temp)
        temp = conv2(temp)
        temp = G @ temp
        temp = torch.sigmoid(temp)
        # temp = torch.softmax(temp, dim=0)
        if params['mode'] == 'task_vec':
            loss = loss_task_weighted_vec(temp, lenc, leninfo)
            if loss == 0:
                print("found zero loss")
                break
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            if i == int(params['epoch']) - 1:
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss = loss

    best_out = best_out.detach().numpy()
    best_out_d = {i + 1: best_out[i,:] for i in range(len(best_out))}
    train_time = timeit.default_timer() - temp_time
    temp_time2 = timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    #name = params['plot_path'] + file_name[:-4] + '.png'
    #plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    #plt.savefig(name)
    #plt.show()
    leninfon=torch.Tensor.numpy(leninfo)
    lencn=torch.Tensor.numpy(lenc)
    best_res = mapping_distribution_vec_task(best_out_d, params, n, info, weights, C, C_dic, all_weights, 1, lencn,leninfon,params['penalty'],
                               params['hyper'])
    map_time = timeit.default_timer() - temp_time2

    return best_res, best_out, train_time, map_time




def centralized_train_att( H, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()

    # fix seed to ensure consistent results
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG

    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]


    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}


    ### have not fixed the transfer learning for ATT ####
    if params['transfer']:
        name = params["model_load_path"] + 'ATT.pt'
        model_att = torch.load(name)
        for param in model_att.parameters():
            param.requires_grad = False

    #### define the HyperGAT model ####
    else:
        model_att=HGNN_ATT(n, f, 3*f//4, 1, params)


    if params["initial_transfer"]:
        name = params["model_load_path"] + 'ATT.pt'
        model_att = torch.load(name)


    dist=[]
    for i in range(int(params['epoch'])):
        print(i)

        #### forward path ####
        temp=model_att(torch.Tensor(H).float())


        ##### calculate the loss #####
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])

        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C,  [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], indicest, params['hyper'])

        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])

        # elif params['mode'] == 'QUBO':
        #     # probs = temp[:, 0]
        #     loss = loss_maxind_QUBO(temp, q_torch)
        #     # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
        #     # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        #     # print(loss, loss3)
        # elif params['mode'] == 'QUBO_maxcut':
        #     # probs = temp[:, 0]
        #     loss = loss_maxind_QUBO(temp, q_torch)
        #     #loss2 = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
        #     # print(loss,loss2)

        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])

        elif params['mode'] == 'task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
            if loss==0:
                print("found zero loss")
                break

        ##### optimization step ####
        model_att.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        model_att.optimizer.step()

        ##### decide if we want to stop based on tolerance (params['tol']) and patience (params['patience']) ######
        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')

                ### if we want to save the model ###
                # name = params["model_save_path"] + 'ATT.pt'
                # torch.save(model_att, name)
                break
        else:
            count = 0

        #### keep the best loss and result ####
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            # if i==int(params['epoch'])-1:
                ### if we want to save the model ###
                # name = params["model_save_path"] + 'ATT.pt'
                # torch.save(model_att, name)

        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                ### if we want to save the model ###
                # name = params["model_save_path"] + 'ATT.pt'
                # torch.save(model_att, name)
                break
        prev_loss=loss

    # with open("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/dist_"+file_name[:-4]+".pkl", "wb") as fp:
    #     pickle.dump(dist, fp)


    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    train_time = timeit.default_timer() - temp_time

    all_weights = [1.0 for c in (C)]

    #### plot the histogram of the HyperGNN output to see if it's learning anything ####
    name = params['plot_path']+ file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()

    ##### fine-tuning ####
    temp_time2 = timeit.default_timer()
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2

    # if params['mode'] != 'QUBO':
    #     res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'], params['hyper'])
    # else:
    #     res = mapping_distribution_QUBO(best_out, params, q_torch, n)
    # params2=params
    # params2['Niter_h']=100
    # params2['N_realize'] = 2
    # if params['mode'] != 'QUBO':
    #     res2 = mapping_distribution(best_out, params2, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    # else:
    #     res2 = mapping_distribution_QUBO(best_out, params2, q_torch, n)
    # return res, res2, best_out
    return res, best_out, train_time, map_time



###### bipartite GNN #####
def centralized_train_bipartite( G, params, f, C, n, n_hyper, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    # if params['mode'] == 'QUBO':
    #     q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    # elif params['mode'] == 'QUBO_maxcut':
    #     q_torch = gen_q_maxcut(C, n, torch_dtype=None, torch_device=None)
    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]

    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')

    if params['transfer']: #not updated for bipartite
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        # embed = nn.Embedding(n, f)
        # embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # # for param in embed.parameters():
        #     param.requires_grad = False
        name=params["model_load_path"]+'conv1_'+file_name[:-4]+'.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"]+'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)

        #4 layers
        # conv1 = single_node_xavier(f, f)
        # conv2 = single_node_xavier(f, f // 2)
        # conv3 = single_node_xavier(f // 2, f // 2)
        # conv4 = single_node_xavier(f // 2, 1)
        # parameters = chain(conv1.parameters(), conv2.parameters(), conv3.parameters(), conv4.parameters(), embed.parameters())

        #two layers
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(),embed.parameters())

    if params["initial_transfer"]: #not updated for bipartite
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    #inputs = embed.weight
    #grad1=torch.zeros((int(params['epoch'])))
    #grad2 = torch.zeros((int(params['epoch'])))
    dist=[]
    for i in range(int(params['epoch'])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        # dis1=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        # dis2=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.relu(temp)
        temp = conv2(temp)
        # dis3 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp


        # 4 layers
        # temp = torch.relu(temp)
        # temp = conv3(temp)
        # temp = G @ temp
        # temp = torch.relu(temp)
        # temp = conv4(temp)
        # temp = G @ temp



        # dis4 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.sigmoid(temp)
        #temp = torch.softmax(temp, dim=0)
        # dist.append([dis1,dis2,dis3,dis4])
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp[0:n_hyper], C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], indicest, params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)
        elif params['mode'] == 'QUBO_maxcut':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            #loss2 = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
            # print(loss,loss2)
        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])
        elif params['mode'] == 'task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
            if loss==0:
                print("found zero loss")
                break
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            if i==int(params['epoch'])-1:
                name=params["model_save_path"]+'embed_'+file_name[:-4]+'.pt'
                torch.save(embed, name)
                name = params["model_save_path"]+'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"]+'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss=loss
    # with open("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/dist_"+file_name[:-4]+".pkl", "wb") as fp:
    #     pickle.dump(dist, fp)
    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(n_hyper)}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    name = params['plot_path']+ file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()
    res = mapping_distribution(best_out, params, n_hyper, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2

    return res, best_out, train_time, map_time
