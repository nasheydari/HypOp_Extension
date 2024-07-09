from hypop.utils import generate_H_from_edges, _generate_G_from_H, generate_H_from_constraints, all_to_weights, all_to_weights_task, gen_q_mis, get_normalized_G_from_con, Maxind_postprocessing, sparsify_graph
import numpy as np
import torch
from hypop.params import Params
import timeit
from hypop.trainer import  centralized_train, GD_train, centralized_train_for, centralized_train_vec_task, centralized_train_att, centralized_train_bipartite
from hypop.loss import loss_maxcut_numpy_boost
import matplotlib.pyplot as plt
# import dgl
from hypop.data_reading import read_hypergraph
import os

#### main solver #### At its current state this does assume ti is not bipartite
def centralized_solver(constraints, header, params: Params, file_name):
    temp_time = timeit.default_timer()

    n = header['num_nodes']

    q_torch = gen_q_mis(constraints, n, 2, torch_dtype=None, torch_device=None)

    f = int(np.sqrt(n)) if not params.f_input else params.f

    info = {x + 1: [] for x in range(n)}
    for constraint in constraints:
        for node in constraint:
            info[abs(node)].append(constraint)
    
    if params.sparsify:
        constraints, header, info_sparse = sparsify_graph(constraints, header, info, params.spars_p)

    edges = [[abs(x) - 1 for x in edge] for edge in constraints]

    #### if params['random_init']==true, it means we are not going to use the HyperGNN training results and use SA with random initialization ######
    
    # TODO Implement saving and loading G matrix
    H = generate_H_from_edges(edges, n)
    G = _generate_G_from_H(H)
    G = torch.from_numpy(G).float()

    all_weights = [1.0 for c in (constraints)]
    weights = all_to_weights(all_weights, n, constraints)

    #initialization
    reses = []
    reses_th = []
    probs = []
    train_times = []
    map_times = []

    for i in range(params.K): 

        if not params.GD and not params.Att: ####### Main Trainer #######
            res,  prob , train_time, map_time= centralized_train( G, params, f, constraints, n, info, weights, file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        elif params.Att: ###### Trainer with Hypergraph Attention Network ######
            res, prob, train_time, map_time = centralized_train_att(H, params, f, constraints, n, info, weights, file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        else: ###### Gradient Descent Solver (no HyperGNN) #####
            res, prob, train_time, map_time = GD_train(params, f, constraints, n, info, weights, file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        res_th = {x: 0 if prob[x] < 0.5 else 1 for x in prob.keys()}

        score, new_w = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params.boosting_mapping)
        score_th, _ =  loss_maxcut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))], inc=params.boosting_mapping)

        probs.append(prob)
        reses.append(score)
        reses_th.append(score_th)

    return reses, reses_th, probs, timeit.default_timer() - temp_time, train_times, map_times
