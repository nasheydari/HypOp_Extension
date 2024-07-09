from src.data_reading import read_uf, read_stanford, read_hypergraph, read_hypergraph_task, read_NDC
from src.solver import  centralized_solver#, centralized_solver_for
import logging
import os
import h5py
import numpy as np

import timeit


def exp_centralized(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                temp_time = timeit.default_timer()
                if params['data'] == "uf":
                    constraints, header = read_uf(path)
                elif params['data'] == "stanford" or params['data'] == "random_reg" or params['data'] == "bipartite" :
                    constraints, header = read_stanford(path)
                elif params['data'] == "hypergraph":
                    constraints, header = read_hypergraph(path)
                elif params['data'] == "task":
                    constraints, header = read_hypergraph_task(path)
                elif params['data'] == "NDC":
                    constraints, header = read_NDC(path)
                else:
                    log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

                res, res_th, probs, total_time, train_time, map_time = centralized_solver(constraints, header, params, file_name)

                #name= 'probs'+'_'+file_name+'.txt'
                #np.savetxt(name, probs)
                time = timeit.default_timer() - temp_time
                log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, training_time: {train_time}, mapping_time: {map_time}')
                print(res)
                print(res_th)
                f.create_dataset(f"{file_name}", data = res)






# def exp_centralized_for(params):
#     logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
#     log = logging.getLogger('main')
#     folder_path = params['folder_path']
#     folder_length = len(os.listdir(folder_path))
#     print(f'Found {folder_length} files. Start experiments')
#     with h5py.File(params['res_path'], 'w') as f:
#         for file_name in os.listdir(folder_path):
#             if not file_name.startswith('.'):
#                 print(f'dealing {file_name}')
#                 path = folder_path + file_name
#                 temp_time = timeit.default_timer()
#                 if params['data'] == "uf":
#                     constraints, header = read_uf(path)
#                 elif params['data'] == "stanford" or params['data'] == "random_reg":
#                     constraints, header = read_stanford(path)
#                 elif params['data'] == "hypergraph":
#                     constraints, header = read_hypergraph(path)
#                 else:
#                     log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

#                 res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_for(constraints, header, params, file_name)

#                 #name= 'probs'+'_'+file_name+'.txt'
#                 #np.savetxt(name, probs)
#                 time = timeit.default_timer() - temp_time
#                 log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
#                 print(np.average(res))
#                 print(np.average(res_th))
#                 if params['mode']=='maxind':
#                     N = 200
#                     print((np.average(res)) / (N*0.45537))
#                 f.create_dataset(f"{file_name}", data = res)




# def exp_centralized_watermark(params):
#     logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
#     log = logging.getLogger('main')
#     folder_path = params['folder_path']
#     folder_length = len(os.listdir(folder_path))
#     print(f'Found {folder_length} files. Start experiments')
#     with h5py.File(params['res_path'], 'w') as f:
#         for file_name in os.listdir(folder_path):
#             if not file_name.startswith('.'):
#                 print(f'dealing {file_name}')
#                 path = folder_path + file_name
#                 temp_time = timeit.default_timer()
#                 if params['data'] == "uf":
#                     constraints, header = read_uf(path)
#                 elif params['data'] == "stanford" or params['data'] == "random_reg" or params['data'] == "bipartite" :
#                     constraints, header = read_stanford(path)
#                 elif params['data'] == "hypergraph":
#                     constraints, header = read_hypergraph(path)
#                 elif params['data'] == "task":
#                     constraints, header = read_hypergraph_task(path)
#                 elif params['data'] == "NDC":
#                     constraints, header = read_NDC(path)
#                 else:
#                     log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

#                 wat_len=16
#                 wat_type='MIS'
#                 wat_seed_value=100
#                 watermark_cons, watermark_nodes =generate_watermark(header['num_nodes'], wat_len, wat_type, wat_seed_value)
#                 res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_watermark(constraints, watermark_cons, watermark_nodes, header, params, file_name)

#                 #name= 'probs'+'_'+file_name+'.txt'
#                 #np.savetxt(name, probs)
#                 time = timeit.default_timer() - temp_time
#                 log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
#                 print(res)
#                 print(res_th)
#                 f.create_dataset(f"{file_name}", data = res)

