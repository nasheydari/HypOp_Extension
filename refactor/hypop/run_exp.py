from hypop.data_reading import read_headers_constraints
from hypop.solver import  centralized_solver
from hypop.params import Params
import logging
import os
import h5py
import numpy as np

import timeit
import pydantic


def exp_centralized(params: Params):
    logging.basicConfig(filename=params.logging_path, filemode='w', level=logging.INFO)
    logger = logging.getLogger('main')
    folder_path = params.folder_path
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params.res_path, 'w') as f:
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                tick = timeit.default_timer()
                constraints, header = read_headers_constraints(path, params.data, logger)

                res, res_th, probs, total_time, train_time, map_time = centralized_solver(constraints, header, params, file_name)

                tock = timeit.default_timer()
                time = tock - tick
                logger.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, training_time: {train_time}, mapping_time: {map_time}')
                print(res)
                print(res_th)
                f.create_dataset(f"{file_name}", data = res)



