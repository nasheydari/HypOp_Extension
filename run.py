from src.run_exp import exp_centralized,  exp_centralized_for
from src.solver import QUBO_solver
import json



with open('configs/Hypermaxcut_syn_new_new.json') as f:
   params = json.load(f)
exp_centralized(params)




