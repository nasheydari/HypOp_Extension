import pydantic

class Params(pydantic.BaseModel):
    data: str = "hypergraph"
    mode: str = "maxcut"
    K: int = 1
    random_init: str = "none"
    transfer: bool = False
    initial_transfer: bool = False
    GD: bool = False
    Att: bool = False
    dropout: float = 0
    f_input: bool = False
    f: int = 70
    penalty_inc: bool = False
    penalty_c: float = 1e-5
    lr: float = 1e-2
    epoch: float = 1e2
    tol: float = 1e-4
    mapping: str = "distribution"
    boosting_mapping: int = 1
    logging_path: str = "./log/Hypermaxcut_syn_new_new.log"
    res_path: str = "./res/Hypermaxcut_syn_new_new.hdf5"
    folder_path: str = "../data/hypergraph_data/synthetic/new/single/"
    plot_path: str = "./res/plots/Hist_maxcut/"
    model_save_path: str = "./models/maxcut/"
    model_load_path: str = "./models/maxcut/"
    G_folder: str = "./models/G/"
    N_realize: int = 1
    Niter_h: int = 30
    t: float = 0.3
    hyper: bool = True
    penalty: int = 0
    patience: int = 50
    sparsify: bool = False
    spars_p: float = 0.8