from hypop.run_exp import exp_centralized
from hypop.params import Params
import json
from omegaconf import OmegaConf



conf_dict = OmegaConf.from_cli()


if "config" in conf_dict.keys():
    with open(conf_dict["config"]) as f:
        json_params = json.load(f)
        params = Params(**json_params)
        params = params.model_copy(update=conf_dict)
else:
    params = Params(**conf_dict)

print(params)

exp_centralized(params)