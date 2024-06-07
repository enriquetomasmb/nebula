import logging
import numpy as np
import os
import random
import torch


def enable_deterministic(config):
    seed = config.participant["scenario_args"]["random_seed"]
    logging.info("Fixing randomness with seed {}".format(seed))
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
