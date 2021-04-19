import random
import numpy as np
import torch

progress_sign = ["/", "-", "\\", "|"]


def fix_random_seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def log_print(*args, **kwargs):
    print("[Session Log]", end=" ")
    print(*args, **kwargs)
