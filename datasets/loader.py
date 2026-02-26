import numpy as np
from pathlib import Path

def load(type):
    
    ROOT = Path(__file__).resolve().parents[1]

    if type == "Ca":
        DATA_PATH = ROOT / "datasets" / "Ca_stack.npy"
    elif type == "NE":
        DATA_PATH = ROOT / "datasets" / "NE_stack.npy"
    else:
        DATA_PATH = ROOT / "datasets" / "HbT_stack.npy"

    return np.load(DATA_PATH)

def sort_trials(type):
    import pandas as pd
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    log = pd.read_pickle(ROOT / "datasets" / "log.pkl")
    N = len(log)

    if type == "lomo":
        mouse = log["mouse_id"]
        MICE = list(set(mouse))

        indices = {}
        test_indices = []
        train_indices = []
        
        for m in MICE:
            indices[m] = []

        for i, m in enumerate(mouse):
            indices[m].append(i)

        for i, m in enumerate(indices):
            train_indices.append([x for x in range(N) if x not in indices[m]])
            test_indices.append([x for x in range(N) if x in indices[m]])

        return test_indices, train_indices