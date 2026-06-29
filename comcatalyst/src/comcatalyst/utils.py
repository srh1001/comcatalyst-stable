import numpy as np
import itertools
import json
import numpy as np
from pathlib import Path
from typing import List

from .context.vars_metadata import VarsMetadata

def build_grid(vars_metadata: VarsMetadata, n_grid: int) -> np.ndarray:
    categ_vars_indexes = vars_metadata.get_categ_like_vars_indexes()
    cont_vars_indexes  = vars_metadata.get_continuous_vars_indexes()

    categ_modalities = [
        list(range(len(vars_metadata[i].support)))
        for i in categ_vars_indexes
    ]
    cont_grids = [
        np.linspace(vars_metadata[i].support[0], vars_metadata[i].support[1], n_grid)
        for i in cont_vars_indexes
    ]

    rows = []
    for categ_combo in itertools.product(*categ_modalities):
        for cont_combo in itertools.product(*cont_grids):
            x_vec = np.full(len(vars_metadata), np.nan)
            for i, var_idx in enumerate(categ_vars_indexes):
                x_vec[var_idx] = categ_combo[i]
            for i, var_idx in enumerate(cont_vars_indexes):
                x_vec[var_idx] = cont_combo[i]
            rows.append(x_vec)

    return np.array(rows)

def calc_convergence_iter_bo(y_data: np.ndarray, n_init: int) -> int:
    return int(np.argmax(y_data)) + 1 - n_init