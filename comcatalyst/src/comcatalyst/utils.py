import numpy as np
import itertools

from .vars.vars_metadata import VarsMetadata

def gower_distance(x1: np.ndarray, x2: np.ndarray, vars_metadata: VarsMetadata) -> float:
    distances = []
    for i, meta in enumerate(vars_metadata):
        if meta.is_categorical():
            # 0 si même modalite, 1 sinon
            distances.append(0.0 if x1[i] == x2[i] else 1.0)
        elif meta.is_continuous():
            # distance normalisée par le range
            range_i = meta.support[1] - meta.support[0]
            distances.append(abs(x1[i] - x2[i]) / range_i)
        else:
            raise ValueError("Gower distance not currently defined for variables other than continous or categorical." \
                             f"Found variable of type: {meta.var_type}")
    return float(np.mean(distances))

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